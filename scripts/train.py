import argparse
import json
import logging
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import torch
import torch.distributed as dist
from torch.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, Sampler
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.optimization import get_cosine_schedule_with_warmup

try:
    from liger_kernel.transformers import apply_liger_kernel_to_gemma3_text
except ImportError:  # pragma: no cover
    apply_liger_kernel_to_gemma3_text = None

apply_liger_kernel_to_gemma3_text()

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None


logger = logging.getLogger(__name__)


TASK_PROMPTS = {
    "normalization": "Instruct- Normalise the following text:\n",
    "transliteration": "Instruct- Transliterate the following text to Latin Script:\n",
    "punctuation": "Instruct- Punctuate the following text:\n",
}


def setup_logging(rank: int) -> None:
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distributed supervised fine-tuning for Gemma3.")
    parser.add_argument("--normalization_path", type=str, default=None, help="Path to normalization task JSONL file")
    parser.add_argument("--transliteration_path", type=str, default=None, help="Path to transliteration task JSONL file")
    parser.add_argument("--punctuation_path", type=str, default=None, help="Path to punctuation task JSONL file")
    parser.add_argument("--model_name", type=str, default="google/gemma-3-1b-pt", help="Model name or path for supervised fine-tuning")
    parser.add_argument("--save_dir", type=str, default="/projects/data/Embedding/IndicToolkit/training_code/ckpts", help="Directory to store checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint directory or file for resuming")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--grad_accum_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.95)
    parser.add_argument("--adam_eps", type=float, default=1e-8)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--max_input_length", type=int, default=2048)
    parser.add_argument("--max_output_length", type=int, default=2048)
    parser.add_argument("--max_sequence_length", type=int, default=4096)
    parser.add_argument("--num_workers", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--backend", type=str, default="nccl")
    parser.add_argument("--no-bf16", dest="bf16", action="store_false", help="Disable bfloat16 mixed precision (default: enabled)")
    parser.add_argument("--bf16", dest="bf16", action="store_true", help="Enable bfloat16 mixed precision (default: True)")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--log_interval", type=int, default=25)
    parser.add_argument("--permanent_save_steps", type=int, default=250)
    parser.add_argument("--rolling_checkpoint_interval", type=int, default=300)
    parser.add_argument("--save_total_limit", type=int, default=5, help="Maximum rolling checkpoints to keep")
    parser.add_argument("--wandb_project", type=str, default="NPT")
    parser.add_argument("--wandb_api_key", type=str, default="a64461a20dc8143b470d6382e18a463c33d5db5f")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument(
        "--punctuation_val_path",
        type=str,
        default="/projects/data/Embedding/IndicToolkit/datasets_final/data/train_set_punct_2.02M_ready.jsonl",
        help="Optional validation JSONL for punctuation",
    )
    parser.add_argument(
        "--transliteration_val_path",
        type=str,
        default="/projects/data/Embedding/IndicToolkit/datasets_final/data/merged_transliteration_sampled.jsonl",
        help="Optional validation JSONL for transliteration",
    )
    parser.add_argument("--validation_sample_size", type=int, default=128)
    parser.add_argument("--validation_interval_steps", type=int, default=200)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--eval_max_new_tokens", type=int, default=128)
    parser.add_argument("--validation_samples_to_log", type=int, default=2)
    parser.add_argument("--validation_temperature", type=float, default=0.0)
    parser.add_argument("--curriculum_learning", action="store_true", help="Enable curriculum learning: sort data by sequence length (small to large)")
    
    # Set bf16 default to True (can be overridden by --no-bf16)
    parser.set_defaults(bf16=True)
    
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_dist(rank: int, world_size: int, backend: str) -> None:
    if dist.is_initialized() or world_size <= 1:
        return
    dist.init_process_group(backend=backend, init_method="env://", rank=rank, world_size=world_size)


def cleanup_dist() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


class LengthSortedSampler(Sampler):
    """Sampler that sorts indices by sequence length (ascending) for curriculum learning."""
    
    def __init__(self, dataset: Dataset, lengths: List[int], num_replicas: int = 1, rank: int = 0, shuffle: bool = False, drop_last: bool = False):
        if num_replicas <= 0:
            raise ValueError(f"num_replicas must be positive, got {num_replicas}")
        if rank < 0 or rank >= num_replicas:
            raise ValueError(f"rank must be in [0, {num_replicas}), got {rank}")
        
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.epoch = 0
        
        # Sort indices by length (ascending: short to long)
        sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i])
        self.sorted_indices = sorted_indices
        
        # Calculate samples per replica
        if self.drop_last:
            self.num_samples = len(self.sorted_indices) // self.num_replicas
        else:
            self.num_samples = math.ceil(len(self.sorted_indices) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
    
    def __iter__(self):
        # Get indices for this replica
        indices = self.sorted_indices[self.rank:self.total_size:self.num_replicas]
        
        if self.shuffle:
            # Shuffle within length groups for better batching
            rng = random.Random(self.epoch + self.rank)
            rng.shuffle(indices)
        
        return iter(indices)
    
    def __len__(self):
        return self.num_samples
    
    def set_epoch(self, epoch: int) -> None:
        """Set epoch for reproducibility when shuffling."""
        self.epoch = epoch


class SupervisedTaskDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        compute_lengths: bool = False,
        max_examples: Optional[int] = None,
        sample_seed: Optional[int] = None,
    ) -> None:
        self.path = Path(jsonl_path).expanduser().resolve()
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset file {self.path} not found")

        self.examples: List[Dict[str, str]] = []
        self.lengths: Optional[List[int]] = [] if compute_lengths else None
        self.total_examples: int = 0

        if max_examples is not None and max_examples < 0:
            raise ValueError(f"max_examples must be non-negative, got {max_examples}")

        reservoir_size = max_examples
        reservoir_rng: Optional[random.Random] = None
        if reservoir_size is not None and reservoir_size > 0:
            reservoir_rng = random.Random(sample_seed) if sample_seed is not None else random.Random()
        
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at line {line_number} in {self.path}") from exc
                input_text = payload.get("input_text")
                output_text = payload.get("output_text")
                if input_text is None or output_text is None:
                    print(f"Invalid JSON at line {line_number} in {self.path}")
                    raise ValueError(
                        f"Both 'input_text' and 'output_text' must be present (line {line_number} in {self.path})"
                    )
                example = {"input_text": str(input_text), "output_text": str(output_text)}
                self.total_examples += 1

                if reservoir_size is None:
                    self.examples.append(example)
                    if self.lengths is not None:
                        self.lengths.append(self._estimate_token_length(example))
                    continue

                if reservoir_size == 0:
                    continue

                if len(self.examples) < reservoir_size:
                    self.examples.append(example)
                    if self.lengths is not None:
                        self.lengths.append(self._estimate_token_length(example))
                    continue

                assert reservoir_rng is not None  # for type checkers
                replace_idx = reservoir_rng.randint(0, self.total_examples - 1)
                if replace_idx < reservoir_size:
                    self.examples[replace_idx] = example
                    if self.lengths is not None:
                        self.lengths[replace_idx] = self._estimate_token_length(example)

        if not self.examples:
            raise ValueError(f"No examples loaded from {self.path}")
        
        if compute_lengths and self.lengths is None:
            # In cases where the dataset was populated without lengths (should not happen), recompute.
            self.lengths = [self._estimate_token_length(example) for example in self.examples]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, str]:
        return self.examples[index]

    @staticmethod
    def _estimate_token_length(example: Dict[str, str]) -> int:
        prompt_chars = len(example["input_text"])
        response_chars = len(example["output_text"])
        return (prompt_chars + response_chars) // 4


class SupervisedDataCollator:
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_input_length: int,
        max_output_length: int,
        max_sequence_length: int,
        task_prompts: Optional[Dict[str, str]] = None,
        debug_logging: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.max_sequence_length = max_sequence_length
        self.task_prompts = task_prompts or {}
        self.debug_logging = debug_logging
        self._collate_calls = 0
        if self.tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer must define a pad token. Set tokenizer.pad_token before creating the collator.")
        if self.tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must define an EOS token for supervised fine-tuning.")

    def __call__(self, batch: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        start_time = time.perf_counter() if self.debug_logging else None
        input_ids: List[torch.Tensor] = []
        labels: List[torch.Tensor] = []
        attention_masks: List[torch.Tensor] = []

        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id

        for example in batch:
            prompt = example["input_text"].strip()
            task_name = example.get("task_name")
            prefix = self.task_prompts.get(task_name, "")
            if prefix:
                prompt = prefix + prompt
            response = example["output_text"].strip()

            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            response_ids = self.tokenizer.encode(response, add_special_tokens=False)

            # Truncate both prompt and response to 2048 tokens if they exceed the limit
            max_prompt_tokens = 2048
            max_response_tokens = 2048

            if len(prompt_ids) > max_prompt_tokens:
                prompt_ids = prompt_ids[:max_prompt_tokens]

            if len(response_ids) > max_response_tokens:
                response_ids = response_ids[:max_response_tokens]

            prompt_ids = prompt_ids + [eos_token_id]
            if not response_ids or response_ids[-1] != eos_token_id:
                response_ids = response_ids + [eos_token_id]

            sequence_ids = prompt_ids + response_ids
            # Labels: -100 for prompt (ignore), actual tokens for response (predict)
            # The model internally shifts: logits[i] predicts labels[i+1]
            # So labels should be aligned: labels[i] = token we want to predict at position i
            sequence_labels = [-100] * len(prompt_ids) + response_ids

            # If total sequence exceeds max_sequence_length, truncate response tokens further
            if len(sequence_ids) > self.max_sequence_length:
                prompt_len = len(prompt_ids)
                max_response_len = self.max_sequence_length - prompt_len
                
                if max_response_len > 0:
                    # Truncate response to fit within max_sequence_length
                    sequence_ids = prompt_ids + response_ids[:max_response_len]
                    sequence_labels = [-100] * prompt_len + response_ids[:max_response_len]
                else:
                    # Prompt alone exceeds max_sequence_length - truncate prompt and remove all response
                    sequence_ids = prompt_ids[:self.max_sequence_length]
                    sequence_labels = [-100] * self.max_sequence_length

            attention = torch.ones(len(sequence_ids), dtype=torch.long)

            input_ids.append(torch.tensor(sequence_ids, dtype=torch.long))
            labels.append(torch.tensor(sequence_labels, dtype=torch.long))
            attention_masks.append(attention)

        # Dynamic padding: pad only to the longest sequence in this batch (not to max_sequence_length)
        # This is especially efficient with curriculum learning where batches have similar-length sequences
        padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
        padded_labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        padded_attention = pad_sequence(attention_masks, batch_first=True, padding_value=0)

        if self.debug_logging:
            self._collate_calls += 1
            max_seq_len = padded_input_ids.size(1)
            elapsed = time.perf_counter() - start_time if start_time is not None else 0.0
            logger.info(
                "[collator-debug] call=%d batch=%d max_seq=%d elapsed=%.3fs",
                self._collate_calls,
                len(batch),
                max_seq_len,
                elapsed,
            )

        return {
            "input_ids": padded_input_ids,
            "labels": padded_labels,
            "attention_mask": padded_attention,
        }


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler_state: Optional[Dict[str, Any]],
    step: int,
    args: argparse.Namespace,
    rng_states: Dict[str, Any],
    checkpoint_dir: Path,
    wandb_run_id: Optional[str],
    extra_state: Optional[Dict[str, Any]] = None,
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"checkpoint_step_{step}.pt"
    model_to_save = model.module if hasattr(model, "module") else model

    state = {
        "model": model_to_save.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler_state,
        "step": step,
        "args": vars(args),
        "rng_states": rng_states,
        "wandb_run_id": wandb_run_id,
    }
    if extra_state:
        state.update(extra_state)

    torch.save(state, checkpoint_path)
    return checkpoint_path


def save_rng_state() -> Dict[str, Any]:
    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def load_rng_state(rng_states: Dict[str, Any]) -> None:
    random.setstate(rng_states["python"])
    torch.set_rng_state(rng_states["torch"])
    torch.cuda.set_rng_state_all(rng_states["cuda"])


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: Path,
) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_to_load = model.module if hasattr(model, "module") else model
    model_to_load.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if checkpoint.get("rng_states"):
        load_rng_state(checkpoint["rng_states"])
    return checkpoint


def resolve_resume_checkpoint(resume_path: str) -> Path:
    candidate = Path(resume_path).expanduser().resolve()
    if candidate.is_file():
        return candidate
    if candidate.is_dir():
        matches = sorted(candidate.glob("checkpoint_step_*.pt"))
        if not matches:
            raise FileNotFoundError(f"No checkpoint_step_*.pt files found in {candidate}")
        return matches[-1]
    raise FileNotFoundError(f"Resume path {resume_path} not found")


def configure_wandb(args: argparse.Namespace, rank: int, resume_id: Optional[str]) -> Optional[str]:
    if args.wandb_project is None:
        return None
    if wandb is None:
        raise ImportError("wandb is not installed but wandb_project was provided")

    if args.wandb_api_key:
        os.environ.setdefault("WANDB_API_KEY", args.wandb_api_key)

    init_kwargs: Dict[str, Any] = {
        "project": args.wandb_project,
        "name": args.wandb_run_name,
        "config": vars(args),
        "resume": "allow" if resume_id else False,
        "id": resume_id,
    }

    if rank == 0:
        run = wandb.init(**init_kwargs)
        return run.id

    if resume_id:
        wandb.init(**init_kwargs)
        return resume_id

    return None


def prepare_dataloaders(
    args: argparse.Namespace,
    tokenizer: AutoTokenizer,
    rank: int,
    world_size: int,
) -> Dict[str, DataLoader]:
    task_paths = {
        "normalization": args.normalization_path,
        "transliteration": args.transliteration_path,
        "punctuation": args.punctuation_path,
    }

    collator = SupervisedDataCollator(
        tokenizer,
        max_input_length=args.max_input_length,
        max_output_length=args.max_output_length,
        max_sequence_length=args.max_sequence_length,
        task_prompts=TASK_PROMPTS,
    )
    dataloaders: Dict[str, DataLoader] = {}

    for task, path in task_paths.items():
        if path is None:
            continue

        # Create dataset with length computation if curriculum learning is enabled
        dataset = SupervisedTaskDataset(
            path,
            compute_lengths=args.curriculum_learning
        )
        for example in dataset.examples:
            example["task_name"] = task
        
        loader_kwargs: Dict[str, Any] = {
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "pin_memory": True,
            "collate_fn": collator,
            "drop_last": False,
        }
        if args.num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 4

        # Choose sampler based on curriculum learning and distributed training
        if args.curriculum_learning:
            if dataset.lengths is None:
                raise ValueError(f"Dataset lengths not computed for curriculum learning. Task: {task}")
            if world_size > 1:
                sampler = LengthSortedSampler(
                    dataset,
                    lengths=dataset.lengths,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=False,  # Already sorted, but can enable for within-group shuffling
                    drop_last=False
                )
            else:
                sampler = LengthSortedSampler(
                    dataset,
                    lengths=dataset.lengths,
                    num_replicas=1,
                    rank=0,
                    shuffle=False,
                    drop_last=False
                )
            loader_kwargs["sampler"] = sampler
            if rank == 0:
                logger.info(
                    "[curriculum-learning] task=%s enabled, sorted %d examples by length (min=%d, max=%d, avg=%.1f)",
                    task,
                    len(dataset),
                    min(dataset.lengths) if dataset.lengths else 0,
                    max(dataset.lengths) if dataset.lengths else 0,
                    sum(dataset.lengths) / len(dataset.lengths) if dataset.lengths else 0.0
                )
        else:
            if world_size > 1:
                sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
                loader_kwargs["sampler"] = sampler
            else:
                sampler = RandomSampler(dataset)
                loader_kwargs["sampler"] = sampler

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=loader_kwargs.get("sampler"),
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collator,
            drop_last=False,
            persistent_workers=args.num_workers > 0,
            prefetch_factor=loader_kwargs.get("prefetch_factor"),
        )

        dataloaders[task] = loader

        if rank == 0:
            logger.info(
                "[train-dataloader] task=%s path=%s dataset_size=%d batches=%d",
                task,
                path,
                len(dataset),
                math.ceil(len(dataset) / max(1, args.batch_size)),
            )

    return dataloaders


def prepare_validation_data(
    args: argparse.Namespace,
    tokenizer: AutoTokenizer,) -> Dict[str, Dict[str, Any]]:
    
    collator = SupervisedDataCollator(
        tokenizer,
        max_input_length=args.max_input_length,
        max_output_length=args.max_output_length,
        max_sequence_length=args.max_sequence_length,
        task_prompts=TASK_PROMPTS,
        debug_logging=False,
    )

    validation_specs = {
        "punctuation": args.punctuation_val_path,
        "transliteration": args.transliteration_val_path,
    }

    validation_data: Dict[str, Dict[str, Any]] = {}

    for task, path in validation_specs.items():
        if not path:
            continue
        sample_limit = max(0, args.validation_sample_size)
        if sample_limit == 0:
            logger.info(
                "[validation-setup] task=%s path=%s skipped (validation_sample_size=0)",
                task,
                path,
            )
            continue

        dataset = SupervisedTaskDataset(
            path,
            max_examples=sample_limit,
            sample_seed=args.seed + hash(task) % 10_000,
        )

        for example in dataset.examples:
            example["task_name"] = task

        loader_kwargs: Dict[str, Any] = {
            "batch_size": args.eval_batch_size,
            "shuffle": False,
            "num_workers": 0,
            "pin_memory": False,
            "collate_fn": collator,
            "drop_last": False,
        }

        loader = DataLoader(
            dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            collate_fn=collator,
            drop_last=False,
        )

        validation_data[task] = {
            "loader": loader,
            "dataset": dataset,
        }

        logger.info(
            "[validation-setup] task=%s path=%s sample_size=%d batches=%d",
            task,
            path,
            len(dataset.examples),
            math.ceil(len(dataset.examples) / max(1, args.eval_batch_size)),
        )

        if getattr(dataset, "total_examples", None) is not None and dataset.total_examples > len(dataset):
            logger.info(
                "[validation-setup] task=%s sampled %d of %d available examples",
                task,
                len(dataset),
                dataset.total_examples,
            )

    return validation_data


def generate_validation_sample(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    example: Dict[str, str],
    task: str,
    device: torch.device,
    args: argparse.Namespace,
) -> str:
    prompt_prefix = TASK_PROMPTS.get(task, "")
    prompt = f"{prompt_prefix}{example['input_text'].strip()}"
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    generation_kwargs = {
        "max_new_tokens": args.eval_max_new_tokens,
        "do_sample": args.validation_temperature > 0.0,
        "temperature": max(args.validation_temperature, 1e-5) if args.validation_temperature > 0.0 else None,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    # Only add no_repeat_ngram_size if it exists in args
    if hasattr(args, "no_repeat_ngram_size") and args.no_repeat_ngram_size is not None:
        generation_kwargs["no_repeat_ngram_size"] = max(0, args.no_repeat_ngram_size)
    generation_kwargs = {key: value for key, value in generation_kwargs.items() if value is not None}

    with torch.no_grad():
        if args.bf16 and device.type == "cuda":
            with autocast("cuda", dtype=torch.bfloat16):
                output_ids = model.generate(**inputs, **generation_kwargs)
        else:
            output_ids = model.generate(**inputs, **generation_kwargs)

    generated_sequence = output_ids[0]
    prompt_length = inputs["input_ids"].shape[-1]
    new_tokens = generated_sequence[prompt_length:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def run_validation(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    device: torch.device,
    validation_data: Dict[str, Dict[str, Any]],
    args: argparse.Namespace,
    global_step: int,
) -> None:
    if not validation_data:
        return

    was_training = model.training
    model.eval()

    for task, payload in validation_data.items():
        loader: DataLoader = payload["loader"]
        dataset = payload["dataset"]
        total_loss = 0.0
        total_examples = 0

        num_batches = len(loader)
        logger.info(
            "[validation] step=%s task=%s begin batches=%d",
            global_step,
            task,
            num_batches,
        )

        loader_iter = iter(loader)

        with torch.no_grad():
            for batch_idx in range(num_batches):
                logger.info(
                    "[validation] step=%s task=%s fetching batch %d/%d",
                    global_step,
                    task,
                    batch_idx + 1,
                    num_batches,
                )
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    logger.warning(
                        "[validation] step=%s task=%s exhausted iterator early at batch %d",
                        global_step,
                        task,
                        batch_idx,
                    )
                    break
                batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}

                # Use same forward pass as training
                outputs = model(**batch, return_dict=True)
                loss = outputs.loss

                # Debug: Verify loss calculation matches manual computation
                labels = batch["labels"]
                non_ignored_mask = (labels != -100)
                num_valid_tokens = non_ignored_mask.sum().item()
                
                # Manually compute loss to verify model's internal calculation
                if hasattr(outputs, "logits") and batch_idx == 0:  # Only for first batch to avoid overhead
                    logits = outputs.logits
                    # Standard CausalLM shifts: logits[i] predicts labels[i+1]
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    # Flatten for loss computation
                    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
                    flat_labels = shift_labels.view(-1)
                    # Compute per-token loss
                    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
                    per_token_loss = loss_fct(flat_logits, flat_labels)
                    valid_loss = per_token_loss[flat_labels != -100]
                    manual_loss = valid_loss.mean() if len(valid_loss) > 0 else torch.tensor(0.0, device=device)
                    
                    # Per-sequence statistics
                    batch_size = labels.size(0)
                    seq_lengths = (labels != -100).sum(dim=1).cpu().tolist()
                    total_lengths = (labels != -100).sum(dim=1).cpu().tolist()  # This is same as seq_lengths for valid tokens
                    attention_mask = batch.get("attention_mask")
                    if attention_mask is not None:
                        actual_lengths = attention_mask.sum(dim=1).cpu().tolist()
                    else:
                        actual_lengths = [labels.size(1)] * batch_size
                    
                    valid_ratio = num_valid_tokens / labels.numel() if labels.numel() > 0 else 0.0
                    avg_valid_per_seq = sum(seq_lengths) / batch_size if batch_size > 0 else 0.0
                    avg_seq_len = sum(actual_lengths) / batch_size if batch_size > 0 else 0.0
                    
                    logger.info(
                        "[validation-debug] step=%s task=%s batch=%d model_loss=%.4f manual_loss=%.4f diff=%.4f "
                        "valid_tokens=%d/%d (%.1f%%) avg_valid_per_seq=%.1f avg_seq_len=%.1f",
                        global_step, task, batch_idx,
                        float(loss.detach().cpu().item()),
                        float(manual_loss.detach().cpu().item()),
                        abs(float(loss.detach().cpu().item()) - float(manual_loss.detach().cpu().item())),
                        num_valid_tokens, labels.numel(), valid_ratio * 100,
                        avg_valid_per_seq, avg_seq_len
                    )

                batch_size = batch["input_ids"].size(0)
                total_loss += float(loss.detach().cpu().item()) * batch_size
                total_examples += batch_size

                logger.debug(
                    "[validation] step=%s task=%s batch=%d loss=%.4f",
                    global_step,
                    task,
                    batch_idx,
                    float(loss.detach().cpu().item()),
                )

                logger.info(
                    "[validation] step=%s task=%s finished batch %d/%d",
                    global_step,
                    task,
                    batch_idx + 1,
                    num_batches,
                )

        avg_loss = total_loss / max(1, total_examples)
        logger.info("[validation][step=%s][task=%s] loss=%.4f", global_step, task, avg_loss)
        if wandb and wandb.run:
            # Use async logging to avoid blocking
            wandb.log({f"val/{task}_loss": avg_loss, "step": global_step}, commit=False)

        samples_to_log = min(len(dataset), max(0, args.validation_samples_to_log))
        for idx in range(samples_to_log):
            example = dataset.examples[idx]
            logger.info(
                "[validation] step=%s task=%s generating sample idx=%d",
                global_step,
                task,
                idx,
            )
            prediction = generate_validation_sample(model, tokenizer, example, task, device, args)
            logger.info(
                "[validation][task=%s] sample %d\nPrompt: %s\nTarget: %s\nPrediction: %s",
                task,
                idx,
                example["input_text"],
                example["output_text"],
                prediction,
            )
            if wandb and wandb.run:
                wandb.log(
                    {
                        f"val/{task}_sample_{idx}/input": example["input_text"],
                        f"val/{task}_sample_{idx}/target": example["output_text"],
                        f"val/{task}_sample_{idx}/prediction": prediction,
                        "step": global_step,
                    },
                    commit=False
                )
    
    # Commit all validation logs at once
    if wandb and wandb.run:
        wandb.log({}, commit=True)

    if was_training:
        model.train()


def train(args: argparse.Namespace) -> None:
    
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_distributed = world_size > 1

    setup_logging(rank)
    init_dist(rank, world_size, args.backend)

    if not args.device.startswith("cuda"):
        raise RuntimeError("This script currently expects CUDA devices for training.")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required for supervised fine-tuning")

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    set_seed(args.seed + rank)

    if args.bf16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Requested bf16 but the current CUDA device does not support bfloat16")

    amp_dtype = torch.bfloat16 if args.bf16 else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
        token="",
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    elif tokenizer.pad_token_id != tokenizer.eos_token_id and tokenizer.eos_token_id is not None:
        # Ensure EOS exists for the collator; Gemma provides both by default.
        pass

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        attn_implementation="flash_attention_2",
        token="",
        dtype="bfloat16",
        trust_remote_code=args.trust_remote_code,
    )

    if args.gradient_checkpointing:
        gradient_kwargs = {"use_reentrant": False}
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_kwargs)
        except TypeError:
            model.gradient_checkpointing_enable()

    model = model.to(device)

    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )

    dataloaders = prepare_dataloaders(args, tokenizer, rank, world_size)
    if not dataloaders:
        raise ValueError(
            "At least one task path must be provided. Provide normalization_path, transliteration_path, or punctuation_path."
        )

    validation_data: Dict[str, Dict[str, Any]] = {}
    if rank == 0:
        validation_data = prepare_validation_data(args, tokenizer)

    batches_per_task = {task: len(loader) for task, loader in dataloaders.items()}
    if any(count == 0 for count in batches_per_task.values()):
        raise ValueError("All provided task dataloaders must contain at least one batch")

    total_batches_per_epoch = sum(batches_per_task.values())
    total_batches_target = total_batches_per_epoch * args.num_epochs
    total_optimizer_steps = max(1, math.ceil(total_batches_target / max(1, args.grad_accum_steps)))
    warmup_steps = max(1, int(total_optimizer_steps * args.warmup_ratio))

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optimizer_steps,
    )

    start_step = 0
    wandb_run_id: Optional[str] = None
    batches_already_processed = 0
    task_batch_counters: Dict[str, int] = {task: 0 for task in dataloaders.keys()}
    task_cycle_index = 0

    if args.resume:
        checkpoint_path = resolve_resume_checkpoint(args.resume)
        checkpoint = load_checkpoint(model, optimizer, checkpoint_path)
        start_step = checkpoint.get("step", 0)
        wandb_run_id = checkpoint.get("wandb_run_id")
        batches_already_processed = checkpoint.get("batches_processed", start_step * args.grad_accum_steps)
        saved_counters = checkpoint.get("task_batch_counters")
        if saved_counters:
            task_batch_counters.update({task: saved_counters[task] for task in dataloaders.keys() if task in saved_counters})
        task_cycle_index = checkpoint.get("task_cycle_index", 0)
        if checkpoint.get("scheduler"):
            scheduler.load_state_dict(checkpoint["scheduler"])

    wandb_run_id = configure_wandb(args, rank, wandb_run_id)

    loader_iters: Dict[str, Iterator] = {}
    sampler_epochs: Dict[str, int] = {}
    task_batch_limits = {task: count * args.num_epochs for task, count in batches_per_task.items()}

    for task, loader in dataloaders.items():
        completed_epochs = task_batch_counters[task] // batches_per_task[task]
        sampler_epochs[task] = completed_epochs
        sampler = getattr(loader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(completed_epochs)
        iterator = iter(loader)
        offset = task_batch_counters[task] % max(1, batches_per_task[task])
        for _ in range(offset):
            try:
                next(iterator)
            except StopIteration:
                completed_epochs += 1
                sampler_epochs[task] = completed_epochs
                if sampler is not None and hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(completed_epochs)
                iterator = iter(loader)
        loader_iters[task] = iterator

    task_cycle = [task for task in ["normalization", "transliteration", "punctuation"] if task in dataloaders]
    if task_cycle_index >= len(task_cycle):
        task_cycle_index = 0
    task_idx = task_cycle_index

    accumulation_steps = args.grad_accum_steps
    save_dir = Path(args.save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    steps_permanent_interval = args.permanent_save_steps or max(1, total_optimizer_steps // 20)

    log_interval = max(1, args.log_interval)

    def refresh_iterator(task_name: str) -> Iterator:
        loader = dataloaders[task_name]
        sampler_epochs[task_name] += 1
        sampler = getattr(loader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(sampler_epochs[task_name])
        return iter(loader)

    def fetch_batch(task_name: str) -> Dict[str, torch.Tensor]:
        iterator = loader_iters[task_name]
        while True:
            try:
                return next(iterator)
            except StopIteration:
                iterator = refresh_iterator(task_name)
                loader_iters[task_name] = iterator

    optimizer.zero_grad(set_to_none=True)

    global_step = start_step
    batches_processed = batches_already_processed
    pbar = None
    if rank == 0:
        pbar = tqdm(total=total_optimizer_steps, initial=global_step, dynamic_ncols=True, leave=True)
        pbar.set_description("Supervised Fine-Tuning")
    
    # Ensure model is in training mode
    model.train()
    
    # Track metrics per task for logging
    task_metrics: Dict[str, Dict[str, torch.Tensor]] = {
        task: {
            "loss_sum": torch.tensor(0.0, device=device),
            "loss_count": torch.tensor(0, device=device),
        }
        for task in dataloaders.keys()
    }
    
    # Store last batch and outputs for each task for prediction logging
    task_last_batch: Dict[str, Optional[Dict[str, torch.Tensor]]] = {
        task: None for task in dataloaders.keys()
    }
    task_last_outputs: Dict[str, Optional[Any]] = {
        task: None for task in dataloaders.keys()
    }

    try:
        while global_step < total_optimizer_steps and batches_processed < total_batches_target:
            selected_task = None
            for _ in range(len(task_cycle)):
                candidate = task_cycle[task_idx % len(task_cycle)]
                task_idx += 1
                if task_batch_counters[candidate] < task_batch_limits[candidate]:
                    selected_task = candidate
                    break
            if selected_task is None:
                break

            remaining_task_batches = task_batch_limits[selected_task] - task_batch_counters[selected_task]
            remaining_overall_batches = total_batches_target - batches_processed
            step_accum_target = min(accumulation_steps, remaining_task_batches, remaining_overall_batches)
            if step_accum_target <= 0:
                continue

            for _ in range(step_accum_target):
                batch = fetch_batch(selected_task)
                batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}

                with autocast("cuda", enabled=args.bf16, dtype=amp_dtype if args.bf16 else None):
                    outputs = model(**batch, return_dict=True)
                    loss = outputs.loss

                loss = loss / step_accum_target
                loss.backward()

                batches_processed += 1
                task_batch_counters[selected_task] += 1
                
                # Accumulate metrics for this task
                if hasattr(outputs, "loss") and isinstance(outputs.loss, torch.Tensor):
                    # Store the original loss (before division by step_accum_target)
                    task_metrics[selected_task]["loss_sum"] += outputs.loss.detach()
                    task_metrics[selected_task]["loss_count"] += 1
                
                # Store last batch and outputs for this task (for prediction logging)
                # Only store if we're close to a logging step to avoid memory overhead
                if (global_step + 1) % log_interval == 0:
                    task_last_batch[selected_task] = {k: v.detach().clone() for k, v in batch.items()}
                    if hasattr(outputs, "logits") and outputs.logits is not None:
                        task_last_outputs[selected_task] = outputs.logits.detach().clone()
                    else:
                        task_last_outputs[selected_task] = None

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if pbar is not None:
                pbar.update(1)

            should_log = (global_step % log_interval == 0) or (global_step == total_optimizer_steps)
            if should_log:
                # Capture grad_norm only when logging (avoid expensive CPU transfer on every step)
                # grad_norm is computed above, but we only transfer to CPU when logging
                if isinstance(grad_norm, torch.Tensor):
                    current_grad_norm = float(grad_norm.detach().cpu().item())
                else:
                    current_grad_norm = float(grad_norm)
                
                # Safety check: grad_norm should always be non-negative
                if current_grad_norm < 0 or not math.isfinite(current_grad_norm):
                    logger.warning(
                        "[grad-norm-warning] step=%s invalid grad_norm=%.2f, setting to 0.0",
                        global_step, current_grad_norm
                    )
                    current_grad_norm = 0.0
                
                logger.info("[logging] step=%s rank=%d starting log collection", global_step, rank)
                
                # Aggregate metrics across all tasks
                log_dict = {
                    "step": global_step,
                    "batches_processed": batches_processed,
                }
                
                logger.info("[logging] step=%s rank=%d computing task metrics", global_step, rank)
                
                # Compute and log metrics for each task
                for task in dataloaders.keys():
                    metrics = task_metrics[task]
                    
                    # Always create tensors for all_reduce, even if count is 0
                    # This ensures all ranks participate and prevents deadlocks
                    task_loss_tensor = torch.tensor(0.0, device=device)
                    
                    if metrics["loss_count"] > 0:
                        task_loss_tensor = metrics["loss_sum"] / metrics["loss_count"]
                    
                    # All ranks participate in all_reduce
                    if is_distributed and dist.is_initialized():
                        dist.all_reduce(task_loss_tensor, op=dist.ReduceOp.AVG)
                    
                    # Only log if we had data
                    if rank == 0:
                        if metrics["loss_count"] > 0:
                            log_dict[f"train/{task}_loss"] = float(task_loss_tensor)
                
                logger.info("[logging] step=%s rank=%d computing overall metrics", global_step, rank)
                
                # Compute overall averages across all tasks - all ranks participate
                total_loss_sum = sum(m["loss_sum"] for m in task_metrics.values())
                total_loss_count = sum(m["loss_count"] for m in task_metrics.values())
                
                # Convert to tensors for all_reduce
                overall_loss_tensor = torch.tensor(0.0, device=device)
                
                if total_loss_count > 0:
                    overall_loss_tensor = total_loss_sum / total_loss_count
                
                # All ranks participate in all_reduce for overall metrics
                if is_distributed and dist.is_initialized():
                    logger.info("[logging] step=%s rank=%d doing all_reduce for overall metrics", global_step, rank)
                    dist.all_reduce(overall_loss_tensor, op=dist.ReduceOp.AVG)
                    logger.info("[logging] step=%s rank=%d finished all_reduce for overall metrics", global_step, rank)
                
                if rank == 0:
                    if total_loss_count > 0:
                        log_dict["train/loss"] = float(overall_loss_tensor)
                    else:
                        logger.warning("[logging] step=%s no loss data accumulated", global_step)
                    
                    # Log per-task counts for debugging
                    for task in dataloaders.keys():
                        metrics = task_metrics[task]
                        logger.info(
                            "[logging] step=%s task=%s loss_count=%d",
                            global_step, task,
                            metrics["loss_count"].item()
                        )
                
                # Add learning rate and grad norm
                current_lr = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]["lr"]
                if rank == 0:
                    log_dict["learning_rate"] = current_lr
                
                # Use the captured grad_norm value - all ranks participate
                logger.info("[logging] step=%s rank=%d computing grad_norm", global_step, rank)
                grad_norm_tensor = torch.tensor(current_grad_norm, device=device)
                if is_distributed and dist.is_initialized():
                    dist.all_reduce(grad_norm_tensor, op=dist.ReduceOp.AVG)
                if rank == 0:
                    log_dict["grad_norm"] = float(grad_norm_tensor)
                
                # Decode and log predictions for one sample per task
                if rank == 0:
                    logger.info("[logging] step=%s rank=%d decoding predictions", global_step, rank)
                    for task in dataloaders.keys():
                        if task_last_batch[task] is not None and task_last_outputs[task] is not None:
                            batch = task_last_batch[task]
                            logits = task_last_outputs[task]
                            
                            # Get the first sample from the batch
                            sample_idx = 0
                            input_ids_sample = batch["input_ids"][sample_idx]
                            labels_sample = batch["labels"][sample_idx]
                            attention_mask_sample = batch.get("attention_mask", None)
                            if attention_mask_sample is not None:
                                attention_mask_sample = attention_mask_sample[sample_idx]
                            
                            # Decode the prompt (input_ids up to where response starts)
                            # Find where labels start (first non-(-100) label)
                            # Move to CPU for decoding
                            labels_sample_cpu = labels_sample.cpu()
                            input_ids_sample_cpu = input_ids_sample.cpu()
                            logits_sample_cpu = logits[sample_idx].cpu()
                            
                            valid_label_positions = (labels_sample_cpu != -100).nonzero(as_tuple=True)[0]
                            if len(valid_label_positions) > 0:
                                prompt_end_idx = valid_label_positions[0].item()
                                prompt_ids = input_ids_sample_cpu[:prompt_end_idx].tolist()
                                
                                # Decode prompt
                                prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=False)
                                
                                # Get predicted tokens from logits
                                # logits[i] predicts labels[i+1], so we need to shift
                                shift_logits = logits_sample_cpu[:-1]  # (seq_len-1, vocab_size)
                                predicted_token_ids = shift_logits.argmax(dim=-1)  # (seq_len-1,)
                                
                                # Extract only response predictions (where labels != -100)
                                shift_labels = labels_sample_cpu[1:]  # Shift labels to align with logits
                                response_mask = (shift_labels != -100)
                                
                                if response_mask.any():
                                    # Get predicted response tokens
                                    predicted_response_ids = predicted_token_ids[response_mask].tolist()
                                    
                                    # Get ground truth response tokens
                                    ground_truth_response_ids = shift_labels[response_mask].tolist()
                                    
                                    # Decode predictions and ground truth
                                    predicted_text = tokenizer.decode(predicted_response_ids, skip_special_tokens=True)
                                    ground_truth_text = tokenizer.decode(ground_truth_response_ids, skip_special_tokens=True)
                                    
                                    # Extract just the input text (without task prefix) for cleaner logging
                                    # The prompt includes the task prefix, so we extract the original input
                                    task_prefix = TASK_PROMPTS.get(task, "")
                                    if prompt_text.startswith(task_prefix):
                                        input_text_only = prompt_text[len(task_prefix):].strip()
                                    else:
                                        input_text_only = prompt_text.strip()
                                    
                                    # Log to wandb
                                    log_dict[f"train/{task}_sample/input"] = input_text_only
                                    log_dict[f"train/{task}_sample/prediction"] = predicted_text
                                    log_dict[f"train/{task}_sample/target"] = ground_truth_text
                                    
                                    # Log to terminal with clear formatting
                                    print(f"\n{'='*80}")
                                    print(f"[TRAIN SAMPLE] Step {global_step} | Task: {task}")
                                    print(f"{'='*80}")
                                    print(f"Input:    {input_text_only}")
                                    print(f"Target:   {ground_truth_text}")
                                    print(f"Prediction: {predicted_text}")
                                    print(f"{'='*80}\n")
                                    
                                    logger.info(
                                        "[train-sample] step=%s task=%s\nInput: %s\nTarget: %s\nPrediction: %s",
                                        global_step, task, input_text_only, ground_truth_text, predicted_text
                                    )

                if rank == 0:
                    logger.info("[logging] step=%s rank=%d updating progress bar and wandb", global_step, rank)
                    if pbar is not None:
                        # Show overall metrics in progress bar
                        loss_str = f"{log_dict.get('train/loss', 0.0):.4f}"
                        pbar.set_postfix(
                            loss=loss_str,
                            lr=f"{current_lr:.2e}",
                            grad=f"{log_dict['grad_norm']:.2f}",
                        )
                    if wandb and wandb.run:
                        # Use commit=False to make logging non-blocking
                        logger.info("[logging] step=%s rank=%d calling wandb.log", global_step, rank)
                        wandb.log(log_dict, commit=False)
                        # Flush periodically to avoid blocking
                        if global_step % (log_interval * 10) == 0:
                            wandb.log({}, commit=True)
                        logger.info("[logging] step=%s rank=%d finished wandb.log", global_step, rank)
                
                # Ensure all ranks are synchronized after logging before continuing
                logger.info("[logging] step=%s rank=%d entering barrier after logging", global_step, rank)
                if is_distributed and dist.is_initialized():
                    dist.barrier()
                logger.info("[logging] step=%s rank=%d exited barrier after logging", global_step, rank)
                
                # Reset task metrics for next logging interval
                for task in dataloaders.keys():
                    task_metrics[task] = {
                        "loss_sum": torch.tensor(0.0, device=device),
                        "loss_count": torch.tensor(0, device=device),
                    }
                    # Clear stored batches to free memory
                    task_last_batch[task] = None
                    task_last_outputs[task] = None

            should_validate = (
                args.validation_interval_steps > 0
                and global_step % args.validation_interval_steps == 0
            )
            if should_validate:
                logger.info(
                    "[validation] step=%s entering barrier rank=%d",
                    global_step,
                    rank,
                )
                if is_distributed and dist.is_initialized():
                    dist.barrier()
                
                # Only rank 0 runs validation
                if rank == 0 and validation_data:
                    logger.info("[validation] step=%s rank0 starting", global_step)
                    try:
                        eval_model = model.module if hasattr(model, "module") else model
                        run_validation(eval_model, tokenizer, device, validation_data, args, global_step)
                    finally:
                        # Ensure model is back in training mode
                        model.train()
                        logger.info("[validation] step=%s rank0 finished", global_step)
                elif rank == 0:
                    logger.info("[validation] step=%s rank0 skipped (no validation data)", global_step)
                
                # All ranks wait here before continuing training
                if is_distributed and dist.is_initialized():
                    dist.barrier()
                
                # Ensure ALL ranks have model in training mode (not just rank 0)
                # This is critical for distributed training
                model.train()
                
                logger.info(
                    "[validation] step=%s exited barrier rank=%d model.training=%s",
                    global_step,
                    rank,
                    model.training,
                )

            extra_state = {
                "batches_processed": batches_processed,
                "task_batch_counters": task_batch_counters,
                "task_cycle_index": task_idx,
            }

            if global_step % args.rolling_checkpoint_interval == 0 and rank == 0:
                rolling_dir = save_dir / "rolling"
                checkpoint_path = save_checkpoint(
                    model,
                    optimizer,
                    scheduler.state_dict(),
                    global_step,
                    args,
                    save_rng_state(),
                    rolling_dir,
                    wandb_run_id,
                    extra_state,
                )
                rolling_files = sorted(rolling_dir.glob("checkpoint_step_*.pt"))
                if len(rolling_files) > args.save_total_limit:
                    for stale in rolling_files[:-args.save_total_limit]:
                        stale.unlink(missing_ok=True)

            if global_step % steps_permanent_interval == 0 and rank == 0:
                permanent_dir = save_dir / "permanent"
                save_checkpoint(
                    model,
                    optimizer,
                    scheduler.state_dict(),
                    global_step,
                    args,
                    save_rng_state(),
                    permanent_dir,
                    wandb_run_id,
                    extra_state,
                )

        if rank == 0:
            final_dir = save_dir / "final"
            save_checkpoint(
                model,
                optimizer,
                scheduler.state_dict(),
                global_step,
                args,
                save_rng_state(),
                final_dir,
                wandb_run_id,
                {
                    "batches_processed": batches_processed,
                    "task_batch_counters": task_batch_counters,
                    "task_cycle_index": task_idx,
                },
            )

    finally:
        if pbar is not None:
            pbar.close()
        cleanup_dist()


def run_training() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    run_training()

