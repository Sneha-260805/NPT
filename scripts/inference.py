import argparse
import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList

# --- Constants ---
TASK_PROMPTS = {
    "normalization": "Instruct- Normalise the following text:\n",
    "transliteration": "Instruct- Transliterate the following text to Latin Script:\n",
    "punctuation": "Instruct- Punctuate the following text:\n",
}

# --- Configuration ---
@dataclass
class ModelConfig:
    """Configuration class."""
    model_name: str = "google/gemma-3-1b-pt"
    # Parameters tailored for the specific weights provided in original script
    bf16: bool = True
    attn_impl: str = "sdpa"
    trust_remote_code: bool = False
    
    # Generation Parameters
    max_new_tokens: int = 512
    num_beams: int = 10
    temperature: float = 0.0
    top_p: float = 0.9
    top_k: int = 0
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 2.0
    no_repeat_ngram_size: int = 4

# --- Custom Classes ---
class PenaltyLogitsProcessor(LogitsProcessor):
    def __init__(self, frequency_penalty: float = 0.0, presence_penalty: float = 0.0) -> None:
        if frequency_penalty < 0.0 or presence_penalty < 0.0:
            raise ValueError("Penalties must be non-negative")
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if (self.frequency_penalty == 0.0 and self.presence_penalty == 0.0) or input_ids.numel() == 0:
            return scores

        for batch_idx in range(input_ids.size(0)):
            sequence = input_ids[batch_idx]
            if sequence.numel() == 0:
                continue

            unique_tokens, counts = torch.unique(sequence, return_counts=True)
            if self.frequency_penalty != 0.0:
                scores[batch_idx, unique_tokens] -= counts.to(scores.dtype) * self.frequency_penalty
            if self.presence_penalty != 0.0:
                scores[batch_idx, unique_tokens] -= self.presence_penalty

        return scores

# --- Helper Functions ---
def load_model_and_tokenizer(token: Optional[str]):
    config = ModelConfig()
    
    print("⏳ Downloading/Locating checkpoint...", file=sys.stderr)
    try:
        checkpoint_path = hf_hub_download(
            repo_id="SSneha2005/NPT",
            filename="weights/checkpoint_step_32000.pt",
            repo_type="dataset",
            token=token
        )
    except Exception as e:
        print(f"Error downloading checkpoint: {e}", file=sys.stderr)
        sys.exit(1)

    print("⏳ Loading Model...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        use_fast=True,
        token=token,
        trust_remote_code=config.trust_remote_code,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.bfloat16 if config.bf16 else torch.float32
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch_dtype,
        attn_implementation=config.attn_impl,
        trust_remote_code=config.trust_remote_code,
        token=token,
        device_map="auto"
    )

    # Load custom weights
    checkpoint = torch.load(Path(checkpoint_path).expanduser().resolve(), map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    return model, tokenizer, config

def generate_text(model, tokenizer, config, prompt):
    device = model.device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    logits_processor = LogitsProcessorList()
    if config.frequency_penalty != 0.0 or config.presence_penalty != 0.0:
        logits_processor.append(
            PenaltyLogitsProcessor(
                frequency_penalty=config.frequency_penalty,
                presence_penalty=config.presence_penalty,
            )
        )

    generation_kwargs = {
        "max_new_tokens": config.max_new_tokens,
        "num_beams": max(1, config.num_beams),
        "do_sample": config.temperature > 0.0,
        "temperature": max(config.temperature, 1e-5) if config.temperature > 0.0 else None,
        "top_p": config.top_p,
        "top_k": config.top_k,
        "repetition_penalty": max(0.0, config.repetition_penalty),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "no_repeat_ngram_size": max(0, config.no_repeat_ngram_size),
    }
    generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs, 
            logits_processor=logits_processor, 
            **generation_kwargs
        )

    # Slice output to remove input prompt
    generated_sequence = output_ids[0]
    prompt_length = inputs["input_ids"].shape[-1]
    new_tokens = generated_sequence[prompt_length:]
    
    output_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    
    # Clean up "assistant:" artifact if present
    if output_text.lower().startswith("assistant:"):
        output_text = output_text[len("assistant:"):].lstrip()
        
    return output_text

# --- Main CLI ---
def main():
    parser = argparse.ArgumentParser(description="Run inference on Gemma-3 NPT model.")
    parser.add_argument("--text", type=str, required=True, help="The input text to process.")
    parser.add_argument("--task", type=str, required=True, choices=TASK_PROMPTS.keys(), 
                        help="Task type: normalization, transliteration, or punctuation.")
    parser.add_argument("--token", type=str, default=None, 
                        help="Hugging Face API token (optional if env var set).")
    
    args = parser.parse_args()

    # Handle Token
    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    
    # Load resources
    model, tokenizer, config = load_model_and_tokenizer(token)
    
    # Format Prompt
    prefix = TASK_PROMPTS[args.task]
    formatted_prompt = f"{prefix}{args.text.strip()}\n\nassistant: "

    # Run Inference
    print("🚀 Generating...", file=sys.stderr)
    result = generate_text(model, tokenizer, config, formatted_prompt)
    
    # Print only the result to stdout
    print(result)

if __name__ == "__main__":
    main()
