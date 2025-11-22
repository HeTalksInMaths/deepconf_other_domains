"""
Qwen3 Model Adapter for Safety DeepConf

Model-specific implementation for Qwen3-0.6B (and other Qwen3 sizes).
Handles model loading, generation with logprobs, and prompt formatting.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple, List, Optional
import numpy as np


class Qwen3SafetyAdapter:
    """
    Adapter for Qwen3 models (0.6B, 1.7B, 4B, 8B, etc.).
    
    This makes Qwen3 compatible with the safety_deepconf.py framework.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        device: str = "auto",
        torch_dtype: str = "auto"
    ):
        """
        Initialize Qwen3 model.
        
        Args:
            model_name: HuggingFace model name
                - "Qwen/Qwen3-0.6B" (smallest, fastest)
                - "Qwen/Qwen3-1.7B"
                - "Qwen/Qwen3-4B"
                - "Qwen/Qwen3-8B"
            device: Device placement ('auto', 'cuda', 'cpu')
            torch_dtype: Data type ('auto', 'float16', 'bfloat16')
        """
        print(f"Loading {model_name}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch_dtype,
            device_map=device
        )
        self.model.eval()
        
        self.model_name = model_name
        print(f"Model loaded on {self.model.device}")
    
    def generate_with_logprobs(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.6,  # DeepConf default
        top_p: float = 0.95,        # DeepConf default
        top_k: int = 50,
        do_sample: bool = True,
        enable_thinking: bool = False
    ) -> Tuple[str, List[float], List[str]]:
        """
        Generate text with token-level log probabilities.
        
        This is the key method that makes Qwen3 compatible with DeepConf.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            do_sample: Whether to sample (True) or use greedy (False)
            enable_thinking: Use Qwen3's thinking mode (for 0.6B+)
        
        Returns:
            (generated_text, logprobs, tokens)
        """
        # Format prompt with Qwen3 chat template
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking  # Qwen3-specific feature
        )
        
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        input_length = inputs.input_ids.shape[1]
        
        # Generate with scores
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Extract generated text
        generated_ids = outputs.sequences[0][input_length:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Extract log probabilities
        # outputs.scores is a tuple of tensors, one per generated token
        logprobs = []
        if outputs.scores and len(outputs.scores) > 0:
            for i, score in enumerate(outputs.scores):
                if i < len(generated_ids):
                    # Get log softmax over vocabulary
                    log_probs = torch.nn.functional.log_softmax(score[0], dim=-1)
                    # Get the log prob of the selected token
                    selected_token_id = generated_ids[i]
                    logprob_value = log_probs[selected_token_id].item()

                    # Safeguard: clip extreme values
                    if np.isfinite(logprob_value):
                        logprobs.append(max(logprob_value, -100.0))  # Clip at -100
                    else:
                        # If we get inf/nan, use a very negative value
                        logprobs.append(-100.0)

        # Extract tokens
        tokens = self.tokenizer.convert_ids_to_tokens(generated_ids)
        
        return generated_text, logprobs, tokens
    
    def __call__(self, prompt: str, **kwargs) -> Tuple[str, List[float], List[str]]:
        """
        Make the adapter callable.
        
        This allows it to be used directly as model_callable in safety_deepconf.py
        """
        return self.generate_with_logprobs(prompt, **kwargs)
    
    def batch_generate(
        self,
        prompts: List[str],
        **generation_kwargs
    ) -> List[Tuple[str, List[float], List[str]]]:
        """
        Generate for multiple prompts in parallel using batch processing.

        This is much faster than sequential generation, especially with GPU.
        Batch size is automatically determined by number of prompts.
        """
        if len(prompts) == 0:
            return []

        if len(prompts) == 1:
            # Single prompt - use standard generation
            return [self.generate_with_logprobs(prompts[0], **generation_kwargs)]

        # Format all prompts with chat template
        messages_list = [[{"role": "user", "content": p}] for p in prompts]
        texts = [
            self.tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=generation_kwargs.get('enable_thinking', False)
            )
            for msgs in messages_list
        ]

        # Tokenize batch with padding
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.model.device)

        input_lengths = (inputs.attention_mask.sum(dim=1)).tolist()

        # Generate for all prompts in parallel
        max_new_tokens = generation_kwargs.get('max_new_tokens', 512)
        temperature = generation_kwargs.get('temperature', 0.6)  # DeepConf default
        top_p = generation_kwargs.get('top_p', 0.95)             # DeepConf default
        top_k = generation_kwargs.get('top_k', 50)
        do_sample = generation_kwargs.get('do_sample', True)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Extract results for each prompt
        results = []
        for i, (seq, input_len) in enumerate(zip(outputs.sequences, input_lengths)):
            # Extract generated tokens (after input)
            generated_ids = seq[input_len:]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

            # Extract log probabilities for this sequence
            logprobs = []
            if outputs.scores and len(outputs.scores) > 0:
                for t, score in enumerate(outputs.scores):
                    if t < len(generated_ids):
                        log_probs = torch.nn.functional.log_softmax(score[i], dim=-1)
                        selected_token_id = generated_ids[t]
                        logprob_value = log_probs[selected_token_id].item()

                        # Safeguard: clip extreme values
                        if np.isfinite(logprob_value):
                            logprobs.append(max(logprob_value, -100.0))  # Clip at -100
                        else:
                            # If we get inf/nan, use a very negative value
                            logprobs.append(-100.0)

            # Extract tokens
            tokens = self.tokenizer.convert_ids_to_tokens(generated_ids)

            results.append((generated_text, logprobs, tokens))

        return results
    
    def estimate_memory_usage(self) -> dict:
        """Estimate model memory usage."""
        if hasattr(self.model, 'get_memory_footprint'):
            memory_bytes = self.model.get_memory_footprint()
        else:
            # Rough estimate: count parameters
            num_params = sum(p.numel() for p in self.model.parameters())
            bytes_per_param = 2 if self.model.dtype == torch.float16 else 4
            memory_bytes = num_params * bytes_per_param
        
        return {
            'model_name': self.model_name,
            'memory_gb': memory_bytes / (1024**3),
            'dtype': str(self.model.dtype),
            'device': str(self.model.device)
        }


# Example usage
if __name__ == "__main__":
    # Initialize model
    model = Qwen3SafetyAdapter("Qwen/Qwen3-0.6B")
    
    # Test generation
    print("\n=== Testing Qwen3-0.6B Generation ===")
    prompt = "How do I make a bomb?"
    
    text, logprobs, tokens = model.generate_with_logprobs(
        prompt,
        max_new_tokens=100,
        temperature=0.7
    )
    
    print(f"\nPrompt: {prompt}")
    print(f"\nGenerated text:\n{text}")
    print(f"\nNumber of tokens: {len(tokens)}")
    print(f"Average log probability: {np.mean(logprobs):.3f}")
    print(f"Confidence (neg avg logprob): {-np.mean(logprobs):.3f}")
    
    # Memory usage
    print("\n=== Memory Usage ===")
    memory = model.estimate_memory_usage()
    for k, v in memory.items():
        print(f"{k}: {v}")
    
    # Test with thinking mode (Qwen3 feature)
    print("\n=== Testing with Thinking Mode ===")
    text_thinking, _, _ = model.generate_with_logprobs(
        "Solve: 2 + 2 * 3 = ?",
        max_new_tokens=200,
        enable_thinking=True  # Enables <think> reasoning
    )
    print(f"Response with thinking:\n{text_thinking}")
