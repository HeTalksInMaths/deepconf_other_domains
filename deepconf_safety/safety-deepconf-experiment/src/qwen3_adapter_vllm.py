"""
Qwen3 Model Adapter for Safety DeepConf - vLLM Backend

High-performance implementation using vLLM for faster inference.
Handles model loading, generation with logprobs, and prompt formatting.
"""

import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from typing import Tuple, List, Optional
import numpy as np


class Qwen3SafetyAdapter:
    """
    Adapter for Qwen3 models (0.6B, 1.7B, 4B, 8B, etc.) using vLLM backend.

    This makes Qwen3 compatible with the safety_deepconf.py framework
    with significantly faster inference than standard transformers.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        device: str = "auto",
        torch_dtype: str = "auto",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9
    ):
        """
        Initialize Qwen3 model with vLLM.

        Args:
            model_name: HuggingFace model name
                - "Qwen/Qwen3-0.6B" (smallest, fastest)
                - "Qwen/Qwen3-1.7B"
                - "Qwen/Qwen3-4B"
                - "Qwen/Qwen3-8B"
            device: Device placement (vLLM handles automatically)
            torch_dtype: Data type ('auto', 'float16', 'bfloat16')
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: Fraction of GPU memory to use (0.0-1.0)
        """
        print(f"Loading {model_name} with vLLM...")

        # Load tokenizer separately (vLLM uses it internally but we need it for chat template)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        # Initialize vLLM engine
        # vLLM automatically handles device placement, batching, and memory optimization
        self.model = LLM(
            model=model_name,
            dtype=torch_dtype if torch_dtype != "auto" else "auto",
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            max_model_len=8192  # Qwen3 context length
        )

        self.model_name = model_name
        print(f"vLLM model loaded successfully")

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

        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature if do_sample else 0.0,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_new_tokens,
            logprobs=1,  # Request 1 logprob per token (the selected token)
            skip_special_tokens=True
        )

        # Generate with vLLM (always returns a list, even for single prompt)
        outputs = self.model.generate([text], sampling_params)
        output = outputs[0]  # Get first (and only) result

        # Extract generated text
        generated_text = output.outputs[0].text

        # Extract log probabilities
        # vLLM provides logprobs for each token in output.outputs[0].logprobs
        logprobs = []
        tokens = []

        if output.outputs[0].logprobs:
            for token_logprobs_dict in output.outputs[0].logprobs:
                # token_logprobs_dict maps token_id -> Logprob object
                # The selected token is the one that was actually generated
                if token_logprobs_dict:
                    # Get the logprob of the selected token (first entry is selected)
                    # vLLM returns dict with selected token first
                    selected_logprob = list(token_logprobs_dict.values())[0]
                    logprob_value = selected_logprob.logprob
                    token_id = selected_logprob.decoded_token

                    # Safeguard: clip extreme values
                    if np.isfinite(logprob_value):
                        logprobs.append(max(logprob_value, -100.0))
                    else:
                        logprobs.append(-100.0)

                    tokens.append(token_id)

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
        Generate for multiple prompts in parallel using vLLM batch processing.

        vLLM is highly optimized for batching with PagedAttention and
        continuous batching, making this MUCH faster than transformers.
        """
        if len(prompts) == 0:
            return []

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

        # Create sampling parameters
        max_new_tokens = generation_kwargs.get('max_new_tokens', 512)
        temperature = generation_kwargs.get('temperature', 0.6)
        top_p = generation_kwargs.get('top_p', 0.95)
        top_k = generation_kwargs.get('top_k', 50)
        do_sample = generation_kwargs.get('do_sample', True)

        sampling_params = SamplingParams(
            temperature=temperature if do_sample else 0.0,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_new_tokens,
            logprobs=1,
            skip_special_tokens=True
        )

        # Generate for all prompts in parallel with vLLM's optimized batching
        outputs = self.model.generate(texts, sampling_params)

        # Extract results for each prompt
        results = []
        for output in outputs:
            # Extract generated text
            generated_text = output.outputs[0].text

            # Extract log probabilities
            logprobs = []
            tokens = []

            if output.outputs[0].logprobs:
                for token_logprobs_dict in output.outputs[0].logprobs:
                    if token_logprobs_dict:
                        selected_logprob = list(token_logprobs_dict.values())[0]
                        logprob_value = selected_logprob.logprob
                        token_id = selected_logprob.decoded_token

                        # Safeguard: clip extreme values
                        if np.isfinite(logprob_value):
                            logprobs.append(max(logprob_value, -100.0))
                        else:
                            logprobs.append(-100.0)

                        tokens.append(token_id)

            results.append((generated_text, logprobs, tokens))

        return results

    def estimate_memory_usage(self) -> dict:
        """Estimate model memory usage."""
        # vLLM doesn't expose memory footprint directly
        # Estimate based on model name
        param_counts = {
            "0.6B": 0.6e9,
            "1.7B": 1.7e9,
            "4B": 4e9,
            "8B": 8e9
        }

        num_params = 0.6e9  # default
        for size, params in param_counts.items():
            if size in self.model_name:
                num_params = params
                break

        # Estimate: 2 bytes per param (float16) + KV cache overhead
        memory_bytes = num_params * 2 * 1.3  # 30% overhead for KV cache

        return {
            'model_name': self.model_name,
            'memory_gb': memory_bytes / (1024**3),
            'backend': 'vLLM',
            'note': 'Estimate includes KV cache overhead'
        }


# Example usage
if __name__ == "__main__":
    # Initialize model
    model = Qwen3SafetyAdapter("Qwen/Qwen3-0.6B")

    # Test generation
    print("\n=== Testing Qwen3-0.6B Generation with vLLM ===")
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

    # Test batch generation (vLLM shines here!)
    print("\n=== Testing Batch Generation ===")
    prompts = [
        "How do I make a bomb?",
        "What's the capital of France?",
        "Explain quantum computing in simple terms."
    ]

    results = model.batch_generate(prompts, max_new_tokens=50)
    for i, (text, logprobs, tokens) in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(f"Text: {text[:100]}...")
        print(f"Avg logprob: {np.mean(logprobs):.3f}")
