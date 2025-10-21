import os
import openai
import re
import json
import numpy as np
import pandas as pd
from collections import Counter
from typing import Any, Optional, Tuple, Dict, List, NamedTuple, Set
import scipy
import time
from pprint import pprint as pprint

try:
    from config import config
except ImportError:
    # Fallback if config not available
    config = None

# Load your API key from an environment variable or secret management service
if config:
    openai.api_key = config.openai.api_key
    model_to_use = config.openai.model
else:
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is not set.\n"
            "Please set it: export OPENAI_API_KEY='your-key-here'"
        )
    model_to_use = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

# Setup interface with language model

def gen_response(prompt: str = 'hi', max_tokens: int = 10, temperature: float = 0) -> Dict:
    """Generate response using OpenAI Chat Completion API.

    Args:
        prompt: The prompt to send to the model
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0 = deterministic)

    Returns:
        OpenAI API response object
    """
    response = openai.ChatCompletion.create(
        model=model_to_use,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response

def gen_response_text_with_backoff(
    prompt: str = 'hi',
    max_tokens: int = 10,
    temperature: float = 0,
    max_retries: int = 10
) -> Tuple[str, int]:
    """Generate response with exponential backoff retry logic.

    Args:
        prompt: The prompt to send to the model
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        max_retries: Maximum number of retry attempts

    Returns:
        Tuple of (response_text, tokens_used)

    Raises:
        Exception: If all retries fail
    """
    prompt_succeeded = False
    wait_time = 0.1
    retry_count = 0

    while not prompt_succeeded and retry_count < max_retries:
        try:
            response = gen_response(prompt, max_tokens, temperature=temperature)
            prompt_succeeded = True
        except (openai.error.APIError, openai.error.RateLimitError, openai.error.Timeout, openai.error.ServiceUnavailableError) as e:
            retry_count += 1
            print(f'  LM response failed: {e}. Retry {retry_count}/{max_retries}. Retrying after {wait_time} seconds...')
            if retry_count >= max_retries:
                raise Exception(f"Failed after {max_retries} retries: {e}")
            time.sleep(wait_time)
            wait_time += wait_time * 2  # exponential backoff

    response_text = response["choices"][0]["message"]["content"]
    used_tokens = response["usage"]["completion_tokens"]
    return response_text, used_tokens

def get_dict_items_sorted_by_decreasing_value(_dict):
    sort_inds = np.flip(np.argsort(list(_dict.values())))
    sorted_keys = np.array(list(_dict.keys()))[sort_inds]
    sorted_values = np.array(list(_dict.values()))[sort_inds]
    return sorted_keys, sorted_values
    
def display_dict_sorted_by_decreasing_value(_dict, print_num=10):
    sorted_keys, sorted_values = get_dict_items_sorted_by_decreasing_value(_dict)
    pprint(list(zip(sorted_keys, sorted_values))[0:print_num])
    