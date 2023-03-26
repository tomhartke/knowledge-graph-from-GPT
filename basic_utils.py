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

# Load your API key from an environment variable or secret management service
openai.api_key = os.getenv("OPENAI_API_KEY") 
model_to_use =  "text-davinci-003"
    # "gpt-3.5-turbo-0301" is more moderns
    # "text-curie-001" is 10x cheaper than "text-davinci-003", but not as good
    # "text-davinci-002" is faster than 3, but not necessarily worse at explaining?
    
# Setup interface with language model

def gen_response(prompt='hi', max_tokens=10, temperature=0):
    response = openai.Completion.create(model=model_to_use, prompt=prompt, temperature=temperature, max_tokens=max_tokens)
    return response

def gen_response_text_with_backoff(prompt='hi', max_tokens=10, temperature=0):
    prompt_succeeded = False
    wait_time = 0.1
    while not prompt_succeeded:
        try:
            response = gen_response(prompt, max_tokens, temperature=temperature)
            prompt_succeeded = True
        except:
            print('  LM response failed. Server probably overloaded. Retrying after ', wait_time, ' seconds...')
            time.sleep(wait_time)
            wait_time += wait_time*2  # exponential backoff 
    response_text = str(response["choices"][0]["text"])
    used_tokens = response["usage"]["completion_tokens"]  # ["total_tokens"], i think completion is what matters
    return response_text, used_tokens

def get_dict_items_sorted_by_decreasing_value(_dict):
    sort_inds = np.flip(np.argsort(list(_dict.values())))
    sorted_keys = np.array(list(_dict.keys()))[sort_inds]
    sorted_values = np.array(list(_dict.values()))[sort_inds]
    return sorted_keys, sorted_values
    
def display_dict_sorted_by_decreasing_value(_dict, print_num=10):
    sorted_keys, sorted_values = get_dict_items_sorted_by_decreasing_value(_dict)
    pprint(list(zip(sorted_keys, sorted_values))[0:print_num])
    