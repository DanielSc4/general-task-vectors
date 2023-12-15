import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import random


def load_gpt_model_and_tokenizer(
        model_name: str,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(model_name)

    # providing a standard config
    std_CONFIG = {
        'n_heads': model.config.n_head,
        'n_layer': model.config.n_layer,
        'd_model': model.config.n_embd,     # residual stream
        'name': model.config.name_or_path,
        'attn_hook_names': [
            f'transformer.h.{layer}.attn.c_proj' for layer in range(model.config.n_layer)
        ],
        'layer_hook_names': [
            f'transformer.h.{layer}' for layer in range(model.config.n_layer)
        ],
    }

    return model, tokenizer, std_CONFIG