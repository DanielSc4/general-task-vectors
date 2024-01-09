import torch
import numpy as np
from nnsight import LanguageModel
import os
import random
import functools
# thanks to https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties
def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))




def load_gpt_model_and_tokenizer(
        model_name: str,
):
    
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    if model_name == 'gpt2':
        model = LanguageModel('gpt2', device_map=device)
        # providing a standard config
        std_CONFIG = {
            'n_heads': model.config.n_head,
            'n_layers': model.config.n_layer,
            'd_model': model.config.n_embd,     # residual stream
            'name': model.config.name_or_path,
            'vocab_size': model.config.vocab_size,
            'layer_name': 'transformer.h',
            'layer_hook_names': [
                f'transformer.h.{layer}' for layer in range(model.config.n_layer)
            ],
            'attn_name': 'attn.c_proj',
            'attn_hook_names': [
                f'transformer.h.{layer}.attn.c_proj' for layer in range(model.config.n_layer)
            ],
        }
    
    elif 'gpt-neox' in model_name.lower():
        std_CONFIG = {
            'n_heads': model.config.num_attention_heads,
            'n_layers': model.config.num_hidden_layers,
            'd_model': model.config.name_or_path,     # residual stream
            'name': model.config.name_or_path,
            'vocab_size': model.config.vocab_size,
            'layer_name': 'gpt_neox.layers',
            'layer_hook_names': [
                f'gpt_neox.layers.{layer}' for layer in range(model.config.num_hidden_layers)
            ],
            'attn_name': 'attention.dense',
            'attn_hook_names': [
                f'gpt_neox.layers.{layer}.attention.dense' for layer in range(model.config.num_hidden_layers)
            ],
        }

    elif 'llama' in model_name.lower():
        std_CONFIG = {
            'n_heads': model.config.num_attention_heads,
            'n_layers': model.config.num_hidden_layers,
            'd_model': model.config.hidden_size,     # residual stream
            'name': model.config._name_or_path,
            'vocab_size': model.config.vocab_size,
            'layer_name': 'model.layers',
            'layer_hook_names': [
                f'model.layers.{layer}' for layer in range(model.config.num_hidden_layers)
            ],
            'attn_name': 'self_attn.o_proj',
            'attn_hook_names': [
                f'model.layers.{layer}.self_attn.o_proj' for layer in range(model.config.num_hidden_layers)
            ],
        }

    elif 'phi-2' in model_name.lower():
        std_CONFIG = {
            'n_heads': model.config.n_head,
            'n_layers': model.config.n_layer,
            'd_model': model.config.n_embd,     # residual stream
            'name': model.config._name_or_path,
            'vocab_size': model.config.vocab_size,
            'layer_name': 'transformer.h',
            'layer_hook_names': [
                f'transformer.h.{layer}' for layer in range(model.config.n_layer)
            ],
            'attn_name': 'mixer.out_proj',
            'attn_hook_names': [
                f'model.layers.{layer}.mixer.out_proj' for layer in range(model.config.num_hidden_layers)
            ],
        }

    else:
        raise NotImplementedError("Model config not yet implemented")

    return model, std_CONFIG


def set_seed(seed: int) -> None:
    """
    Sets the seed to make everything deterministic, for reproducibility of experiments

    Parameters:
    seed: the number to set the seed to

    Return: None
    """

    # Random seed
    random.seed(seed)

    # Numpy seed
    np.random.seed(seed)

    # Torch seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # os seed
    os.environ['PYTHONHASHSEED'] = str(seed)