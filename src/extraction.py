import torch
import numpy as np
from transformers import AutoModelForCausalLM



def split_activation(activations, config):
    """split the residual stream (d_model) into n_heads activations for each layer

    Args:
        activations (list[torch.Tensor]): list of residual streams for each layer shape: (n_layers, batch, seq, d_model)
        config (dict[str, Any]): model's config

    Returns:
        activations: reshaped activation in [n_layers, seq, n_heads, d_head]
    """
    new_shape = torch.Size([
        activations[0].shape[0],         # batch_size == 1
        activations[0].shape[1],         # seq_len
        config['n_heads'],                          # n_heads
        config['d_model'] // config['n_heads'],     # d_head
    ])
    attn_activations = torch.vstack([act.view(*new_shape) for act in activations])
    return attn_activations


def extract_activations(tokenized_prompts: list[torch.Tensor], model: AutoModelForCausalLM, config: dict[str, any]):
    """Extract the activation and the output produced from the model using the tokenized prompts provided

    Args:
        tokenized_prompts (list[torch.Tensor]): list of tokenized prompts
        model (AutoModelForCausalLM): HuggingFace model
        config (dict[str, any]): model's config

    Returns:
        tuple[list[torch.Tensor], list[torch.Tensor]]: tuple corresponding to the activations and the model output
    """
    dataset_activations, outputs = [], []
    for prompt in tokenized_prompts:
        with model.generate(max_new_tokens=3) as generator:
            # invoke works in a generation context, where operations on inputs and outputs are tracked
            with generator.invoke(prompt) as invoker:
                layer_attn_activations = []
                for layer in range(config['n_layer']):
                    layer_attn_activations.append(model.transformer.h[layer].attn.c_proj.output.save())
        outputs.append(generator.output)

        # get the values from the activations
        layer_attn_activations = [att.value for att in layer_attn_activations]
        
        # from hidden state split heads and permute: n_layers, tokens, n_heads, d_head -> n_layers, n_heads, tokens, d_head
        attn_activations = split_activation(layer_attn_activations, config).permute(0, 2, 1, 3)
        dataset_activations.append(attn_activations)
    return dataset_activations, outputs


def get_mean_activations(tokenized_prompts: list[torch.Tensor], important_ids: list[int], model: AutoModelForCausalLM, config: dict[str, any]):
    """Compute the average of all the model's activation on the provided prompts

    Args:
        tokenized_prompts (list[torch.Tensor]): list of tokenized prompts
        important_ids (list[int]): list of important indexes i.e. the tokens where the average is computed
        model (AutoModelForCausalLM): HuggingFace model
        config (dict[str, any]): model's config

    Returns:
        tuple[torch.Tensor, list[torch.Tensor]: mean of activations and list of tokenized output from the model
    """

    activations, outputs = extract_activations(tokenized_prompts=tokenized_prompts, model=model, config=config)

    # keep only important tokens
    activations_clean = [activations[i][:, :, important_ids[i], :] for i in range(len(activations))]
    mean_activations = torch.stack(activations_clean).mean(axis = 0)

    return mean_activations, outputs
