import torch
import json
import numpy as np

from typing import Any
from nnsight import LanguageModel
from transformers import PreTrainedTokenizer
from tqdm import tqdm

from src.utils.eval.multi_token_evaluator import Evaluator
from src.utils.prompt_helper import pad_input_and_ids
from .utils.model_utils import rgetattr
from .utils.prompt_helper import find_missing_ranges



def filter_activations(activation, important_ids):
    """
    TODO: Deprecated
    Average activations of multi-token words across all its tokens
    """
    to_avg = find_missing_ranges(important_ids)
    for i, j in to_avg:
        activation[:, :, j] = activation[:, :, i : j + 1].mean(axis = 2)

    activation = activation[:, :, important_ids]
    return activation


def split_activation(activations, config):
    """split the residual stream (d_model) into n_heads activations for each layer

    Args:
        activations (list[torch.Tensor]): list of residual streams for each layer shape: (list: n_layers[tensor: batch, seq, d_model])
        config (dict[str, Any]): model's config

    Returns:
        activations: reshaped activation in [batch, n_layers, n_heads, seq, d_head]
    """
    new_shape = torch.Size([
        activations[0].shape[0],         # batch_size
        activations[0].shape[1],         # seq_len
        config['n_heads'],                          # n_heads
        config['d_model'] // config['n_heads'],     # d_head
    ])
    attn_activations = torch.stack([act.view(*new_shape) for act in activations])
    attn_activations = torch.einsum("lbshd -> blhsd", attn_activations)     # layers batch seq heads dhead -> batch layers heads seq dhead
    return attn_activations


def extract_activations(
        tokenized_prompts: dict[str, torch.Tensor] | torch.Tensor, 
        model: LanguageModel, 
        config: dict[str, Any],
        tokenizer: PreTrainedTokenizer,
        last_token_only: bool = False,
    ):
    """Extract the activation and the output produced from the model using the tokenized prompts provided

    Args:
        tokenized_prompts (dict[str, torch.Tensor]): input for the model (input_ids and attention mask)
        model (AutoModelForCausalLM): HuggingFace model
        config (dict[str, any]): model's config
        tokenizer (AutoTokenizer): HuggingFace tokenizer
        device (str): device

    Returns:
        tuple[list[torch.Tensor], torch.Tensor]: tuple corresponding to the activations (batch, n_layers, n_heads, seq, d_head) and the model output [batch, seq]
    """

    with model.generate(
        max_new_tokens=50,
        pad_token_id=tokenizer.pad_token_id,
    ) as generator:
        with generator.invoke(tokenized_prompts) as invoker:
            layer_attn_activations = []
            for layer_i in range(len(config['attn_hook_names'])):
                # pbar.set_description(f"[x] Extracting activations (layer: {layer_i:02d}/{len(config['attn_hook_names'])})")
                layer_attn_activations.append(
                    rgetattr(model, config['attn_hook_names'][layer_i]).input[0][0].save()       # before: *_i]).output.save() torch.size(batch, seq, hidden) 
                )
    # get the values from the activations
    layer_attn_activations = [att.value for att in layer_attn_activations]

    if not isinstance(layer_attn_activations[0], torch.Tensor):
        # take the first element (should be the hidden tensor according to 
        # https://github.com/huggingface/transformers/blob/224ab70969d1ac6c549f0beb3a8a71e2222e50f7/src/transformers/models/gpt2/modeling_gpt2.py#L341)
        layer_attn_activations = [att[0] for att in layer_attn_activations]

    output = generator.output   # contains also the prompt

    # from hidden state split heads and permute: batch, n_layers, tokens, n_heads, d_head -> batch, n_layers, n_heads, tokens, d_head
    attn_activations = split_activation(layer_attn_activations, config)
    if last_token_only:
        attn_activations = attn_activations[:, :, :, -1, :]

    return attn_activations, output


def get_mean_activations(
        tokenized_prompts: list[torch.Tensor], 
        important_ids: list[int],
        tokenizer: PreTrainedTokenizer,
        model: LanguageModel, 
        config: dict[str, Any],
        correct_labels: list[str],      # TODO: implement the correct_labels usage for the evaluation strategies that requires it
        device: str,
        batch_size: int = 1,
        max_len: int = 256,
        evaluator: Evaluator | None = None,
        label_of_interest: str | int | None = None,
        save_output_path: str | None = None,
    ):
    """Compute the average of all the model's activation on the provided prompts

    Args:
        tokenized_prompts (list[torch.Tensor]): list of tokenized prompts
        important_ids (list[int]): list of important indexes i.e. the tokens where the average is computed
        tokenizer (AutoTokenizer): HuggingFace tokenizer
        model (AutoModelForCausalLM): HuggingFace model
        config (dict[str, any]): model's config
        correct_labels (list[str]): list of correct labels for each ICL prompt
        device (str): device
        batch_size (int): batch size for the model. Default to 10.
        max_len (int): max lenght of the prompt to be used (tokenizer parameter). Default to 256.
        save_output_path (str | None): output path to save the model generation. Default to None.

    Returns:
        torch.Tensor: mean of activations (`n_layers, n_heads, seq_len, d_head`)

    """
    assert evaluator is not None, 'Evaluator object is required when using multi token generation'
    assert label_of_interest is not None, 'label_of_interest is required when using multi token generation'

    all_activations = []
    all_outputs = []

    for start_index in (pbar := tqdm(
        range(0, len(tokenized_prompts), batch_size), 
        total = int(np.ceil(len(tokenized_prompts) / batch_size)),
        desc = '[x] Extracting activations batch',
    )):

        end_index = min(start_index + batch_size, len(tokenized_prompts))

        if batch_size > 1:
            current_batch_tokens, _ = pad_input_and_ids(
                tokenized_prompts = tokenized_prompts[start_index : end_index], 
                important_ids = important_ids[start_index : end_index],
                max_len = max_len,
                pad_token_id = tokenizer.eos_token_id,
            )
            for k in current_batch_tokens:
                current_batch_tokens[k] = current_batch_tokens[k].to(device)
        else:
            # avoid to pad the input when batch_size == 1 and use it as is
            current_batch_tokens = tokenized_prompts[start_index].to(device)

        activations, outputs = extract_activations(
            tokenized_prompts=current_batch_tokens, 
            model=model, 
            config=config,
            tokenizer=tokenizer,
            last_token_only=True,
        )

        # move tensors to CPU for memory issues and store it
        all_activations.append(activations.cpu())       # [batch, n_layers, n_heads, seq, d_head]
        all_outputs.append(outputs.cpu())               # [batch, seq]

    # stack all the batches
    all_activations = torch.vstack(all_activations)     # [batch, n_layers, n_heads, seq(no if last_token_only), d_head]


    # take only the generated tokens (from len of original_prompt to the end)
    only_output_tokens = [output.squeeze()[prompt.shape[0] :] for output, prompt in zip(all_outputs, tokenized_prompts)]

    # detokenize prompts and outputs to get the evaluation
    detokenized_prompts = [
        tokenizer.decode(ele, skip_special_tokens=True) for ele in tokenized_prompts
    ]
    detokenized_outputs = [
        tokenizer.decode(ele, skip_special_tokens=True) for ele in only_output_tokens
    ]

    evaluation_results = evaluator.get_evaluation(
        prompts=detokenized_prompts,
        generations=detokenized_outputs,
    )

    # saves outpus
    if save_output_path:
        json_to_write = [
            {
                "input": prompt,
                "output": output,
                "output_label": assigned_label,
            }
            for prompt, output, assigned_label in zip(
                detokenized_prompts, 
                detokenized_outputs, 
                evaluation_results['output'],
            )
        ]
        with open(save_output_path, 'w+', encoding='utf-8') as f:
            json.dump(json_to_write, f, indent=4)


    correct_idx = [True if x == label_of_interest else False for x in evaluation_results['output']]
        
    if sum(correct_idx) > 0:
        print(f'[x] taking {sum(correct_idx)} examples out of {len(correct_idx)}. [{sum(correct_idx)/len(correct_idx)*100:.2f}%]')
    else:
        raise ValueError("Activations cannot be computed when there are no label of interest returned by the evaluator")

    # using only activations from correct prediction to compute the mean_activations
    correct_activations = all_activations[correct_idx]

    mean_activations = correct_activations.mean(axis = 0)
    mean_activations = mean_activations.to(device)
    
    return mean_activations

