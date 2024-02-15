from typing import Any
from nnsight import LanguageModel
import torch
import numpy as np
from transformers import PreTrainedTokenizer
from tqdm import tqdm

from src.utils.eval.multi_token_evaluator import Evaluator
from src.utils.prompt_helper import pad_input_and_ids
from .utils.model_utils import rsetattr, rgetattr
from .utils.prompt_helper import find_missing_ranges



def filter_activations(activation, important_ids):
    """
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
        activations (list[torch.Tensor]): list of residual streams for each layer shape: (list:n_layers[tensor: batch, seq, d_model])
        config (dict[str, Any]): model's config

    Returns:
        activations: reshaped activation in [n_layers, seq, n_heads, d_head]
    """
    new_shape = torch.Size([
        activations[0].shape[0],         # batch_size
        activations[0].shape[1],         # seq_len
        config['n_heads'],                          # n_heads
        config['d_model'] // config['n_heads'],     # d_head
    ])
    print(f'modified: {activations[0].view(*new_shape).shape}')
    attn_activations = torch.stack([act.view(*new_shape) for act in activations])
    return attn_activations


def extract_activations(
        tokenized_prompts: dict[str, torch.Tensor], 
        model: LanguageModel, 
        config: dict[str, Any],
        tokenizer: PreTrainedTokenizer,
        device: str,
        multi_token_generation: bool = False,
    ):
    """Extract the activation and the output produced from the model using the tokenized prompts provided

    Args:
        tokenized_prompts (list[torch.Tensor]): list of tokenized prompts
        model (AutoModelForCausalLM): HuggingFace model
        config (dict[str, any]): model's config
        tokenizer (AutoTokenizer): HuggingFace tokenizer
        device (str): device
        multi_token_generation (bool | int): Allow for multi token generation. If False the model will generate 1 token. Default to False.

    Returns:
        tuple[list[torch.Tensor], list[torch.Tensor]]: tuple corresponding to the activations and the model output
    """


    dataset_activations, outputs = [], []
    # pbar = tqdm(tokenized_prompts, total = len(tokenized_prompts), desc = '[x] Extracting activations')
    # for prompt in pbar:

    for k in tokenized_prompts:
        tokenized_prompts[k] = tokenized_prompts[k].to(device)
    with model.generate(
        max_new_tokens=1 if not multi_token_generation else 200,
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

    outputs.append(generator.output)

    # from hidden state split heads and permute: n_layers, tokens, n_heads, d_head -> n_layers, n_heads, tokens, d_head
    print(f'element of layer_attn_activations has shape: {layer_attn_activations[0].shape}')
    attn_activations = split_activation(layer_attn_activations, config).permute(0, 2, 1, 3)
    dataset_activations.append(attn_activations)
    
    print(dataset_activations[-1].shape)
    dataset_activations = torch.stack(dataset_activations)
    return dataset_activations, outputs


def get_mean_activations(
        tokenized_prompts: list[torch.Tensor], 
        important_ids: list[int],
        tokenizer: PreTrainedTokenizer,
        model: LanguageModel, 
        config: dict[str, Any],
        correct_labels: list[str],
        device: str,
        batch_size: int = 10,
        multi_token_generation: bool = False,
        evaluator: Evaluator | None = None,
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
        multi_token_generation (bool | int): Allow for multi token generation. If False the model will generate 1 token. Default to False.
        evaluator (Evaluator): Required if multi_token_generation is active. Defines an evaluation strategy. Default to None.

    Returns:
        torch.Tensor: mean of activations (`n_layers, n_heads, seq_len, d_head`)
    """


    for start_index in (pbar := tqdm(
        range(0, len(tokenized_prompts), batch_size), 
        total = int(np.ceil(len(tokenized_prompts) / batch_size))
    )):

        end_index = min(start_index + batch_size, len(tokenized_prompts))
        current_batch_size = end_index - start_index

        current_batch_tokens, current_batch_important_ids = pad_input_and_ids(
            tokenized_prompts = tokenized_prompts[start_index : end_index], 
            important_ids = important_ids[start_index : end_index],
            max_len = 256,
            pad_token_id = tokenizer.eos_token_id,
        )

        activations, outputs = extract_activations(
            tokenized_prompts=current_batch_tokens, 
            model=model, 
            config=config,
            tokenizer=tokenizer,
            device=device,
            multi_token_generation = multi_token_generation,
        )
        print(activations.shape)

        # move tensors to CPU for memory issues
        for idx in range(len(activations)):
            activations[idx] = activations[idx].cpu()
        
        else:
            # keep only important tokens averaging the rest (activations_clean: [batch, n_layers, n_heads, seq, d_head])
            activations_clean = torch.stack(
                [filter_activations(activations[i], important_ids[i]) for i in range(len(activations))]
            )

        if multi_token_generation:
            assert evaluator is not None, 'When using multi_token_generation an evaluator is required'
            only_output_tokens = []
            for original_prompt, output in zip(tokenized_prompts, outputs):
                # take only the generated tokens (from len of original_prompt to the end)
                only_output_tokens.append(
                    output.squeeze().cpu()[- original_prompt.shape[0] :].unsqueeze(0)   # adding batchsize dim = 1
                )
                # detokenize prompt the get the evaluation
                detokenized_outputs = [
                    tokenizer.decode(ele.squeeze(), skip_special_tokens=True) for ele in only_output_tokens
                ]
                print(detokenized_outputs)
                evaluation_results = evaluator.get_evaluation(prompts=detokenized_outputs)
                evaluation_results = torch.tensor(evaluation_results)
                # suppopsing label == 1 -> negative output (so, using torch.ones)
                correct_idx = (evaluation_results == torch.ones(evaluation_results.shape[0]))
                
                print(f'[x] taking {correct_idx.sum()} examples out of {evaluation_results.shape[0]}')

        else:
            # considering only the first token to evaluate the output
            only_output_tokens = torch.tensor(list(map(lambda x: x.squeeze()[-1].item(), outputs)))
            only_labels_tokens = torch.tensor([ele[0] for ele in tokenizer(correct_labels)['input_ids']])

            correct_idx = (only_output_tokens == only_labels_tokens)
            accuracy = correct_idx.sum() / len(correct_idx)
            if correct_idx.sum() > 0:
                print(f'[x] Model accuracy: {accuracy:.2f}, using {correct_idx.sum()} (out of {len(correct_idx)}) examples to compute mean activations')
            else:
                print(f'[x] Model accuracy is 0, mean_activations cannot be computed!')
                raise ValueError("Activations cannot be computed when model accuracy is 0%")

        # TODO applica correzioni per batch da qui in poi
    # using only activations from correct prediction to compute the mean_activations
    correct_activations = activations_clean[correct_idx]
    
    mean_activations = correct_activations.mean(axis = 0)
    mean_activations = mean_activations.to(device)
    
    return mean_activations
