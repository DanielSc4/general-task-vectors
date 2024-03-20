from typing import Any
from nnsight import LanguageModel
import torch
from tqdm import tqdm
import numpy as np
import random
import json

from transformers import PreTrainedTokenizer
from src.utils.model_utils import rgetattr
from src.utils.prompt_helper import tokenize_ICL, randomize_dataset, pad_input_and_ids, find_missing_ranges
from src.utils.eval.multi_token_evaluator import Evaluator


def filter_activations(activation, important_ids):
    """
    Average activations of multi-token words across all its tokens
    """   
    to_avg = find_missing_ranges(important_ids)
    for i, j in to_avg:
        activation[:, :, j] = activation[:, :, i : j + 1].mean(axis = 2)

    activation = activation[:, :, important_ids]
    return activation


def simple_forward_pass(
        model: LanguageModel,
        prompt: torch.Tensor | dict[str, torch.Tensor],
        pad_token_id: int | None = None,
    ) -> torch.Tensor:
    """
    Perform a single forward pass with no intervention. Return a tensor [batch, full_output_len].
    """
    # use generate function and return the full output
    with model.generate(
        max_new_tokens=50,
        pad_token_id=pad_token_id,
    ) as generator:
        with generator.invoke(prompt) as invoker:
            pass
    ret = generator.output

    return ret



def replace_heads_w_avg_multi_token(
    tokenized_prompt: dict[str, torch.Tensor] | torch.Tensor,
    layers_heads: list[tuple[int, int]], 
    avg_activations: list[torch.Tensor], 
    model: LanguageModel, 
    config: dict[str, Any],
    pad_token_id: int | None,
) -> torch.Tensor:
    """
    Same as `replace_heads_w_avg` but for multi token generation. Here last_token_only is default since prompt can change in length
    Only batchsize = 1 is allowed
    """
        
    assert len(layers_heads) == len(avg_activations), f'layers_heads and avg_activations must have the same length. Got {len(layers_heads)} and {len(avg_activations)}'

    d_head = config['d_model'] // config['n_heads']

    """
    Developer note: here avg_activations is a list of activations for each head to change
        if calcultaing AIE, the length of the list is 1.
    """

    with model.generate(
        max_new_tokens=50,
        pad_token_id=pad_token_id,
    ) as generator:
        with generator.invoke(tokenized_prompt) as _:
            for idx, (num_layer, num_head) in enumerate(layers_heads):
                attention_head_values = rgetattr(model, config['attn_hook_names'][num_layer]).input[0][0][
                    :, :, (num_head * d_head) : ((num_head + 1) * d_head)
                ]
                # shape: [batch = 1, seq (e.g. 256), d_head]
                attention_head_values[
                    :, len(tokenized_prompt) - 1, :
                ] = avg_activations[idx]

    return generator.output



def replace_heads_w_avg(
    tokenized_prompt: dict[str, torch.Tensor] | torch.Tensor, 
    important_ids: list[list[int]], 
    layers_heads: list[tuple[int, int]], 
    avg_activations: list[torch.Tensor], 
    model, 
    config,
    last_token_only: bool = True,
) -> torch.Tensor:
    """Replace the activation of specific head(s) (listed in `layers_heads`) with the avg_activation for each 
    specific head (listed in `avg_activations`) only in `important_ids` positions. 
    Than compute the output (softmaxed logits) of the model with all the new activations.

    Args:
        tokenized_prompt (dict[str, torch.Tensor]): tokenized prompt
        important_ids (list[list[int]]): list of important indexes i.e. the tokens where the average must be substituted
        layers_heads (list[tuple[int, int]]): list of tuples each containing a layer index, head index
        avg_activations (list[torch.tensor]): list of activations (`size: (seq_len, d_head)`) for each head listed in layers_heads. The length must be the same of layers_heads
        model (): model
        config (): model's config
        last_token_only (bool): Whether consider the last token from activation as the only important token to replace. 
            If False, every important_id with the mean_activation and mutli tokens word takes the mean activation from their last token activation. Defaults: True.

    Returns:
        torch.Tensor: model's probabilities over vocab (post-softmax) with the replaced activations with shape (batch, vocab_size)
    """
    assert len(layers_heads) == len(avg_activations), f'layers_heads and avg_activations must have the same length. Got {len(layers_heads)} and {len(avg_activations)}'

    d_head = config['d_model'] // config['n_heads']

    """
    Developer note: here avg_activations is a list of activations for each head to change
        if calcultaing AIE, the length of the list is 1.
    """
    with model.invoke(tokenized_prompt) as invoker:
        for idx, (num_layer, num_head) in enumerate(layers_heads):
            # https://github.com/huggingface/transformers/blob/224ab70969d1ac6c549f0beb3a8a71e2222e50f7/src/transformers/models/gpt2/modeling_gpt2.py#L341
            # shape: tuple[output from the attention module (hidden state, ), present values (cache), attn_weights] (taking 0-th value)
            # new shape: torch.size([batchsize, seq_len(256 if max len), hidden_size])
            attention_head_values = rgetattr(model, config['attn_hook_names'][num_layer]).input[0][0][
                :, :, (num_head * d_head) : ((num_head + 1) * d_head)
            ]
            # for each prompt in batch and important ids of that prompt
            # substitute with the mean activation (unsqueeze for adding the batch dimension)
            for prompt_idx, prompt_imp_ids in zip(
                range(attention_head_values.shape[0]), 
                important_ids,
            ):
                if last_token_only:
                    attention_head_values[prompt_idx][      # shape: [seq (256), d_head]
                        prompt_imp_ids[-1], :
                    ] = avg_activations[idx][-1]     # shape: [seq (1 (the -1 pos.), d_model)] pos. 255 aka 256-th token (when max_len = 256)
                else:
                    # replace important ids with the mean activations
                    attention_head_values[prompt_idx][prompt_imp_ids] = avg_activations[idx].unsqueeze(0)
                    to_avg = find_missing_ranges(prompt_imp_ids)
                    # replace non important ids (i.e. other tokens of the same word) with the same values (avg of the token activation)
                    for n_interval in range(len(to_avg)):
                        # calculate range where the substitution must take place (e.g. from [2, 5] to [2, 3, 4])
                        range_where_replace = list(range(*to_avg[n_interval]))
                        for token_col in range_where_replace:
                            # replace with the token emb. with the avg taken from the last token of the word emb. 
                            # explaination: (for ele in [2, 3, 4] replace with the values in col 5, (same as range_where_replace[-1] + 1))
                            attention_head_values[prompt_idx][token_col] = attention_head_values[prompt_idx][to_avg[n_interval][-1]]
        
    # store the output probability
    output = invoker.output.logits[:,-1,:].softmax(dim=-1)

    return output


def _aie_loop(
    model: LanguageModel,
    tokenizer: PreTrainedTokenizer,
    config: dict[str, Any],
    main_pbar: tqdm,
    mean_activations: torch.Tensor,
    prompt: dict[str, torch.Tensor] | torch.Tensor,
    important_ids: list[int] | None,
    ) -> list[list[str | torch.Tensor]] | torch.Tensor:
    """
    Returns, for each layer, for each head a string if multi_token_generation. 
    """
    
    inner_bar_layers = tqdm(
       range(config['n_layers']),
       total=config['n_layers'],
       leave=False,
       desc='  -th layer',
    )
    layers_output = []
    for layer_i in inner_bar_layers:
        inner_bar_heads = tqdm(
            range(config['n_heads']),
            total=config['n_heads'],
            leave=False,
            desc='    -th head',
        )
        heads_output = []
        for head_j in inner_bar_heads:
            main_pbar.set_description(
                f'Processing edited model (l: {layer_i}/{config["n_layers"]}, h: {head_j}/{config["n_heads"]})'
            )
            # here the return value has already the softmaxed scored from the evaluator object
            model_output = replace_heads_w_avg_multi_token(
                tokenized_prompt=prompt,
                layers_heads=[(layer_i, head_j)],
                avg_activations=[mean_activations[layer_i, head_j]],
                model=model,
                config=config,
                pad_token_id=tokenizer.pad_token_id,
            )
            # del batch_size
            model_output = model_output.squeeze()
            # get only output tokens
            only_output_tokens = model_output[len(prompt) :]
            heads_output.append(
                tokenizer.decode(only_output_tokens, skip_special_tokens = True)
            )

        layers_output.append(heads_output)

    return layers_output
    # else:
    #     return edited


def _compute_scores_multi_token(
    model: LanguageModel,
    tokenizer: PreTrainedTokenizer,
    config: dict[str, Any],
    tokenized_prompts: tuple[torch.Tensor], # tuple len = aie_support, size tensor Size([seq])
    mean_activations: torch.Tensor,
    evaluator: Evaluator,
    save_output_path: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:     # TODO, controlla se effettivamente è così

    # intervention
    prompts_and_outputs_original = {
        'prompt': [],
        'output': [],
    }
    prompts_and_outputs_edited = {
        'prompt': [],
        'output': [],
    }
    for idx in (pbar := tqdm(
        range(len(tokenized_prompts)),
        total=len(tokenized_prompts),
    )):
        pbar.set_description('Original forward pass')
        # simple forward pass
        model_output = simple_forward_pass(
            model=model, 
            prompt=tokenized_prompts[idx], 
            pad_token_id=tokenizer.pad_token_id,
        )

        # del batch_size dim
        model_output = model_output.squeeze()
        # get only output tokens
        only_output_tokens = model_output[len(tokenized_prompts[idx]) : ]

        prompts_and_outputs_original['prompt'].append(
            tokenizer.decode(tokenized_prompts[idx], skip_special_tokens=True)
        )
        prompts_and_outputs_original['output'].append(
            tokenizer.decode(only_output_tokens, skip_special_tokens=True)
        )

         
        pbar.set_description('Edited forward pass')
        # for each prompt the function returns a list of a list [n_layers x n_heads]
        edited_out = _aie_loop(
            model=model,
            tokenizer=tokenizer,
            config=config,
            main_pbar=pbar,
            mean_activations=mean_activations,
            prompt=tokenized_prompts[idx],
            important_ids=None,
        )

        prompts_and_outputs_edited['prompt'].append(
            tokenizer.decode(tokenized_prompts[idx], skip_special_tokens=True)
        )
        prompts_and_outputs_edited['output'].append(edited_out)
    

    # Evaluation
    label_of_interest = evaluator.negative_label

    print('[x] Evaluating original model outputs')
    evaluation_result = evaluator.get_evaluation(
        prompts=prompts_and_outputs_original['prompt'],
        generations=prompts_and_outputs_original['output'],
    )
    score_of_interest = evaluation_result[label_of_interest]
    scores_original = torch.tensor(score_of_interest)       # shape: len(tokenized_prompts)


    print('[x] Evaluating edited model outputs')
    scores_edited = torch.zeros([len(tokenized_prompts), config['n_layers'], config['n_heads']])
    
    for idx_prompt in range(len(prompts_and_outputs_edited['prompt'])):
        for layer in range(config['n_layers']):
            for head in range(config['n_heads']):
                # qui devo passare un prompt alla volta, nonostante get_evaluation accetterebbe una lista di prompt
                evaluation_result = evaluator.get_evaluation(
                    prompts=[prompts_and_outputs_edited['prompt'][idx_prompt]],
                    generations=[prompts_and_outputs_edited['output'][idx_prompt][layer][head]],
                )
                # uso [0] visto che sto passando un prompt alla volta, quindi il risultato è una lista ma che contiene un solo score
                scores_edited[idx_prompt][layer][head] = evaluation_result[label_of_interest][0]

                

    # saving everything (logs_output)
    print('[x] Saving logs')
    logs_output = []
    for idx in range(len(prompts_and_outputs_edited['prompt'])):
        logs_output.append(
            {
                'input': prompts_and_outputs_original['prompt'][idx],
                'original': {
                    'output': prompts_and_outputs_original['output'][idx],
                    'eval': evaluation_result,
                },
                'edited': [
                    (
                        f'{layer},{head}', {
                            'output': prompts_and_outputs_edited['output'][idx][layer][head],
                            'eval': {
                                label_of_interest: scores_edited[idx][layer][head].item(),
                            },
                        }
                    )
                    for layer in range(config['n_layers'])
                    for head in range(config['n_heads'])
                ]
            }
        )

    if save_output_path:
        with open(save_output_path, 'w+') as fout:
            json.dump(logs_output, fout, indent=4)

    return scores_original, scores_edited


def compute_indirect_effect(
    model,
    tokenizer: PreTrainedTokenizer,
    config: dict[str, Any],
    dataset: list[tuple[str, str]],
    mean_activations: torch.Tensor,
    ICL_examples: int = 4,
    batch_size: int = 32,
    aie_support: int = 25,
    evaluator: Evaluator | None = None,
    save_output_path: str | None = None,
):
    """Compute indirect effect on the provided dataset by comparing the prediction of the original model
    to the predicition of the modified model. Specifically, for the modified model, each attention head
    activation is substituted with the corresponding mean_activation provided to measure the impact 
    on the final correct label predicition.

    Args:
        model (_type_): Language model
        tokenizer (tokenizer): Tokenizer
        config (dict[str, Any]): model's config dictionary
        dataset (list[tuple[str, str]]): list of tuples with the first element being the prompt and the second the correct label
        mean_activations (torch.Tensor): mean activation for each head in the model. Should be [n_layers, n_heads, seq_len, d_head]
        ICL_examples (int, optional): number of ICL examples exluding the final prompt. Defaults to 4.
        batch_size (int, optional): batch size dimension. Defaults to 32.
        aie_support (int, optional): number of prompt supporting the average indirect effect on the model. Defaults to 25.

    Returns:
        _type_: TBD
    """
    # randomize prompt's labels to make the model unable to guess the correct answer
    # randomized_dataset = randomize_dataset(dataset)
    randomized_dataset = dataset
    
    all_tokenized_prompt, all_important_ids, all_correct_labels = tokenize_ICL(
        tokenizer, 
        ICL_examples = ICL_examples, 
        dataset = randomized_dataset
    )
    # create a subset with aie_support elements
    idx_for_aie = random.sample(range(len(all_tokenized_prompt)), aie_support)
    selected_examples = [
        (all_tokenized_prompt[i], all_important_ids[i], all_correct_labels[i])
        for i in idx_for_aie
    ]
    all_tokenized_prompt, all_important_ids, all_correct_labels = zip(*selected_examples)

    assert evaluator is not None, 'Evaluator object is required when using multi token generation'
    assert batch_size == 1, 'Batchsize > 1 is not supported when using multi_token_generation'

    scores_original, scores_edited = _compute_scores_multi_token(
        model=model,
        tokenizer=tokenizer,
        config=config,
        tokenized_prompts=all_tokenized_prompt,
        mean_activations=mean_activations,
        evaluator=evaluator,
        save_output_path=save_output_path,
    )
    
    cie = torch.zeros([scores_original.shape[0], config['n_layers'], config['n_heads']])
    for prompt_idx in range(scores_original.shape[0]):
        for layer in range(config['n_layers']):
            for head in range(config['n_heads']):
                cie[prompt_idx, layer, head] = scores_edited[prompt_idx, layer, head] - scores_original[prompt_idx]

    cie = cie.mean(dim=0)
    
    return cie, scores_original, scores_edited

