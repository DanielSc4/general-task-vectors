from typing import Any
import torch
from tqdm import tqdm
import numpy as np
import random

from src.utils.model_utils import rgetattr
from src.utils.prompt_helper import tokenize_ICL, randomize_dataset, pad_input_and_ids, find_missing_ranges



def filter_activations(activation, important_ids):
    """
    Average activations of multi-token words across all its tokens
    """   
    to_avg = find_missing_ranges(important_ids)
    for i, j in to_avg:
        activation[:, :, j] = activation[:, :, i : j + 1].mean(axis = 2)

    activation = activation[:, :, important_ids]
    return activation


def replace_heads_w_avg(
        tokenized_prompt: torch.tensor, 
        important_ids: list[int], 
        layers_heads: list[tuple[int, int]], 
        avg_activations: list[torch.tensor], 
        model, 
        config,
        last_token_only: bool = True,
    ):
    """Replace the activation of specific head(s) (listed in `layers_heads`) with the avg_activation for each 
    specific head (listed in `avg_activations`) only in `important_ids` positions. 
    Than compute the output (softmaxed logits) of the model with all the new activations.

    Args:
        tokenized_prompt (torch.tensor): tokenized prompt
        important_ids (list[int]): list of important indexes i.e. the tokens where the average must be substituted
        layers_heads (list[tuple[int, int]]): list of tuples each containing a layer index, head index
        avg_activations (list[torch.tensor]): list of activations (`size: (seq_len, d_head)`) for each head listed in layers_heads. The length must be the same of layers_heads
        model (): model
        config (): model's config
        last_token_only (bool): Whether consider the last token from activation as the only important token to replace. 
            If False, every important_id with the mean_activation and mutli tokens word takes the mean activation from their last token activation. Defaults: True.

    Returns:
        torch.Tensor: model's probabilities over vocab (post-softmax) with the replaced activations
    """
    assert len(layers_heads) == len(avg_activations), f'layers_heads and avg_activations must have the same length. Got {len(layers_heads)} and {len(avg_activations)}'

    d_head = config['d_model'] // config['n_heads']

    """
    Developer note: here avg_activations is a list of activations for each head to change
        if calcultaing AIE, the length of the list is 1.
    """
    with model.invoke(tokenized_prompt) as invoker:
        for idx, (num_layer, num_head) in enumerate(layers_heads):
            # select the head (output shape is torch.Size([batch, seq_len, d_model]))
            
            # https://github.com/huggingface/transformers/blob/224ab70969d1ac6c549f0beb3a8a71e2222e50f7/src/transformers/models/gpt2/modeling_gpt2.py#L341
            # shape: tuple[output from the attention module (hidden state), present values (cache), attn_weights] 
            # (taking 0-th value)
            attention_head_values = rgetattr(model, config['attn_hook_names'][num_layer]).output[0][
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
                
    # store the output probabilities
    probs = invoker.output.logits[:,-1,:].softmax(dim=-1)
    return probs


def compute_indirect_effect(
        model,
        tokenizer,
        config: dict[str, Any],
        dataset: list[tuple[str, str]],
        mean_activations: torch.Tensor,
        ICL_examples: int = 4,
        batch_size: int = 32,
        aie_support: int = 25,
    ):
    """Compute indirect effect on the provided dataset by comparing the prediction of the original model
    to the predicition of the modified model. Specifically, for the modified model, each attention head
    activation is substituted with the corresponding mean_activation provided to measure the impacto 
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

    # randomize prompts to make the model unable to guess the correct answer
    randomized_dataset = randomize_dataset(dataset)
    
    all_tokenized_prompt, all_important_ids, all_correct_labels = tokenize_ICL(
        tokenizer, 
        ICL_examples = ICL_examples, 
        dataset = randomized_dataset
    )
    # create a subset with aie_support elements
    print(len(all_tokenized_prompt))
    idx_for_aie = random.sample(range(len(all_tokenized_prompt)), aie_support)
    selected_examples = [
        (all_tokenized_prompt[i], all_important_ids[i], all_correct_labels[i])
        for i in idx_for_aie
    ]
    all_tokenized_prompt, all_important_ids, all_correct_labels = zip(*selected_examples)

    # probabilities over vocab from the original model
    probs_original = [] # torch.zeros([len(randomized_dataset), config['vocab_size']])
    probs_edited = []   # torch.zeros([len(all_tokenized_prompt), config['n_layers'], config['n_heads'], config['vocab_size']])

    for start_index in (pbar := tqdm(
        range(0, len(all_tokenized_prompt), batch_size), 
        total = int(np.ceil(len(all_tokenized_prompt) / batch_size))
    )):

        end_index = min(start_index + batch_size, len(all_tokenized_prompt))
        current_batch_size = end_index - start_index

        current_batch_tokens, current_batch_important_ids = pad_input_and_ids(
            tokenized_prompts = all_tokenized_prompt[start_index : end_index], 
            important_ids = all_important_ids[start_index : end_index],
            max_len = 256,
            pad_token_id = tokenizer.eos_token_id,
        )

        pbar.set_description('Processing original model')
        
        # take the original result from the model (probability of correct response token y)
        with model.invoke(current_batch_tokens) as invoker:
            pass # no changes to make in the forward pass
        logits = invoker.output.logits
        logits = logits[:,-1,:] # select only the predicted token (i.e. the final token, keeping batch and vocab_size)
        # store the probabilities for each token in vocab
        probs_original.append(logits.softmax(dim=-1).cpu())

        # for each layer i, for each head j in the model save the vocab size in output
        edited = torch.zeros([current_batch_size, config['n_layers'], config['n_heads'], config['vocab_size']])

        inner_bar_layers = tqdm(
           range(config['n_layers']),
           total=config['n_layers'],
           leave=False,
           desc=' -th layer',
        )
        for layer_i in inner_bar_layers:
            inner_bar_heads = tqdm(
                range(config['n_heads']),
                total=config['n_heads'],
                leave=False,
                desc=' -th head',
            )
            for head_j in inner_bar_heads:
                pbar.set_description(
                    f'Processing edited model (l: {layer_i}/{config["n_layers"]}, h: {head_j}/{config["n_heads"]})'
                )
                returned = replace_heads_w_avg(
                    tokenized_prompt=current_batch_tokens,
                    important_ids=current_batch_important_ids,
                    layers_heads=[(layer_i, head_j)],
                    avg_activations=[mean_activations[layer_i, head_j]],
                    model=model,
                    config=config,
                )
                edited[:, layer_i, head_j, :] = returned
            
        probs_edited.append(edited.cpu())

    probs_original = torch.vstack(probs_original)
    probs_edited = torch.vstack(probs_edited)

    # CIE(ij) = probability of correct_label token y (w/ edited model) - probability of correct_label token y (w/ original model). Some e.g.
    #      e.g. CIE(ij) = 0.9 - 0.1 = 0.8      head has great effect
    #      e.g. CIE(ij) = 0.3 - 0.1 = 0.2      head does not influence too much the output
    #      e.g. CIE(ij) = 0.3 - 0.8 = -0.5     head contribute to the output in an inverse way

    # considering only the first generated id
    correct_ids = list(map(lambda x: x[0], tokenizer(all_correct_labels)['input_ids']))

    cie = torch.zeros([len(correct_ids), config['n_layers'], config['n_heads']])

    for prompt_idx in range(len(correct_ids)):
        for layer in range(config['n_layers']):
            for head in range(config['n_heads']):
                prob_correct_token_edited_model = probs_edited[
                    prompt_idx, layer, head, correct_ids[prompt_idx]
                ].item()
                prob_correct_token_original_model = probs_original[
                    prompt_idx, correct_ids[prompt_idx]
                ].item()
                cie[prompt_idx, layer, head] = prob_correct_token_edited_model - prob_correct_token_original_model
    
    
    return cie.mean(dim=0), probs_original, probs_edited



def use_task_vector(
    mean_activations: torch.Tensor,
    top_heads: list[tuple],
    model,
    prompt
):
    # TBD
    pass