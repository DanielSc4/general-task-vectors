from typing import Any
import torch
from tqdm import tqdm
import numpy as np

from src.utils.model_utils import rgetattr
from src.utils.prompt_helper import tokenize_ICL, randomize_dataset, pad_input


def replace_heads_w_avg(tokenized_prompt: torch.tensor, important_ids: list[int], layers_heads: list[tuple[int, int]], avg_activations: list[torch.tensor], model, config):
    """Replace the activation of specific heads (listed in layers_heads) with the avg_activation for each specific head (listed in avg_activations). 
    Than compute the output of the model with all the new activations

    Args:
        tokenized_prompt (torch.tensor): tokenized prompt (single batch)
        important_ids (list[int]): list of important indexes i.e. the tokens where the average must be substituted
        layers_heads (list[tuple[int, int]]): list of tuples each containing a layer index, head index
        avg_activations (list[torch.tensor]): list of activations (`size: (seq_len, d_head)`) for each head listed in layers_heads. The length must be the same of layers_heads
        model (): model
        config (): model's config

    Returns:
        torch.Tensor: model's probabilities over vocab (post-softmax) with the replaced activations
    """
    assert len(layers_heads) == len(avg_activations), f'layers_heads and avg_activations must have the same length. Got {len(layers_heads)} and {len(avg_activations)}'

    d_head = config['d_model'] // config['n_heads']

    with model.invoke(tokenized_prompt) as invoker:
        for idx, (num_layer, num_head) in enumerate(layers_heads):
            # select the head (output shape is torch.Size([batch, seq_len, d_model]))
            
            # https://github.com/huggingface/transformers/blob/224ab70969d1ac6c549f0beb3a8a71e2222e50f7/src/transformers/models/gpt2/modeling_gpt2.py#L341
            # shape: tuple[output from the attention module (hidden state), present values (cache), attn_weights] 
            # (taking 0-th value)
            raw = rgetattr(model, config['attn_hook_names'][num_layer]).output[0]
            print(f'raw shape: {raw.shape}. b, seq, d_model')
            attention_head_values = rgetattr(model, config['attn_hook_names'][num_layer]).output[0][
                :, :, (num_head * d_head) : ((num_head + 1) * d_head)
            ]
            
            print(f'Single attention head has shape: {attention_head_values.shape}. b, seq, d_head')
            print(f'Avg activations has shape: {avg_activations[idx].unsqueeze(0).shape}. :b:, seq, d_head')
            print(f'Important_ids[0] len: {len(important_ids[0])}')
            print(important_ids)
            print()

            # substitute only the important indexes (unsqueeze for adding the batch dimension) TODO: check correctness for important ids
            attention_head_values[:, important_ids, :] = avg_activations[idx].unsqueeze(0)
            
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

    # probabilities over vocab from the original model
    probs_original = [] # torch.zeros([len(randomized_dataset), config['vocab_size']])
    probs_edited = []   # torch.zeros([len(all_tokenized_prompt), config['n_layers'], config['n_heads'], config['vocab_size']])

    for start_index in (pbar := tqdm(
        range(0, len(all_tokenized_prompt), batch_size), 
        total = int(np.ceil(len(all_tokenized_prompt) / batch_size))
    )):

        end_index = min(start_index + batch_size, len(all_tokenized_prompt))
        current_batch_size = end_index - start_index

        current_batch_tokens = pad_input(
            tokenized_prompts = all_tokenized_prompt[start_index : end_index], 
            max_len = 256,
            pad_token_id = tokenizer.eos_token_id
        )
        current_batch_important_ids = all_important_ids[start_index : end_index]

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

        for layer_i in range(config['n_layers']):
            for head_j in range(config['n_heads']):

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

    # CIE(ij) = probability of correct_label token y (w/ edited model) - probability of correct_label token y (w/ original model)
    #      e.g. CIE(ij) = 0.9 - 0.1 = 0.8      head has great effect
    #      e.g. CIE(ij) = 0.3 - 0.1 = 0.2      head does not influence too much the output

    # considering only the first generated id
    correct_ids = list(map(lambda x: x[0], tokenizer(all_correct_labels)['input_ids']))

    cie = torch.zeros([len(correct_ids), config['n_layers'], config['n_heads']])

    for prompt_idx in range(len(correct_ids)):
        for layer in range(config['n_layers']):
            for head in range(config['n_heads']):
                prob_correct_token_original_model = probs_edited[
                    prompt_idx, layer, head, correct_ids[prompt_idx]
                ].item()
                prob_correct_token_edited_model = probs_original[
                    prompt_idx, correct_ids[prompt_idx]
                ].item()
                cie[prompt_idx, layer, head] = prob_correct_token_edited_model - prob_correct_token_original_model
    
    
    return cie, probs_original, probs_edited
