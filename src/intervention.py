from typing import Any
from nnsight import LanguageModel
import torch
from tqdm import tqdm
import numpy as np
import random

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
        multi_token_generation: bool = False,
        pad_token_id: int | None = None,
    ) -> torch.Tensor:
    """
    Perform a single forward pass with no intervention. Return a tensor [batch, vocab_size] if multi_token_generation is False, [batch, full_output_len] if multi_token_generation.
    """
    # TODO: check whether this two if statement can be a single operation using the generation context but getting the vocab size for the first token only
    if multi_token_generation:
        # use generate function and return the full output
        with model.generate(
            max_new_tokens=30,
            pad_token_id=pad_token_id,
        ) as generator:
            with generator.invoke(prompt) as invoker:
                pass
        ret = generator.output
    else:
        # use invoker and return a softmaxed tensor with shape: [batch, vocab_size]
        with model.invoke(prompt) as invoker:
            pass    # no action required
        logits = invoker.output.logits[:, -1, :]    # getting only the predicted token (i.e. final token), keeping batchsize and vocab_size
        ret = logits.softmax(dim=-1)        # has shape [batch, vocab_size]

    return ret



def replace_heads_w_avg_multi_token(
    tokenized_prompt: dict[str, torch.Tensor],
    important_ids: list[list[int]],
    layers_heads: list[tuple[int, int]], 
    avg_activations: list[torch.Tensor], 
    model: LanguageModel, 
    config: dict[str, Any],
    pad_token_id: int,
) -> torch.Tensor:
    """
    Same as `replace_heads_w_avg` but for multi token generation. Here last_token_only is default since prompt can change in length
    """
        
    assert len(layers_heads) == len(avg_activations), f'layers_heads and avg_activations must have the same length. Got {len(layers_heads)} and {len(avg_activations)}'

    d_head = config['d_model'] // config['n_heads']

    """
    Developer note: here avg_activations is a list of activations for each head to change
        if calcultaing AIE, the length of the list is 1.
    """

    with model.generate(
        max_new_tokens=150,
        pad_token_id=pad_token_id,
    ) as generator:
        with generator.invoke(tokenized_prompt) as invoker:
            for idx, (num_layer, num_head) in enumerate(layers_heads):
                
                attention_head_values = rgetattr(model, config['attn_hook_names'][num_layer]).input[0][0][
                    :, :, (num_head * d_head) : ((num_head + 1) * d_head)
                ]

                for prompt_idx, prompt_imp_ids in zip(
                    range(attention_head_values.shape[0]), 
                    important_ids,
                ):
                    attention_head_values[prompt_idx][      # shape: [seq (256), d_head]
                        prompt_imp_ids[-1], :
                    ] = avg_activations[idx]

    return generator.output



def replace_heads_w_avg(
    tokenized_prompt: dict[str, torch.Tensor], 
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



def compute_cie_single_token(
    tokenizer: PreTrainedTokenizer,
    config: dict[str, Any],
    correct_labels,
    probs_original,
    probs_edited,

):
    """
    Return the CIE matrix averaged accross all prompts, calculated as follow:
    CIE(ij) = probability of correct_label token y (w/ edited model) - probability of correct_label token y (w/ original model). Some e.g.
         e.g. CIE(ij) = 0.9 - 0.1 = 0.8      head has great effect
         e.g. CIE(ij) = 0.3 - 0.1 = 0.2      head does not influence too much the output
         e.g. CIE(ij) = 0.3 - 0.8 = -0.5     head contribute to the output in an inverse way
    """
    
    correct_ids = list(map(lambda x: x[0], tokenizer(correct_labels)['input_ids']))

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


    return cie.mean(dim=0)


def compute_indirect_effect(
    model,
    tokenizer: PreTrainedTokenizer,
    config: dict[str, Any],
    dataset: list[tuple[str, str]],
    mean_activations: torch.Tensor,
    ICL_examples: int = 4,
    batch_size: int = 32,
    aie_support: int = 25,
    multi_token_generation: bool = False,
    evaluator: Evaluator | None = None,
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


    if multi_token_generation:
        assert evaluator is not None, 'Evaluator object is required when using multi token generation'

    # randomize prompts to make the model unable to guess the correct answer
    randomized_dataset = randomize_dataset(dataset)
    
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

    # probabilities over vocab from the original model
    probs_original = [] # torch.zeros([len(randomized_dataset), config['vocab_size']])
    probs_edited = []   # torch.zeros([len(all_tokenized_prompt), config['n_layers'], config['n_heads'], config['vocab_size'] or len(generated_output)])

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
        

        # for each layer i, for each head j in the model save the vocab size in output
        if multi_token_generation:
            model_output = simple_forward_pass(model, current_batch_tokens, multi_token_generation, pad_token_id=tokenizer.pad_token_id)

            only_output_tokens = [
                output.squeeze()[current_batch_tokens['input_ids'].shape[1] :] for output in model_output
            ]
            detokenize_outputs = [
                tokenizer.decode(ele, skip_special_tokens=True) for ele in only_output_tokens
            ]

            # returns [torch.tensor([0.99, 0.01]), ...] like
            evaluation_result = evaluator.get_evaluation(texts=detokenize_outputs, softmaxed=True)
            results = torch.tensor([res[evaluator.negative_label] for res in evaluation_result])
            
            # result has shape [batch] (i.e. for each element of the batch (sentence) there is a score)
            probs_original.append(results) 

            # inventati una strutta dati
            edited = torch.zeros([current_batch_size, config['n_layers'], config['n_heads']])       # no vocab_size, store directly the score returned from the Evaluator
        else:
            # take the original result from the model (probability of correct response token y)
            probs_original.append(
                simple_forward_pass(model, current_batch_tokens, multi_token_generation)
            )
            edited = torch.zeros([current_batch_size, config['n_layers'], config['n_heads'], config['vocab_size']])

        inner_bar_layers = tqdm(
           range(config['n_layers']),
           total=config['n_layers'],
           leave=False,
           desc='  -th layer',
        )
        for layer_i in inner_bar_layers:
            inner_bar_heads = tqdm(
                range(config['n_heads']),
                total=config['n_heads'],
                leave=False,
                desc='    -th head',
            )
            for head_j in inner_bar_heads:
                pbar.set_description(
                    f'Processing edited model (l: {layer_i}/{config["n_layers"]}, h: {head_j}/{config["n_heads"]})'
                )
                if multi_token_generation:
                    # here the return value has already the softmaxed scored from the evaluator object
                    model_output = replace_heads_w_avg_multi_token(
                        tokenized_prompt=current_batch_tokens,
                        important_ids=current_batch_important_ids,
                        layers_heads=[(layer_i, head_j)],
                        avg_activations=[mean_activations[layer_i, head_j]],
                        model=model,
                        config=config,
                        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
                    )
                    # detokenize output, keeping as a list since can have different lenght (if batch is not used)
                    only_output_tokens = [
                        output.squeeze()[current_batch_tokens['input_ids'].shape[1] :] for output in model_output
                    ]
                    detokenize_outputs = [
                        tokenizer.decode(ele, skip_special_tokens=True) for ele in only_output_tokens
                    ]

                    # returns [torch.tensor([0.99, 0.01]), ...] like
                    evaluation_result = evaluator.get_evaluation(texts=detokenize_outputs, softmaxed=True)
                    results = torch.tensor([res[evaluator.negative_label] for res in evaluation_result])
                    edited[:, layer_i, head_j] = results

                else:
                    # here the returned value has shape [batch, vocab_size]
                    returned = replace_heads_w_avg(
                        tokenized_prompt=current_batch_tokens,
                        important_ids=current_batch_important_ids,
                        layers_heads=[(layer_i, head_j)],
                        avg_activations=[mean_activations[layer_i, head_j]],
                        model=model,
                        config=config,
                    )
                    edited[:, layer_i, head_j, :] = returned
            
        # end of the current batch
        probs_edited.append(edited.cpu())

    # stack all batch together
    if multi_token_generation:
        # here probs_original is a list of single tensors (the scores). E.g. [tensor(0), tensor(1), tensor(0)]
        probs_original = torch.hstack(probs_original)
    else:
        # here probs_original is a list of tensors with shape [batch, vocab_size]
        probs_original = torch.vstack(probs_original)

    probs_edited = torch.vstack(probs_edited)
    
    print(probs_original.shape)
    print(probs_edited.shape)
    
    if multi_token_generation:
        cie = torch.zeros([probs_original.shape[0], config['n_layers'], config['n_heads']])
        for prompt_idx in range(probs_original.shape[0]):
            for layer in range(config['n_layers']):
                for head in range(config['n_heads']):
                    cie[prompt_idx, layer, head] = probs_edited[prompt_idx, layer, head] - probs_original[prompt_idx]

        cie = cie.mean(dim=0)
    else:
        cie = compute_cie_single_token(
            tokenizer=tokenizer,
            config=config,
            correct_labels=all_correct_labels,
            probs_original=probs_original,
            probs_edited=probs_edited,
        )
    
    return cie, probs_original, probs_edited

