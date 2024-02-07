import torch
import random
import json


def load_json_dataset(json_path):
    with open(json_path, encoding='utf-8') as file:
        dataset = json.load(file)
    return dataset


def build_prompt_txt(queries: list[str], answers: list[str]):
    """Build the prompt following the default template. Provide a list of queries (length = n ICL examples + 1)
    and a list of answers (length = n ICL examples)

    Args:
        queries (list[str]): queries (ICL examples + final query)
        answers (list[str]): answers (ICL examples)

    Returns:
        full_prompt: full prompt following the default template with notation
    """
    full_prompt = []

    # prompt default template for structural parts
    begin = [('Q:', 'structural'),]
    middle = [('\nA:', 'structural'),]
    end = [('\n\n', 'structural'),]

    for i in range(len(answers)):
        full_prompt.extend(begin)
        full_prompt.append(
            (queries[i], 'sentence')
        )
        full_prompt.extend(middle)
        full_prompt.append(
            (answers[i], 'sentence')
        )
        full_prompt.extend(end)
    # append the final query without the answer
    full_prompt.extend(begin)
    full_prompt.append(
        (queries[-1], 'sentence')
    )
    full_prompt.extend(middle)

    return full_prompt



def tokenize_from_template(tokenizer, promtp_w_template: tuple[tuple[str, str]] = None):
    """tokenize the prompt following the provided template and return a list of indexes referring to the structural toknes
    and only the last token of sentences (automatically include the bos token at the beginning)

    Args:
        tokenizer (*Tokenizer): huggingface tokenizer
        promtp_w_template (tuple[tuple[str, str]], optional): prompt_template in the form of:
        ```python
            TODO
        ```. Defaults to None.

    Returns:
        torch.LongTensor: tokenized input ids
        list: index of every structural token and last sentence token
    """
    
    full_tokenized = torch.LongTensor([tokenizer.bos_token_id])
    indexes = [0]
    for prompt_ele in promtp_w_template:
        tokenized = tokenizer(prompt_ele[0], return_tensors='pt').input_ids[0]
        full_tokenized = torch.cat(
            (full_tokenized, tokenized), dim = -1,
        )
        if prompt_ele[1] == 'structural':
            # all the index of structural tokens must be included
            actual_idxs = list(
                range(indexes[-1] + 1, len(tokenized) + indexes[-1] + 1)
            )
            indexes.extend(actual_idxs)
        elif prompt_ele[1] == 'sentence':
            # include only the last index of the sentence tokens
            indexes.append(indexes[-1] + len(tokenized))

    return full_tokenized, indexes


def tokenize_ICL(tokenizer, ICL_examples: int, dataset: list[tuple[str, str]]):
    """build ICL prompt from the dataset, tokenize them and return the tokenized prompt with the important ids.

    Args:
        tokenizer (HuggingFace tokenizer): tokenizer from HuggingFace
        ICL_examples (int): number of ICL examples (excluding the last one without the solution)
        dataset (list[tuple[str, str, optional str]]): list of tuples (query, answer) default or (query, wrong_answer, correct_answer) if the labels are shufflet to trick the model

    Returns:
        tuple[list[torch.LongTensor], list[list[int]]]: tokenied prompt and important ids for each prompt
    """
    assert len(dataset) > ICL_examples, f'dataset dimension ({len(dataset)}) is <= ICL_examples ({ICL_examples})'

    prompts = []
    for i in range(0, len(dataset), ICL_examples + 1):
        # select examples to end up in the prompt
        group = dataset[i : i + ICL_examples + 1]
        
        if len(group) > ICL_examples:
            # enough ICL examples in the split (or group), otherwise don't use them
            X = build_prompt_txt(
                queries=list(map(lambda x: x[0], group)),
                answers=list(map(lambda x: x[1], group))[:-1],
            )
            # store prompt (X) and label (placed in 
            #    the last position (pos group[-1][1] if (query, answer) is provided 
            #    or pos group[-1][2] if (query, wrong_answer, correct_answer); 
            # using -1 to get both of them.
            prompts.append(
                (X, group[-1][-1])  
            )
        
    tokenized_and_ids = list(map(
        lambda prmpt: tokenize_from_template(tokenizer=tokenizer, promtp_w_template=prmpt[0]),
        prompts
    ))
    
    labels = list(map(lambda x: x[1], prompts))
    tokenized, ids = zip(*tokenized_and_ids, )

    return tokenized, ids, labels


def randomize_dataset(dataset):
    """shuffle the second column (labels) and copy the original column to a third one keeping the correct label
    e.g. for antonym: (good, bad) -> (good, funny, bad)

    Args:
        dataset (_type_): _description_

    Returns:
        _type_: _description_
    """
    shuffled = list(map(lambda x: x[1], dataset))
    random.shuffle(shuffled)

    new_dataset = list(
        zip(
            list(map(lambda x: x[0], dataset)),     # input x
            shuffled,     # new shuffled label (that make no sense)
            list(map(lambda x: x[1], dataset)),     # old correct label
    ))
    
    return new_dataset



def pad_input_and_ids(tokenized_prompts, important_ids: list[int], max_len = 256, pad_token_id = 50256):
    """pad a batched input 

    Args:
        tokenized_prompts (list[torch.Tensor]): list of tokenized sentences
        important_ids (list[int]). Important ids to be shifted according to the pad length
        max_len (int, optional): max len to pad. Defaults to 256.
        pad_token_id (int, optional): as name. Defaults to tokenizer.eos_token_id.

    Returns:
        dict[torch.Tensor, torch.Tensor]: dict with padded sentences and corresponding attention mask
    """
    padded_prompts = []
    attention_masks = []
    adapted_ids = []

    # Process each tokenized prompt individually
    for tokenized_prompt, imp_ids in zip(tokenized_prompts, important_ids):
        padded_prompt = torch.nn.functional.pad(
            tokenized_prompt,
            pad=(max_len - len(tokenized_prompt), 0),
            value=pad_token_id,
        )
        padded_prompts.append(padded_prompt)

        attention_mask = torch.zeros(max_len, dtype=torch.long)
        attention_mask[- len(tokenized_prompt):] = 1
        attention_masks.append(attention_mask)

        adapted_ids.append(
            [ele + sum(attention_mask == 0).item() for ele in imp_ids]
        )
        
    padded_prompts_tensor = torch.vstack(padded_prompts)
    attention_masks_tensor = torch.vstack(attention_masks)

    return {
        "input_ids": padded_prompts_tensor,
        "attention_mask": attention_masks_tensor,
    }, adapted_ids


def find_missing_ranges(lst: list[int]):
    """
    Given a list of important_ids (e.g. [0, 3, 4, 5, 6, 8]) find the missing ranges
    to average on ([1, 3], [7, 8])
    """
    ranges = [(start, end) for start, end in zip(lst, lst[1:]) if start + 1 < end]
    return [[start + 1, end] for start, end in ranges]
