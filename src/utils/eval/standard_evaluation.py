import torch
import random
from tqdm import tqdm
import numpy as np


from src.utils.prompt_helper import tokenize_ICL, randomize_dataset
from src.intervention import replace_heads_w_avg, simple_forward_pass


def calculate_accuracy(probs: dict[str, list], labels: list, top_k_accuracy: bool | int = False):
    # probs fileds:
    #   'argmax_token': [],     # actual predicted token
    #   'argmax_token_prob': [],     # actual predicted token probability
    #   'gold_token_prob': [],       # probability assigned to the gold token
    #   'top_k_tokens': [],     # top x tokens predicted

    corrects = 0
    k_corrects = 0
    for idx, label in enumerate(labels):
        if label == probs['argmax_token'][idx]:
            corrects += 1
        if top_k_accuracy:
            if label in probs['top_k_tokens'][idx]:
                k_corrects += 1
    return {
        'accuracy': corrects / len(labels),
        'top_k_accuracy': k_corrects / len(labels) if top_k_accuracy else None
    }


def evaluate_tv_kshot(
    mean_activation: torch.Tensor,
    top_heads: list[tuple[int, int]],
    model,
    tokenizer,
    config,
    prompts_from_dataset: list[tuple[str, str]],
    corrupted_ICL: bool = False,
    ICL_examples: int = 0,
    print_examples = True,
):
    """Evaluate the original and edited model with the task vector. Can perform evaluation for zero-shot ICL examples and 
    k-shot corrupted ICL examples, according to the `corrupted_ICL` parameter.

    Args:
        mean_activation (torch.tensor): average activations extracted
        top_heads (list[tuple[int, int]]): list of tuples each containing the layer and head index to be replaced with the average activation (task vector)
        model (_type_): HuggingFace model
        tokenizer (_type_): AutoTokenizer
        config (_type_): Config from the load function
        prompts_from_dataset (_type_): evaluation subset from the dataset
        corrupted_ICL (bool, optional): If false evaluation is carried out with zero-shots prompt, if True a corrupted ICL prompt is used instead. Defaults to False.
        ICL_examples (int, optional): Number of ICL examples (final query excluded) to be used for corrupted ICL prompts. Must be > 0 if `corrupted_ICL` is True. Defaults to 0.
        print_examples (bool, optional): whether to print a result subset at the end. Defaults to True.
    """

    assert ICL_examples > 0 if corrupted_ICL else True, f'ICL_examples must be > 0 for corrupted ICL examples. {ICL_examples} were given.'
    assert ICL_examples == 0 if not corrupted_ICL else True, f'ICL_examples must be 0 when using corrupted_ICL = False (zero-shot evaluation) {ICL_examples} were given.'

    # shuffle dataset
    if corrupted_ICL:
        prompts_from_dataset = randomize_dataset(prompts_from_dataset)

    all_tokenized_prompt, all_important_ids, all_correct_labels = tokenize_ICL(
        tokenizer=tokenizer,
        ICL_examples=ICL_examples,
        dataset=prompts_from_dataset,
    )
    # tokenize correct labels and get the first token
    all_correct_labels = torch.tensor(
        [tokenizer(label)['input_ids'][0] for label in all_correct_labels]
    )


    # original forward pass
    probs_original = {
        'argmax_token': [],     # actual predicted token
        'argmax_token_prob': [],     # actual predicted token probability
        'gold_token_prob': [],       # probability assigned to the gold token
        'top_k_tokens': [],     # top x tokens predicted
    }
    probs_edited = {
        'argmax_token': [],     # actual predicted token
        'argmax_token_prob': [],     # actual predicted token probability
        'gold_token_prob': [],       # probability assigned to the gold token
        'top_k_tokens': [],     # top x tokens predicted
    }

    # forward pass
    pbar = tqdm(
        zip(all_tokenized_prompt, all_important_ids, all_correct_labels),
        total=len(all_tokenized_prompt),
        desc='[x] Doing forward pass for both models',
    )
    for prompt, imp_ids, gold_token in pbar:
        # keeping batchsize = 1 for semplicity
        softmaxed = simple_forward_pass(model=model, prompt=prompt).cpu().squeeze()

        probs_original['argmax_token'].append(softmaxed.argmax().item())
        probs_original['argmax_token_prob'].append(softmaxed[softmaxed.argmax()].item())
        probs_original['gold_token_prob'].append(softmaxed[gold_token].item())
        probs_original['top_k_tokens'].append(torch.topk(softmaxed, k=5, axis=-1)[1].tolist())

        # edited model
        probs_task = replace_heads_w_avg(
            tokenized_prompt=prompt,
            important_ids=[imp_ids],
            layers_heads=top_heads,
            avg_activations=[mean_activation[i, j] for i, j in top_heads],
            model=model,
            config=config,
            last_token_only=True,
        )
        # squeeze batch_size
        softmaxed = probs_task.cpu().squeeze()
        probs_edited['argmax_token'].append(softmaxed.argmax().item())
        probs_edited['argmax_token_prob'].append(softmaxed[softmaxed.argmax()].item())
        probs_edited['gold_token_prob'].append(softmaxed[gold_token].item())
        probs_edited['top_k_tokens'].append(torch.topk(softmaxed, k=5, axis=-1)[1].tolist())

    # calculate accuracy
    accuracy_original = calculate_accuracy(probs_original, all_correct_labels, top_k_accuracy=True)
    accuracy_edited = calculate_accuracy(probs_edited, all_correct_labels, top_k_accuracy=True)
    
    print('\n------ Results ------')
    print(f'[v] Accuracy of the original model: {accuracy_original["accuracy"]:.2f}')
    print(f'[v]   top {len(probs_original["top_k_tokens"][0])} accuracy: {accuracy_original["top_k_accuracy"]:.2f}')

    print(f'[v] Accuracy of the edited model: {accuracy_edited["accuracy"]:.2f}')
    print(f'[v]   top {len(probs_edited["top_k_tokens"][0])} accuracy: {accuracy_edited["top_k_accuracy"]:.2f}')
    print()


    print('\n------ Examples ------')
    if print_examples:
        # print out min(10, 10% of the dataset) examples
        idx_to_print = random.sample(
            range(len(all_tokenized_prompt)),
            min(10, int(np.ceil(0.1 * len(all_correct_labels)))),
        )

        for idx, prompt in enumerate([
            all_tokenized_prompt[i] for i in idx_to_print
        ]):
            print('Prompt: ' + tokenizer.decode(all_tokenized_prompt[idx], skip_special_tokens = True).replace("\n", " "))
            print(f'  Gold: "{tokenizer.decode(all_correct_labels[idx])}", (token: {all_correct_labels[idx]})')
            print(f'  Original out:')
            print(f'    predicted token: "{tokenizer.decode(probs_original["argmax_token"][idx])}" with a probability of {probs_original["argmax_token_prob"][idx]:.2f}')
            print(f'    gold token has a proability of {probs_original["gold_token_prob"][idx]:.2f}')
            print(f'    top {len(probs_original["top_k_tokens"][idx])} tokens predicted: {[tokenizer.decode(ele) for ele in probs_original["top_k_tokens"][idx]]}')
            print(f'  Edited out:')
            print(f'    predicted token: "{tokenizer.decode(probs_edited["argmax_token"][idx])}" with a probability of {probs_edited["argmax_token_prob"][idx]:.2f}')
            print(f'    gold token has a proability of {probs_edited["gold_token_prob"][idx]:.2f}')
            print(f'    top {len(probs_edited["top_k_tokens"][idx])} tokens predicted: {[tokenizer.decode(ele) for ele in probs_edited["top_k_tokens"][idx]]}')
            print()

