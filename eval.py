import fire
import torch
import os
import random
from tqdm import tqdm


from src.utils.model_utils import set_seed, get_top_attention_heads, load_gpt_model_and_tokenizer
from src.utils.prompt_helper import load_json_dataset, tokenize_ICL, randomize_dataset
from src.intervention import replace_heads_w_avg



def evaluate_tv_kshot(
    mean_activation: torch.tensor,
    top_heads: list[tuple[int, int]],
    model,
    tokenizer,
    config,
    prompts_from_dataset,
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
    # tokenizer correct labels and get the first token
    all_correct_labels = torch.tensor(
        [tokenizer(label)['input_ids'][0] for label in all_correct_labels]
    )


    # original forward pass
    probs_original = {
        'argmax_token': [],     # actual predicted token
        'argmax_token_prob': [],     # actual predicted token probability
        'gold_token_prob': [],       # probability assigned to the gold token
        'top_x_tokens': [],     # top x tokens predicted
    }
    pbar = tqdm(
        zip(all_tokenized_prompt, all_correct_labels),
        total=len(all_tokenized_prompt),
        desc='Zero-shot forward pass on original model',
    )
    for prompt, gold_token in pbar:
        # keeping batchsize = 1 for semplicity
        with model.invoke(prompt) as invoker:
            pass    # no action required
        logits = invoker.output.logits[:, -1, :]    # getting only the predicted token (i.e. final token), keeping batchsize and vocab_size
        softmaxed = logits.softmax(dim=-1).cpu().squeeze()
        probs_original['argmax_token'].append(softmaxed.argmax().item())
        probs_original['argmax_token_prob'].append(softmaxed[softmaxed.argmax()].item())
        probs_original['gold_token_prob'].append(softmaxed[gold_token].item())
        probs_original['top_x_tokens'].append(torch.topk(softmaxed, k=5, axis=-1)[1].tolist())

   
    # edited model forward pass
    probs_task = replace_heads_w_avg(
        tokenized_prompt=all_tokenized_prompt,
        important_ids=all_important_ids,
        layers_heads=top_heads,
        avg_activations=[mean_activation[i, j] for i, j in top_heads],
        model=model,
        config=config,
        last_token_only=True,
    )
    probs_edited = {
        'argmax_token': [],     # actual predicted token
        'argmax_token_prob': [],     # actual predicted token probability
        'gold_token_prob': [],       # probability assigned to the gold token
        'top_x_tokens': [],     # top x tokens predicted
    }
    for softmaxed, gold_token in zip(probs_task.cpu(), all_correct_labels):
        probs_edited['argmax_token'].append(softmaxed.argmax().item())
        probs_edited['argmax_token_prob'].append(softmaxed[softmaxed.argmax()].item())
        probs_edited['gold_token_prob'].append(softmaxed[gold_token].item())
        probs_edited['top_x_tokens'].append(torch.topk(softmaxed, k=5, axis=-1)[1].tolist())

    if print_examples: 
        # print out results
        for idx, prompt in enumerate(all_tokenized_prompt):
            print('Prompt: ' + tokenizer.decode(prompt, skip_special_tokens = True).replace("\n", " "))
            print(f'  Gold: "{tokenizer.decode(all_correct_labels[idx])}"')
            print(f'  Original out:')
            print(f'    predicted token: "{tokenizer.decode(probs_original["argmax_token"][idx])}" with a probability of {probs_original["argmax_token_prob"][idx]:.2f}')
            print(f'    gold token has a proability of {probs_original["gold_token_prob"][idx]:.2f}')
            print(f'    top {len(probs_original["top_x_tokens"][idx])} tokens predicted: {[tokenizer.decode(ele) for ele in probs_original["top_x_tokens"][idx]]}')
            print(f'  Edited out:')
            print(f'    predicted token: "{tokenizer.decode(probs_edited["argmax_token"][idx])}" with a probability of {probs_edited["argmax_token_prob"][idx]:.2f}')
            print(f'    gold token has a proability of {probs_edited["gold_token_prob"][idx]:.2f}')
            print(f'    top {len(probs_edited["top_x_tokens"][idx])} tokens predicted: {[tokenizer.decode(ele) for ele in probs_edited["top_x_tokens"][idx]]}')
            print()





def main(
    model_name: str,
    dataset_name: str,
    mean_activation_path: str,
    cie_path: str,
    top_n_heads:int = 10,
    eval_dim: int = 111,
    load_in_8bit: bool = True,
    corrupted_ICL: bool = False,
    ICL_examples: int = 0,
    print_examples: bool = True,
):
    """Evalutate task vectors using model, mean_activations and selecting top_n_heads from cie

    Args:
        model_name (str): model name from HuggingFace
        dataset_name (str): dataset name without extension (.json)
        mean_activation_path (str): mean_activation file
        cie_path (str): cie file
        eval_dim (int, optional): evaluation dataset dimension. Defaults to 10.
        load_in_8bit (bool, optional): loads the model in 8bit for computational efficency. Defaults to True.
        corrupted_ICL (bool, optional): If False evaluate the model using zero-shot prompts, if true evaluates the model using a corrupted ICL prompt. Remember to choose ICL_examples > 0. Defaults to False.
        ICL_examples (int, optional): Number of ICL examples in the corrupted prompt. Must be zero if corrupted_ICL is False (zero-shot evaluation). Defaults to 0.
        print_examples (bool, optional): print a summary and some examples at the end of the evaluation. Defaults to True.
    """

    dataset = load_json_dataset(f'./data/{dataset_name}.json')
    dataset = list(map(lambda x: tuple(x.values()), dataset))

    torch.set_grad_enabled(False)
    set_seed(32)

    # load model, tokenizer and config
    model, tokenizer, config, device = load_gpt_model_and_tokenizer(model_name, load_in_8bit)

    print(f'[x] Loading mean_activations')
    mean_activations = torch.load(mean_activation_path).to(device)
    print(f'[x] Loading CIE matrix')
    cie = torch.load(cie_path).to(device)

    top_heads = get_top_attention_heads(cie, num_heads=top_n_heads)

    idx_for_eval = random.sample(range(len(dataset)), eval_dim)

    evaluate_tv_kshot(
        mean_activation=mean_activations,
        top_heads=top_heads,
        model=model,
        tokenizer=tokenizer,
        config=config,
        prompts_from_dataset=[dataset[i] for i in idx_for_eval],
        corrupted_ICL=corrupted_ICL,
        ICL_examples=ICL_examples,
        print_examples=print_examples,
    )

if __name__ == "__main__":
    fire.Fire(main)