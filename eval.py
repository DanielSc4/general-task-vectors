import fire
import torch
import os
import random
from tqdm import tqdm


from src.utils.model_utils import set_seed, get_top_attention_heads, load_gpt_model_and_tokenizer
from src.utils.prompt_helper import load_json_dataset, tokenize_ICL
from src.intervention import replace_heads_w_avg



def evaluate_tv_zeroshot(
    mean_activation: torch.tensor,
    top_heads: list[tuple[int, int]],
    model,
    tokenizer,
    config,
    prompts_from_dataset,
    print_examples: bool = True,
):
    all_tokenized_prompt, all_important_ids, all_correct_labels = tokenize_ICL(
        tokenizer=tokenizer,
        ICL_examples=0,
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
    eval_dim: int = 10,
    load_in_8bit: bool = True,
):

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

    top_heads = get_top_attention_heads(cie, num_heads=10)

    idx_for_eval = random.sample(range(len(dataset)), eval_dim)

    evaluate_tv_zeroshot(
        mean_activation=mean_activations,
        top_heads=top_heads,
        model=model,
        tokenizer=tokenizer,
        config=config,
        prompts_from_dataset=[dataset[i] for i in idx_for_eval]
    )

if __name__ == "__main__":
    fire.Fire(main)