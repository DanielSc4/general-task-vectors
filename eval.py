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
    probs_original = []
    pbar = tqdm(
        all_tokenized_prompt,
        total=len(all_tokenized_prompt),
        desc='Zero-shot forward pass on original model',
    )
    for prompt in pbar:
        # keeping batchsize = 1 for semplicity
        with model.invoke(prompt) as invoker:
            pass    # no action required
        logits = invoker.output.logits[:, -1, :]    # getting only the predicted token (i.e. final token), keeping batchsize and vocab_size
        probs_original.append(
            logits.softmax(dim=-1).cpu().argmax(dim=1).item()
        )
    probs_original = torch.tensor(probs_original)
   
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
    probs_task = probs_task.cpu().argmax(dim=1)


    # print out results
    for prompt, correct, original_out, edited_out in zip(
        all_tokenized_prompt,
        all_correct_labels, 
        probs_original, 
        probs_task
    ):
        print('Prompt: ' + tokenizer.decode(prompt, skip_special_tokens = True).replace("\n", " "))
        print(f'\t Gold: {tokenizer.decode(correct)}')
        print(f'\t Original out: {tokenizer.decode(original_out)}')
        print(f'\t Edited out: {tokenizer.decode(edited_out)}')



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