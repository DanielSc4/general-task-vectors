import fire
import torch
from pathlib import Path
import json
import os

from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.model_utils import load_gpt_model_and_tokenizer, set_seed
from src.extraction import get_mean_activations
from src.utils.prompt_helper import tokenize_ICL
from src.intervention import compute_indirect_effect

def load_json_dataset(json_path):
    with open(json_path, encoding='utf-8') as file:
        dataset = json.load(file)
    return dataset


def main(
        model_name: str = 'gpt2',
        load_in_8bit: bool = False,
        dataset_name: str = 'following',
        icl_examples: int = 4,
        batch_size: int = 12,
):
    # create directory for storage
    Path('./output/').mkdir(parents=True, exist_ok=True) 

    # load dataset
    dataset = load_json_dataset(f'./data/{dataset_name}.json')
    dataset = list(map(lambda x: tuple(x.values()), dataset))
    print(f'[x] Loading dataset, len: {len(dataset)}')


    torch.set_grad_enabled(False)
    set_seed(32)

    # load model, tokenizer and config
    model, tokenizer, config, device = load_gpt_model_and_tokenizer(model_name, load_in_8bit)

    print(f'{model_name} on {device} device')
    
    # generate prompts
    tok_ret, ids_ret, correct_labels = tokenize_ICL(
        tokenizer, ICL_examples = icl_examples, dataset = dataset,
    )

    # get mean activations from the model (or stored ones if already exist)
    path_to_mean_activations = f'./output/{dataset_name}_mean_activations_{model_name.replace("/", "-")}.pt'

    if os.path.isfile(path_to_mean_activations):
        print(f'[x] Found mean_activations at: {path_to_mean_activations}')
        mean_activations = torch.load(path_to_mean_activations)
        mean_activations = mean_activations.to(device)
    else:
        mean_activations = get_mean_activations(
            tokenized_prompts=tok_ret,
            important_ids=ids_ret,
            tokenizer=tokenizer,
            model=model,
            config=config,
            correct_labels=correct_labels,
            device=device,
        )
        # store mean_activations
        torch.save(mean_activations, path_to_mean_activations)
    
    # compute causal mediation analysis over attention heads
    cie, probs_original, probs_edited  = compute_indirect_effect(
        model=model,
        tokenizer=tokenizer,
        config=config,
        dataset=dataset, 
        mean_activations=mean_activations,
        ICL_examples = icl_examples,
        batch_size=batch_size,
    )
    torch.save(cie, f'./output/{dataset_name}_cie_{model_name.replace("/", "-")}.pt')


if __name__ == "__main__":
    fire.Fire(main)

