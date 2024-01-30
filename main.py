import fire
import torch
from pathlib import Path
import json
import os
import random
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer
import seaborn as sns
import matplotlib.pyplot as plt

from src.utils.model_utils import load_gpt_model_and_tokenizer, set_seed
from src.extraction import get_mean_activations
from src.utils.prompt_helper import tokenize_ICL
from src.intervention import compute_indirect_effect

def load_json_dataset(json_path):
    with open(json_path, encoding='utf-8') as file:
        dataset = json.load(file)
    return dataset


def get_top_attention_heads(
    cie = torch.Tensor,
    num_heads: int = 15,
):
    """
    Get the indices of the top attention heads in CIE
    """
    # indeces of the top num_heads highest numbers
    flat_indices = np.argsort(cie.flatten())[-num_heads:]
    # convert flat indices to 2D incices
    top_indices = np.unravel_index(flat_indices, cie.shape)
    coordinates_list = list(zip(top_indices[0], top_indices[1]))

    # sort the list based on the corresponding values in descending order
    sorted_coordinates_list = sorted(
        coordinates_list, 
        key=lambda x: cie[x[0], x[1]], 
        reverse=True
    )

    return sorted_coordinates_list


def main(
    model_name: str = 'gpt2',
    load_in_8bit: bool = False,
    dataset_name: str = 'following',
    icl_examples: int = 4,
    batch_size: int = 12,
    mean_support: int = 100,
    aie_support: int = 25,
    save_plot: bool = True,
):
    # create directory for storage
    Path('./output/').mkdir(parents=True, exist_ok=True)
    if save_plot:
        Path('./output/plots').mkdir(parents=True, exist_ok=True)

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
    # create a subset with mean_support elements
    idx_for_mean = random.sample(range(len(tok_ret)), mean_support)
    selected_examples = [
        (tok_ret[i], ids_ret[i], correct_labels[i])
        for i in idx_for_mean
    ]
    tok_ret, ids_ret, correct_labels = zip(*selected_examples)

    # get mean activations from the model (or stored ones if already exist)
    path_to_mean_activations = f'./output/{dataset_name}_mean_activations_{model_name.replace("/", "-")}_ICL{icl_examples}.pt'

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
        aie_support=aie_support,
    )
    torch.save(cie, f'./output/{dataset_name}_cie_{model_name.replace("/", "-")}_ICL{icl_examples}.pt')

    print('[x] Done')

    if save_plot:
        print('[x] Generating CIE plot')
        ax = sns.heatmap(cie, linewidth=0.5, cmap='RdBu', center=0)
        plt.title(model_name.replace("/", "-"))
        plt.xlabel('head')
        plt.ylabel('layer')
        plt.savefig(f'./output/plots/{dataset_name}_cie_{model_name.replace("/", "-")}_ICL{icl_examples}.png')

    # selecting top_10 heads
    top_heads = get_top_attention_heads(cie, num_heads=10)



if __name__ == "__main__":
    fire.Fire(main)

