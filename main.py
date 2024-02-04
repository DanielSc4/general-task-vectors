import fire
import torch
from pathlib import Path
import os
import random
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer
import seaborn as sns
import matplotlib.pyplot as plt

from src.utils.model_utils import load_gpt_model_and_tokenizer, set_seed 
from src.extraction import get_mean_activations
from src.utils.prompt_helper import tokenize_ICL, load_json_dataset
from src.intervention import compute_indirect_effect


def main(
    model_name: str = 'gpt2',
    load_in_8bit: bool = False,
    dataset_name: str = 'following',
    icl_examples: int = 4,
    batch_size: int = 12,
    mean_support: int = 100,
    aie_support: int = 25,
    save_plot: bool = True,
    use_local_backups: bool = False,
):
    """Main function to get the mean_attention, CIE on model and zero-shot results

    Args:
        model_name (str, optional): model name as huggingface. Defaults to 'gpt2'.
        load_in_8bit (bool, optional): option to load the model in 8bit. Defaults to False.
        dataset_name (str, optional): name of the dataset (`.json` in `./data/` dir). Defaults to 'following'.
        icl_examples (int, optional): number of ICL examples, 0 for zero-shot. Defaults to 4.
        batch_size (int, optional): batch size for cie intervention. Defaults to 12.
        mean_support (int, optional): number of example to average over when computing mean_activation. Defaults to 100.
        aie_support (int, optional): number of example to average over when computning CIE matrix. Defaults to 25.
        task_vector_eval_dim (int, optional): number of example for the final zero-shot evaluation. Defaults to 40.
        save_plot (bool, optional): whether to save a plot of the CIE matrix in `./output/plot/` dir. Defaults to True.
        use_local_backups (bool, optional): when exists, do not compute mean_activation and CIE bt get the backup in the `./output/` dir. Defaults to False.
    """
    # create directory for storage and models output
    Path('./output/').mkdir(parents=True, exist_ok=True)
    path_to_output = f'./output/{model_name.split("/")[1]}'
    Path(path_to_output).mkdir(parents=True, exist_ok=True)

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
    path_to_mean_activations = f'{path_to_output}/{dataset_name}_mean_activations_{model_name.replace("/", "-")}_ICL{icl_examples}.pt'

    if os.path.isfile(path_to_mean_activations) and use_local_backups:
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
    

    # get mean activations from the model (or stored ones if already exist)
    path_to_cie = f'{path_to_output}/{dataset_name}_cie_{model_name.replace("/", "-")}_ICL{icl_examples}.pt'
    if os.path.isfile(path_to_cie) and use_local_backups:
        print(f'[x] Found CIE at {path_to_cie}')
        cie = torch.load(path_to_cie)
        cie = cie.to(device)
    else:
        # compute causal mediation analysis over attention heads
        cie, _, _  = compute_indirect_effect(
            model=model,
            tokenizer=tokenizer,
            config=config,
            dataset=dataset, 
            mean_activations=mean_activations,
            ICL_examples = icl_examples,
            batch_size=batch_size,
            aie_support=aie_support,
        )
        torch.save(cie, path_to_cie)
        print('[x] CIE output saved')

    if save_plot:
        print('[x] Generating CIE plot')
        ax = sns.heatmap(cie.cpu(), linewidth=0.5, cmap='RdBu', center=0)
        plt.title(model_name.replace("/", "-"))
        plt.xlabel('head')
        plt.ylabel('layer')
        plt.savefig(f'./output/plots/{dataset_name}_cie_{model_name.replace("/", "-")}_ICL{icl_examples}.png')

if __name__ == "__main__":
    fire.Fire(main)

