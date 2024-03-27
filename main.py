import fire
import torch
from pathlib import Path
import os
import random
import warnings

# from transformers import AutoModelForCausalLM, AutoTokenizer
import seaborn as sns
import matplotlib.pyplot as plt

from src.utils.model_utils import load_model_and_tokenizer, set_seed 
from src.extraction import get_mean_activations
from src.utils.prompt_helper import tokenize_ICL, load_json_dataset
from src.intervention import compute_indirect_effect
from src.utils.eval.multi_token_evaluator import Evaluator

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
    pre_append_instruction: str | None = None,
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
        use_local_backups (bool, optional): when exists, do not compute mean_activation but get the backup in the `./output/` dir. Defaults to False.
        pre_append_instruction (str, optional): instruction to use before the prompt. Defaults to None.
    """
    # create directory for storage and models output
    path_to_output = f'./output/{model_name.split("/")[1]}/{dataset_name}'
    Path(path_to_output).mkdir(parents=True, exist_ok=True)

    path_to_mean_activations = os.path.join(path_to_output, f'mean_activations_icl{icl_examples}_sup{mean_support}.pt')
    path_to_cie = os.path.join(path_to_output, f'cie_ICL{icl_examples}_sup{aie_support}.pt')
    path_to_output_generation = os.path.join(path_to_output, f'output_mean_activations_icl{icl_examples}.json')
    path_to_output_all = os.path.join(path_to_output, f'output_intervention.json')
    path_to_plot = os.path.join(path_to_output, f'plot_{model_name.replace("/", "-")}_ICL{icl_examples}_sup{aie_support}.png')

    # load dataset
    dataset = load_json_dataset(f'./data/{dataset_name}.json')
    dataset = list(map(lambda x: tuple(x.values()), dataset))
    print(f'[x] Loading dataset, len: {len(dataset)}')

    if batch_size != 1:
        warnings.warn(f'batch_size set to {batch_size} not supported when using multi token generation. Setting batch_size to 1')
        batch_size = 1

    torch.set_grad_enabled(False)
    set_seed(32)

    # load model, tokenizer and config
    model, tokenizer, config, device = load_model_and_tokenizer(model_name, load_in_8bit)

    print(f'{model_name} on {device} device')
    
    # generate prompts
    tok_ret, ids_ret, correct_labels = tokenize_ICL(
        tokenizer, ICL_examples = icl_examples, dataset = dataset,
        pre_append_instruction=pre_append_instruction,
    )

    # create a subset with mean_support elements
    print(f'[x] New dataset dimension after ICL: {len(tok_ret)}')
    idx_for_mean = random.sample(range(len(tok_ret)), mean_support)
    selected_examples = [
        (tok_ret[i], ids_ret[i], correct_labels[i])
        for i in idx_for_mean
    ]
    tok_ret, ids_ret, correct_labels = zip(*selected_examples)

    # change default behaviour for evaluation strategy
    evaluator = Evaluator('meta-llama/LlamaGuard-7b', load_in_8bit=load_in_8bit)
    # set the label of interest
    label_of_interest = evaluator.positive_label

    # get mean activations from the model (or stored ones if already exist)
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
            batch_size=batch_size,
            evaluator=evaluator,
            label_of_interest=label_of_interest,
            save_output_path=path_to_output_generation,
        )
        # store mean_activations
        torch.save(mean_activations, path_to_mean_activations)
    

    # compute CIE
    cie, _, _  = compute_indirect_effect(
        model=model,
        tokenizer=tokenizer,
        config=config,
        dataset=dataset, 
        mean_activations=mean_activations,
        ICL_examples = icl_examples,
        batch_size=batch_size,
        aie_support=aie_support,
        evaluator=evaluator,
        label_of_interest=label_of_interest,
        save_output_path=path_to_output_all,
    )
    torch.save(cie, path_to_cie)
    print('[x] CIE output saved')


    if save_plot:
        print('[x] Generating CIE plot')
        _ = sns.heatmap(cie.cpu(), linewidth=0.5, cmap='RdBu', center=0)
        plt.title(f'plot_{model_name.replace("/", "-")}_support{aie_support}')
        plt.xlabel('head')
        plt.ylabel('layer')
        plt.savefig(path_to_plot)

if __name__ == "__main__":
    fire.Fire(main)

