import fire
import torch
import random
import json
import os
from src.utils.eval.multi_token_evaluator import Evaluator

from src.utils.model_utils import set_seed, get_top_attention_heads, load_model_and_tokenizer
from src.utils.prompt_helper import load_json_dataset
from src.utils.eval.general_evaluation_fun import evaluate_tv_multi_token


def main(
    model_name: str,
    dataset_name: str,
    mean_activation_name: str,
    cie_name: str,
    top_n_heads:int = 10,
    eval_dim: int = 111,
    load_in_8bit: bool = True,
):
    """Evalutate task vectors using model, mean_activations and selecting top_n_heads from cie
    Always use a 0-shot prompt for evaluation.

    Args:
        model_name (str): model name from HuggingFace
        dataset_name (str): dataset name without extension (.json)
        mean_activation_name (str): mean_activation file name.
        cie_name (str): cie file name.
        top_n_heads (int, optional): number of top heads to select. Defaults to 10.
        eval_dim (int, optional): evaluation dataset dimension. Defaults to 10.
        load_in_8bit (bool, optional): loads the model in 8bit for computational efficency. Defaults to True.
        print_examples (bool, optional): print a summary and some examples at the end of the evaluation. Defaults to True.
    """
    path_to_output = f'./output/{model_name.split("/")[1]}/{dataset_name}'
    # check if path_to_output exists
    assert os.path.exists(path_to_output), f'Be sure that {path_to_output} exists'

    mean_activation_path = os.path.join(path_to_output, mean_activation_name)
    cie_path = os.path.join(path_to_output, cie_name)
    output_path = os.path.join(path_to_output, 'output_evaluation.json')


    dataset = load_json_dataset(f'./data/{dataset_name}.json')
    dataset = list(map(lambda x: tuple(x.values()), dataset))

    torch.set_grad_enabled(False)
    set_seed(32)

    # load model, tokenizer and config
    model, tokenizer, config, device = load_model_and_tokenizer(model_name, load_in_8bit)

    print(f'[x] Loading mean_activations')
    mean_activations = torch.load(mean_activation_path).to(device)
    print(f'[x] Loading CIE matrix')
    cie = torch.load(cie_path).to(device)

    top_heads = get_top_attention_heads(cie, num_heads=top_n_heads)

    idx_for_eval = random.sample(range(len(dataset)), eval_dim)

    evaluator = Evaluator('meta-llama/LlamaGuard-7b', load_in_8bit=True)
    label_of_interest = evaluator.positive_label


    results = evaluate_tv_multi_token(
        mean_activation=mean_activations,
        top_heads=top_heads,
        model=model,
        tokenizer=tokenizer,
        config=config,
        prompts_from_dataset=[dataset[i] for i in idx_for_eval],
        evaluator=evaluator,
        label_of_interest=label_of_interest,
    )

    # Print here the evaluation results
    print('------- Evaluation results -------')
    original_labels = [ele['original']['eval']['label'] for ele in results]
    edited_output = [ele['edited']['eval']['label'] for ele in results]

    print(f'Label of interest: {label_of_interest}')
    print(f'Original prompts following label of interest: {original_labels.count(label_of_interest) / len(results) :.2f}')
    print(f'Edited prompts following label of interest: {edited_output.count(label_of_interest) / len(results) :.2f}')

    # save result to json file
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    fire.Fire(main)

