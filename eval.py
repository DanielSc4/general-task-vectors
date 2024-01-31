import fire
import torch
import os
import random

from src.utils.model_utils import set_seed, get_top_attention_heads, load_gpt_model_and_tokenizer
from src.utils.prompt_helper import load_json_dataset


def evaluate_tv_zeroshot(
    mean_activation: torch.tensor,
    top_heads: list[tuple[int, int]],
    model,
    tokenizer,
    config,
    prompts,
):
    pass



def main(
    model_name: str,
    dataset_name: str,
    mean_activation_path: str,
    cie_path: str,
    task_vector_eval_dim: int = 40,
    load_in_8bit: bool = False,
):

    dataset = load_json_dataset(f'.data/{dataset_name}.json')

    torch.set_grad_enabled(False)
    set_seed(32)

    # load model, tokenizer and config
    model, tokenizer, config, device = load_gpt_model_and_tokenizer(model_name, load_in_8bit)

    print(f'[x] Loading mean_activations')
    mean_activations = torch.load(mean_activation_path).to(device)
    print(f'[x] Loading CIE matrix')
    cie = torch.load(cie_path).to(device)

    top_heads = get_top_attention_heads(cie, num_heads=10)

    idx_for_eval = random.samele(range(len(dataset)), task_vector_eval_dim)


if __name__ == "__main__":
    fire.Fire(main)