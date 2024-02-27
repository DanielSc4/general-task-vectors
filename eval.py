import fire
import torch
import random
from src.utils.eval.multi_token_evaluator import Evaluator

from src.utils.model_utils import set_seed, get_top_attention_heads, load_model_and_tokenizer
from src.utils.prompt_helper import load_json_dataset
from src.utils.eval.general_evaluation_fun import evaluate_tv_kshot, evaluate_tv_multi_token


def main(
    model_name: str,
    dataset_name: str,
    mean_activation_path: str,
    cie_path: str,
    top_n_heads:int = 10,
    eval_dim: int = 111,
    load_in_8bit: bool = True,
    multi_token_evaluation: bool = False,
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
    model, tokenizer, config, device = load_model_and_tokenizer(model_name, load_in_8bit)

    print(f'[x] Loading mean_activations')
    mean_activations = torch.load(mean_activation_path).to(device)
    print(f'[x] Loading CIE matrix')
    cie = torch.load(cie_path).to(device)

    top_heads = get_top_attention_heads(cie, num_heads=top_n_heads)

    idx_for_eval = random.sample(range(len(dataset)), eval_dim)


    if multi_token_evaluation:
        evaluator = Evaluator('meta-llama/LlamaGuard-7b', load_in_8bit=True)
        
        evaluate_tv_multi_token(
            mean_activation=mean_activations,
            top_heads=top_heads,
            model=model,
            tokenizer=tokenizer,
            config=config,
            prompts_from_dataset=[dataset[i] for i in idx_for_eval],
            evaluator=evaluator,
            print_examples=print_examples,
        )
    else:
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

