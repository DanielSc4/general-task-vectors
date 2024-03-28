import torch
import random
from tqdm import tqdm
import numpy as np
from src.utils.eval.multi_token_evaluator import Evaluator


from src.utils.prompt_helper import tokenize_ICL
from src.intervention import simple_forward_pass, replace_heads_w_avg_multi_token


def calculate_accuracy(probs: dict[str, list], labels: list, top_k_accuracy: bool | int = False):
    # probs fileds:
    #   'argmax_token': [],     # actual predicted token
    #   'argmax_token_prob': [],     # actual predicted token probability
    #   'gold_token_prob': [],       # probability assigned to the gold token
    #   'top_k_tokens': [],     # top x tokens predicted

    corrects = 0
    k_corrects = 0
    for idx, label in enumerate(labels):
        if label == probs['argmax_token'][idx]:
            corrects += 1
        if top_k_accuracy:
            if label in probs['top_k_tokens'][idx]:
                k_corrects += 1
    return {
        'accuracy': corrects / len(labels),
        'top_k_accuracy': k_corrects / len(labels) if top_k_accuracy else None
    }



def evaluate_tv_multi_token(
    mean_activation: torch.Tensor,
    top_heads: list[tuple[int, int]],
    model,
    tokenizer,
    config,
    prompts_from_dataset: list[tuple[str, str]],
    evaluator: Evaluator,
    label_of_interest: str | int | None = None,
):
    assert label_of_interest is not None, 'Please provide a label of interest for evaluation'
    
    tokenize_dict = tokenize_ICL(
        tokenizer=tokenizer,
        ICL_examples=0,
        dataset=prompts_from_dataset,
    )
    all_tokenized_prompt = tokenize_dict['tokenized_prompts']
    all_important_ids = tokenize_dict['important_ids']

    results = []
    pbar = tqdm(
        zip(all_tokenized_prompt, all_important_ids),
        total=len(all_tokenized_prompt),
        desc='[x] Generating responses'
    )
    # keeping batchsize =1 to allow for any output dim.
    for prompt, _ in pbar:
        original_output = simple_forward_pass(
            model=model, 
            prompt=prompt, 
            pad_token_id=tokenizer.pad_token_id,
        ).cpu().squeeze()

        edited_output = replace_heads_w_avg_multi_token(
            tokenized_prompt=prompt,
            # important_ids=[imp_ids],
            layers_heads=top_heads,
            avg_activations=[mean_activation[i, j] for i, j in top_heads],
            model=model,
            config=config,
            pad_token_id=tokenizer.pad_token_id,
        ).squeeze()

        # evaluate both original and edited outputs
        only_output_original = tokenizer.decode(
            original_output[prompt.shape[0] :],
            skip_special_tokens=True,
        )
        
        only_output_edited = tokenizer.decode(
            edited_output[prompt.shape[0] :],
            skip_special_tokens=True,
        )

        decoded_prompt = tokenizer.decode(prompt, skip_special_tokens=True)

        # get evaluation of the original and edited outputs
        eval_results = evaluator.get_evaluation(
            prompts=[decoded_prompt] * 2,
            generations=[only_output_original, only_output_edited],
        )

        # = evaluation_result[label_of_interest][0] 

        struct_res = {
            "prompt": decoded_prompt,
            "original": {
                "output": only_output_original,
                "eval": {
                    "label": eval_results['output'][0],
                    "addition": eval_results['addition'][0],
                    label_of_interest: eval_results[label_of_interest][0],
                }
            },
            "edited": {
                "output": only_output_edited,
                "eval": {
                    "label": eval_results['output'][1],
                    "addition": eval_results['addition'][1],
                    label_of_interest: eval_results[label_of_interest][1],
                }
            }
        }

        results.append(struct_res)

    return results

