from typing import Any
import torch

from nnsight import LanguageModel
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from src.extraction import extract_activations
from src.utils.prompt_helper import tokenize_ICL


        
def get_attribution_patching(
    model: LanguageModel, 
    tokenizer: PreTrainedTokenizer,
    config: dict[str, Any],
    dataset: list[tuple[str, str]],
    ICL_examples: int = 4,
    ):

    # TODO: Remember to shuffle the dataset

    # using a subset of the dataset to create the prompts
    dataset = dataset[:10]
    tokenized_dict = tokenize_ICL(
        tokenizer=tokenizer,
        ICL_examples=ICL_examples,
        dataset=dataset,
        pre_append_instruction=None,
    )

    for idx in range(len(tokenized_dict['tokenized_prompts'])):
        # extract activations from the ICL_prompt forward pass
        activations_dict = extract_activations(
            tokenized_prompts=tokenized_dict['tokenized_prompts'][idx],
            model=model,
            config=config,
            tokenizer=tokenizer,
            return_gradient=False,
        )
        activations = activations_dict['activations']
        output = activations_dict['output']
        print(f'{activations.shape = }')
        print(f'{output.shape = }')

        print('-------')
        # extract activations and gradients from the NON ICL forward pass
        activations_dict_no_ICL = extract_activations(
            tokenized_prompts=tokenized_dict['tokenized_prompts_no_ICL'][idx],
            model=model,
            config=config,
            tokenizer=tokenizer,
            return_gradient=True,
        )
        activations_no_ICL = activations_dict_no_ICL['activations']
        output_no_ICL = activations_dict_no_ICL['output']
        gradients_no_ICL = activations_dict_no_ICL['gradients']
        print(f'{activations_no_ICL.shape = }')
        print(f'{output_no_ICL.shape = }')
        print(f'{gradients_no_ICL.shape = }')

        exit()
    

    return 0, 0



