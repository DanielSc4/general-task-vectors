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
    # check what's inside the tokenized_dict
    for key, value in tokenized_dict.items():
        print(f'{key}: {value}')

    return 0, 0



