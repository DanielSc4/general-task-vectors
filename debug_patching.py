
import torch
from src.utils.model_utils import load_model_and_tokenizer
from src.utils.prompt_helper import load_json_dataset
from src.patching import get_attribution_patching

if __name__ == '__main__':
    # test here
    ICL_prompt = "Q: What is the capital of France? A: Paris \n Q: What is the capital of Italy? A:"
    normal_prompt = "Q: What is the capital of Italy? A:"

    # load dataset
    dataset_name = 'cona-facts'
    dataset = load_json_dataset(f'./data/{dataset_name}.json')
    dataset = list(map(lambda x: tuple(x.values()), dataset))
    print(f'[x] Loading dataset, len: {len(dataset)}')
    
    # load model
    is_cuda = torch.cuda.is_available()
    model, tokenizer, config, _ = load_model_and_tokenizer(
        model_name='openai-community/gpt2',
        load_in_8bit=is_cuda,
    )

    tokenized_prompts = tokenizer(ICL_prompt, return_tensors='pt')

    _ = get_attribution_patching(
        model=model,
        tokenizer=tokenizer,
        config=config,
        dataset=dataset,
    )


