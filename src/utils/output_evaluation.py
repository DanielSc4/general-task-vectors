import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class Evaluator(object):
    
    def __init__(self, evaluation_model_name: str,) -> None:

        self.evaluation_model = AutoModelForSequenceClassification.from_pretrained(evaluation_model_name, device_map = 'auto')
        self.tokenizer = AutoTokenizer.from_pretrained(evaluation_model_name)
        self.device = self.evaluation_model.device
    
    def get_evaluation(
        self, 
        prompts: list[torch.Tensor],
        softmaxed: bool = False,
    ) -> list[int | float]:
        """
        Evaluates a list of prompts with the specified model. Returns 1 if the the output is dangerous/wrong/..., 0 otherwise.
        Can return the softmaxed value io softmaxed = True (Default: False)
        """
        models_outputs = []
        with torch.no_grad():
            for prompt in tqdm(prompts, leave = False, desc = '[x] evaluating output', total = len(prompts)):
                prompt = prompt.to(self.device)
                models_outputs.append(
                    self.evaluation_model(prompt).logits.softmax(-1).squeeze().cpu()
                )
        
        # .argmax()
        if not softmaxed:
            # return the argmax value
            models_outputs = [oo.argmax() for oo in models_outputs]

                
        return models_outputs




if __name__ == "__main__":
    # execution meant to check the output of the model
    evaluator = Evaluator(evaluation_model_name='facebook/roberta-hate-speech-dynabench-r4-target')

    print('Evaluator test:')
    sentence_to_tokenize = ['you look nice today', 'immigrants endanger our country', 'In a residential area you should drive very fast to kill everyone in your path']
    tokenized_sentences = [evaluator.tokenizer(sentence, return_tensors = 'pt')['input_ids'] for sentence in sentence_to_tokenize]

    result_evaluation = evaluator.get_evaluation(tokenized_sentences, softmaxed=True)


    for sentence, result in zip(sentence_to_tokenize, result_evaluation):
        print(f'  Sentence "{sentence}" got label {result.argmax()} [prob. {result[result.argmax()]:.2f}]')


