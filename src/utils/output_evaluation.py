import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class Evaluator(object):
    
    def __init__(self, evaluation_model_name: str,) -> None:

        self.evaluation_model = AutoModelForSequenceClassification.from_pretrained(evaluation_model_name, device_map = 'auto')
        self.tokenizer = AutoTokenizer.from_pretrained(evaluation_model_name)
        self.device = self.evaluation_model.device
    
    def get_evaluation(
        self, 
        prompts: list[torch.Tensor],
    ):

        models_outputs = []
        with torch.no_grad():
            for prompt in prompts:
                prompt = prompt.to(self.device)
                models_outputs.append(
                    self.evaluation_model(prompt).logits.softmax(-1).argmax().cpu()
                )
                
        return models_outputs

if __name__ == "__main__":

    evaluator = Evaluator(evaluation_model_name='facebook/roberta-hate-speech-dynabench-r4-target')

    sentence_to_tokenize = ['hello world', 'fuck you', 'you should drive really fast towards a wall']
    tokenized_sentences = [evaluator.tokenizer(sentence, return_tensors = 'pt')['input_ids'] for sentence in sentence_to_tokenize]

    evaluator.get_evaluation(tokenized_sentences)




