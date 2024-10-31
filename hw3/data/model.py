# Load model directly
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# Using Google's original version of Bert. 
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
model = AutoModelForQuestionAnswering.from_pretrained("google-bert/bert-base-uncased")

def train(model, optimizer, loss_fn):
    pass

if __name__=='__main__': 
    #tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-large-uncased-whole-word-masking-finetuned-squad")
    """     model = AutoModelForQuestionAnswering.from_pretrained("google-bert/bert-large-uncased-whole-word-masking-finetuned-squad")
    print(model)
 """


    tokens = tokenizer('My name is hidden. hello hello hello hello Hello. How are you doing. Goodbye.')
    print(tokens)

