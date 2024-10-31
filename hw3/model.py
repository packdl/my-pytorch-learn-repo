# Load model directly
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import pandas as pd

from dataloader import BertDataset

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

# Using Google's original version of Bert. 
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
model = AutoModelForQuestionAnswering.from_pretrained("google-bert/bert-base-uncased")

def f1score(candidate:list, actual:list):
    tp = sum([a == b for a,b in zip(candidate, actual)])
    fp = sum([(word not in actual) for word in candidate])
    fn = sum([(word not in candidate) for word in actual ])

    precision = tp/(tp + fp)
    recall = tp/(tp + fn)

    if precision + recall == 0:
        return 0

    return (2*precision * recall) / (precision + recall )
        
    

def train(model, dataloader, optimizer, loss_fn):
    for data in dataloader:
        continue
        # (context, question, answer_text, (start, end), answer_start)
        

def test(model, dataloader, loss_fn):
    pass

if __name__=='__main__': 
    #tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-large-uncased-whole-word-masking-finetuned-squad")
    """     model = AutoModelForQuestionAnswering.from_pretrained("google-bert/bert-large-uncased-whole-word-masking-finetuned-squad")
    print(model)
 """


    tokens = tokenizer(('My name is hidden. hello hello hello hello Hello. How are you doing. Goodbye.','paint is a noun'))
    print(torch.tensor(tokens['input_ids']).to(device))
    #print(tokenizer.convert_ids_to_tokens(tokens['input_ids'], skip_special_tokens=True))

    #print(f1score('george washington'.split(), 'george washington'.split()))
    #print(f1score('george washington'.split(), 'washington is'.split()))
    #print(f1score('george washington'.split(), 'george washington carver'.split()))

    dl = DataLoader(BertDataset('data/spoken_test-v1.1.json'), batch_size=2, shuffle=True)
    

    """for data in dl:
        print(data)
        break"""

