import torch
import transformers
from torch.utils.data import DataLoader
from transformers import AutoTokenizer,AutoModelForQuestionAnswering
import copy
from collections import Counter

import torch
import torch.nn as nn
import pandas as pd


from dataloader import BertDataset


if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
    
torch.set_default_device(device)

def f1score(candidate:list, actual:list):
    #The number of shared words between the prediction and the truth is the basis of the F1 score: precision is the ratio of the number of shared words to the total number of words in the prediction, and recall is the ratio of the number of shared words to the total number of words in the ground truth.
    
    shared_words = sum((Counter(candidate) & Counter(actual)).values())
    
    if shared_words == 0:
        return 0.0
    
    if len(candidate) == 0 or len(actual) ==0:
        return int(candidate == actual) * 100.0
    
    precision = (shared_words / len(candidate)) 
    recall = (shared_words / len(actual))
    
    if (precision + recall ) == 0:
        return 0.0
    
    return 100.0 * (2*precision * recall) / (precision + recall )

def validate(model, dataloader):
    model.eval()
    total_loss = 0
    num_batches = len(dataloader)
    f1s = []
    for data in dataloader:
        context, question, answer_text, (start, end), _ = data
        batch_encoding = tokenizer(question, context,max_length=512, padding='max_length', truncation='only_second', return_tensors='pt')
        input_ids = batch_encoding['input_ids']
        masks = batch_encoding['attention_mask']
        # out = model(**batch_encoding, start_positions=start, end_positions=end)
        out = model(input_ids=input_ids, attention_mask = masks, start_positions=start, end_positions=end)
        #batch_encoding = tokenizer(question, context, padding=True, truncation=True, return_tensors='pt')
        #out = model(**batch_encoding, start_positions=start, end_positions=end)

        starts = torch.argmax(nn.Softmax(dim=1)(out.start_logits), dim=1)
        ends = torch.argmax(nn.Softmax(dim=1)(out.end_logits), dim=1)

        for c,actual_start,actual_end, candidate_start, candidate_end,  in zip(batch_encoding['input_ids'], start, end, starts, ends):
            c = c.tolist()
            c = tokenizer.convert_ids_to_tokens(c)
            sample = c[actual_start:actual_end+1]
            candidate = c[candidate_start:candidate_end+1]

            f1s.append(f1score(candidate, sample))
        loss = out.loss
        total_loss += loss.item()

    total_loss /= num_batches
    return total_loss, sum(f1s)/len(f1s) 


tokenizer = AutoTokenizer.from_pretrained("albert/albert-base-v2")
model = AutoModelForQuestionAnswering.from_pretrained("albert/albert-base-v2")
model.load_state_dict(torch.load('albert_finetuned2-1.pth', weights_only=True))
model.eval()

batch_size=8

val_dl = DataLoader(BertDataset('data/spoken_train-v1.1.json'), batch_size=batch_size, shuffle=True, generator=torch.Generator(device))

loss, f1 = validate(model, val_dl)

print(f'{loss = }')
print(f'{f1 = }' )
