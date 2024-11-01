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
    
    total_loss = 0
    num_batches = len(dataloader)
    f1s = []
    for data in dataloader:
        optimizer.zero_grad()
        context, question, answer_text, (start, end), _ = data
        batch_encoding = tokenizer(question, context, padding=True, truncation=True, return_tensors='pt')
        out = model(**batch_encoding, start_positions=start, end_positions=end)

        starts = torch.argmax(nn.Softmax()(out.start_logits), dim=1)
        stops = torch.argmax(nn.Softmax()(out.start_logits), dim=1)

        for c,actual_start,actual_end, candidate_start, candidate_end,  in enumerate(context, start, end, starts, ends):
            c = c.split()
            sample = c[actual_start:actual_end+1]
            candidate = c[candidate_start:candidate_end+1]

            f1s.append(f1score(candidate, sample))


        loss = out.loss
        total_loss += loss.item()
        loss.backwards()
        optimizer.step()

    total_loss /= num_batches
    return total_loss, sum(f1s)/len(f1s) 
        #sm_out = nn.Softmax()(out)
        # (context, question, answer_text, (start, end), answer_start)
        

def test(model, dataloader, loss_fn):
    pass

if __name__=='__main__': 
    
    batch_size = 32
    loss_fn = torch.nn.MSELoss()
    epochs = 10
    lr = .01
    batch_size=32
    # Need to add a scheduler!!!!
    # torch.optim.Adam()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    dl = DataLoader(BertDataset('data/spoken_test-v1.1.json'), batch_size=batch_size, shuffle=True)

    for epoch in epochs:
        pass


    
    
    
    
    
    #tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-large-uncased-whole-word-masking-finetuned-squad")

    """ a =('My name is hidden. hello hello hello hello Hello. How are you doing. Goodbye.','paint is a noun')
    batch_encoding = tokenizer(a,a, padding=True, truncation=True, return_tensors='pt')
    #print(batch_encoding['input_ids'])
    #print(tokenizer.convert_ids_to_tokens(tokens['input_ids'], skip_special_tokens=True))

    #print(f1score('george washington'.split(), 'george washington'.split()))
    #print(f1score('george washington'.split(), 'washington is'.split()))
    #print(f1score('george washington'.split(), 'george washington carver'.split()))

    dl = DataLoader(BertDataset('data/spoken_test-v1.1.json'), batch_size=32, shuffle=True)
    
    i = 0
    for data in dl:
        context, question, answer_text, (start, end), _ = data

        batch_encoding = tokenizer(question, context, padding=True, truncation=True, return_tensors='pt')
        out = model(**batch_encoding, start_positions=start, end_positions=end)

        #print(out)
        #print(f'{out.start_logits = }')
        #print(f'{out.loss = }')
        #print(f'{out.start_logits.shape}')

        starts = torch.argmax(nn.Softmax()(out.start_logits), dim=1)
        stops = torch.argmax(nn.Softmax()(out.start_logits), dim=1)
        
        i += 1
        
        if i==3:
            break

        print(f'{starts = }')
        print(f'{stops = }')
        print(f'{start = }')
        print(f'{end = }')
 """



        
        # (context, question, answer_text, (start, end), answer_start)


