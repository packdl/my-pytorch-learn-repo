# Load model directly
import copy
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import transformers
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import pandas as pd

from dataloader import BertDataset

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
    
torch.set_default_device(device)
    
print(f'{device} is available')

# Using Google's original version of Bert. 

tokenizer = AutoTokenizer.from_pretrained("albert/albert-base-v2")
model = AutoModelForQuestionAnswering.from_pretrained("albert/albert-base-v2", torch_dtype=torch.float16).to(device)

#tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
#model = AutoModelForQuestionAnswering.from_pretrained("distilbert/distilbert-base-uncased", torch_dtype=torch.float16).to(device)

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
        
    

def train(model, dataloader, optimizer, scheduler):
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    f1s = []
    
    #accum_iter = 2  
    for batch, data in enumerate(dataloader):
        context, question, answer_text, (start, end), _ = data
        batch_encoding = tokenizer(question, context,max_length=512, padding='max_length', truncation='only_second',  return_tensors='pt')
        input_ids = batch_encoding['input_ids']
        
        #print(batch_encoding.keys())
        #print(start, end)
        masks = batch_encoding['attention_mask']
        # out = model(**batch_encoding, start_positions=start, end_positions=end)
        out = model(input_ids=input_ids, attention_mask = masks, start_positions=start, end_positions=end)

        starts = torch.argmax(nn.Softmax(dim=1)(out.start_logits), dim=1)
        ends = torch.argmax(nn.Softmax(dim=1)(out.end_logits), dim=1)

        for c,actual_start,actual_end, candidate_start, candidate_end,  in zip(batch_encoding['input_ids'], start, end, starts, ends):
            
            if candidate_end < candidate_start:
                continue
            c = c.tolist()
            
            c = tokenizer.convert_ids_to_tokens(c)
            if candidate_end > len(c):
                continue
            sample = c[actual_start:actual_end+1]
            candidate = c[candidate_start:candidate_end+1]

            f1s.append(f1score(candidate, sample))

        loss = out.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        
        
        #if ((batch +1) % accum_iter == 0) or (batch+1 == len(dataloader)):
        #    optimizer.step()
        #    optimizer.zero_grad()

    total_loss /= num_batches
    return total_loss, sum(f1s)/len(f1s), c, sample, candidate 
        #sm_out = nn.Softmax()(out)
        # (context, question, answer_text, (start, end), answer_start)
        

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


if __name__=='__main__': 
    
    #loss_fn = torch.nn.MSELoss()
    epochs = 10
    #lr = .00176
    lr = 5e-6
    #lr = 6e-4
    
    batch_size=8
    #optimizer = torch.optim.AdamW(model.parameters(), lr)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        
    #scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=.1, total_iters=epochs)
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=.99)
    train_dl = DataLoader(BertDataset('data/spoken_test-v1.1.json'), batch_size=batch_size, shuffle=True, generator=torch.Generator(device))
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, 10000, len(train_dl) * (epochs+1))
    
    
    
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        
    #scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=.1, total_iters=epochs)
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=.99)
    #scheduler = transformers.get_linear_schedule_with_warmup(optimizer, 0, epochs)
    train_dl = DataLoader(BertDataset('data/spoken_test-v1.1.json'), batch_size=batch_size, shuffle=True, generator=torch.Generator(device))
    scheduler = transformers.get_polynomial_decay_schedule_with_warmup(optimizer, 0, len(train_dl) * epochs, lr_end = 4.1e-06)
    #scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, 0, len(train_dl) * epochs, 3)
    val_dl = DataLoader(BertDataset('data/spoken_train-v1.1.json'), batch_size=batch_size, shuffle=True, generator=torch.Generator(device))
    
    best_model = {'epoch':0,'f1':0, 'params':None}
    
    epoch_list, train_loss_list, train_f1_list, val_loss_list, val_f1_list = [],[],[],[],[]
    
    for epoch in range(epochs):
        train_loss, train_f1, c, sample, candidate = train(model,train_dl, optimizer, scheduler)
        val_loss, val_f1 = validate(model, val_dl)
        
        epoch_list.append(epoch)
        train_loss_list.append(train_loss)
        train_f1_list.append(train_f1)
        val_loss_list.append(val_loss)
        val_f1_list.append(val_f1)
        
        print(f'Epoch {epoch}. Train_Loss: {train_loss}, Train_F1: {train_f1}, Val_loss: {val_loss}, Val_F1 {val_f1}')
        print(f'{c = }') 
        print(f'{sample = }')
        print(f'{candidate = }')
        print(f'{scheduler.get_last_lr() = }')
        if val_f1 > best_model['f1']:
            best_model['epoch'] = epoch
            best_model['f1'] = val_f1
            best_model['params'] = copy.deepcopy(model.state_dict())
        #scheduler.step()
        
    print(f"Best F1: {best_model['f1']}. Epoch: {best_model['epoch']}")
    torch.save(best_model['params'], 'albert_finetuned2.pth')
    df = pd.DataFrame({'epoch':epoch_list, 'train_loss':train_loss_list, 'train_f1':train_f1_list,'val_loss':val_loss_list,'val_f1':val_f1_list})
    df.to_csv('raw_data2.txt')