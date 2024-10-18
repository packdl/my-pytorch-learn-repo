import time
from pathlib import Path
import copy

import torch
import torch.nn as nn
from bleu_eval import BLEU
from dictionary import clean_caption, sentence_to_idx, word_to_idx, remove_other_tokens, max_length
from dataloader import MLDSVideoDataset
from torch.utils.data import DataLoader
from model_seq2seq import Seq2Seq
import pandas as pd

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

print(f'{device} is available')
torch.set_default_device(device)
torch.set_default_dtype(torch.float64)

dtype = torch.float

def train_loop(dataloader, model, loss_fn, optimize):
    model.train()
    num_batches = len(dataloader)
    train_loss, correct = 0,0
    size = len(dataloader.dataset)

    bleus = []
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        batch_size = y.size(0)
        sequence_length = y.size(1)


        optimize.zero_grad()
        y_pred = model(X, y)
        #print(f'{y_pred.shape = }')
        #print(f'{y.shape = }')
        #loss = loss_fn(y_pred, y)
        loss = loss_fn(y_pred.view(-1, y_pred.size(-1)), y.view(-1))
        train_loss += loss.item()

        batch_size = y_pred.size(0)
        seq_length = y_pred.size(1)
        #y_pred_view, y_view = y_pred.view(-1, y_pred.size(-1)), y.view(-1)
                        
        final_candidate, final_actual = [], []
        for candidate in range(batch_size):
            pred_sentence = []
            actual = []    
            for seq in range(seq_length):
                pred_sentence.append(idx_to_word[torch.argmax(y_pred[candidate][seq]).item()])
                actual.append(idx_to_word[y[candidate][seq].item()])

            final_candidate.append(remove_other_tokens(pred_sentence))
            final_actual.append(remove_other_tokens(actual))

        bleus = bleus + [(BLEU(c, t) if c.strip() else 0) for c, t in zip(final_candidate, final_actual)]
        
        loss.backward()
        optimize.step()
    
    train_loss /= num_batches
    
    return sum(bleus)/len(bleus), train_loss
        
idx_to_word = {v:k for k, v in word_to_idx.items()}
def eval_loop(dataloader, mode, loss_fn):
    model.eval()
    size= len(dataloader.dataset)

    with torch.no_grad():
        eval_loss = 0
        bleus = []
        for X,y in dataloader:
            X = X.to(device)
            y = y.to(device)
            y_pred = model(X)
            
            loss = loss_fn(y_pred.view(-1, y_pred.size(-1)), y.view(-1))
            eval_loss += loss.item()

            batch_size = y_pred.size(0)
            seq_length = y_pred.size(1)
            #y_pred_view, y_view = y_pred.view(-1, y_pred.size(-1)), y.view(-1)
            
            
            final_candidate, final_actual = [], []
            for candidate in range(batch_size):
                pred_sentence = []
                actual = []    
                for seq in range(seq_length):
                    pred_sentence.append(idx_to_word[torch.argmax(y_pred[candidate][seq]).item()])
                    actual.append(idx_to_word[y[candidate][seq].item()])

                final_candidate.append(remove_other_tokens(pred_sentence))
                final_actual.append(remove_other_tokens(actual))

            bleus = bleus + [(BLEU(c, t) if c.strip() else 0) for c, t in zip(final_candidate, final_actual)]

        return sum(bleus)/len(bleus), eval_loss/len(dataloader), final_candidate[0], final_actual[0]

if __name__=='__main__':
    #model = Seq2Seq(4096, 512, 512, 512)
    # input size 4096, hidden size 128
    model = Seq2Seq(4096, 128)
    model2 = Seq2Seq(4096, 256)
    model.to(device, dtype=torch.float64)
    model2.to(device, dtype=torch.float64)

    lr = .001
    optimizer1 = torch.optim.Adam(model.parameters(), lr)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr)
    
    epoch = 0
    loss = 0

    epochs = 100

    loss_fn = torch.nn.CrossEntropyLoss()
    
    batch_size=16
    SAMPLE = 1

    b_bleu_model_one = {'epic':0,'bleu':0, 'params':None}
    b_bleu_model_two = {'epic':0,'bleu':0, 'params':None}

    if torch.cuda.is_available():
        train_dataloader = DataLoader(MLDSVideoDataset('training_label.json', 'training_data'), shuffle=True, batch_size=batch_size, generator=torch.Generator(device))
        eval_dataloader = DataLoader(MLDSVideoDataset('testing_label.json', 'testing_data'), shuffle=True,  batch_size=batch_size, generator=torch.Generator(device))
    else:
        train_dataloader = DataLoader(MLDSVideoDataset('training_label.json', 'training_data'),  batch_size=batch_size)
        eval_dataloader = DataLoader(MLDSVideoDataset('testing_label.json', 'testing_data'),  batch_size=batch_size)
    
    epoch_list, model_size, loss_list, eval_loss, bleu_list, train_bleu = [],[], [], [], [], []

    start = time.time()
    with open('training_run.txt','a') as run:
            run.write(f'Started at {start}\n')

    for epoch in range(epoch, epochs):
        with open('training_run.txt','a') as run:
            bleu, loss = train_loop(train_dataloader, model, loss_fn, optimizer1)
            if epoch%SAMPLE==0:
                run.write(f'Epoch: {epoch}. Model: 128. Training Loss: {loss}. Training BLEU:{bleu}  Time: {time.time()}\n')
                print(f'Epoch: {epoch}. Model: 128. Training Loss: {loss}. Training BLEU:{bleu} Time: {time.time()}')
                model_size.append(128)
                loss_list.append(loss)
                epoch_list.append(epoch)
                train_bleu.append(bleu)
                
            bleu, loss, pred, actual = eval_loop(eval_dataloader, model, loss_fn)
            if epoch%SAMPLE ==0:
                run.write(f'Epoch: {epoch}. Model: 128. Test loss {loss}. Test BLEU: {bleu}\n')
                print(f'Epoch: {epoch}. Model: 128. Test loss {loss}. Test BLEU: {bleu}') 
                run.write(f'{epoch = }. {pred = }. {actual =}\n')
                print(f'{epoch = }. {pred = }. {actual =}')
                bleu_list.append(bleu)
                eval_loss.append(loss)

            if b_bleu_model_one['bleu'] < bleu:
                b_bleu_model_one['epoch'] = epoch
                b_bleu_model_one['bleu'] = bleu
                b_bleu_model_one['params'] = copy.deepcopy(model.state_dict())


            bleu, loss = train_loop(train_dataloader, model2, loss_fn, optimizer2)
            if epoch%SAMPLE==0:
                run.write(f'Epoch: {epoch}. Model: 256. Training Loss: {loss}. Training BLEU:{bleu}  Time: {time.time()}\n')
                print(f'Epoch: {epoch}. Model: 256. Training Loss: {loss}. Training BLEU:{bleu} Time: {time.time()}')
                model_size.append(256)
                loss_list.append(loss)
                epoch_list.append(epoch)
                train_bleu.append(bleu)
                
            bleu, loss, pred, actual = eval_loop(eval_dataloader, model2, loss_fn)
            if epoch%SAMPLE ==0:
                run.write(f'Epoch: {epoch}. Model: 256. Test loss {loss}. Test BLEU: {bleu}\n')
                print(f'Epoch: {epoch}. Model: 256. Test loss {loss}. Test BLEU: {bleu}') 
                run.write(f'{epoch = }. {pred = }. {actual =}\n')
                print(f'{epoch = }. {pred = }. {actual =}')
                bleu_list.append(bleu)
                eval_loss.append(loss)

            if b_bleu_model_two['bleu'] < bleu:
                b_bleu_model_two['epoch'] = epoch
                b_bleu_model_two['bleu'] = bleu
                b_bleu_model_two['params'] = copy.deepcopy(model2.state_dict())

    end = time.time()
    with open('training_run.txt','a') as run:
        run.write(f'Model@128: Best BLEU at epoch {b_bleu_model_one["epoch"]}. Bleu: {b_bleu_model_one["bleu"]}\n')
        run.write(f'Model@256: Best BLEU at epoch {b_bleu_model_two["epoch"]}. Bleu: {b_bleu_model_two["bleu"]}\n')
        run.write(f'Ended at {end}. Total runtime is {end-start}\n')

        torch.save(b_bleu_model_one['params'], 'model1.pth')
        torch.save(b_bleu_model_two['params'], 'model2.pth')

    df = pd.DataFrame({'epoch':epoch_list, 'model':model_size ,'eval_loss':eval_loss, 'eval_bleu':bleu_list,'train_loss':loss_list, 'train_bleu':train_bleu})
    df.to_csv('raw_data.txt')                
    
