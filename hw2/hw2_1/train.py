import time
from pathlib import Path

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
    model=Seq2Seq(4096, 128)
    model.to(device, dtype=torch.float64)
    lr = .001
    optimizer1 = torch.optim.Adam(model.parameters(), lr)
    
    if (Path('.')/'checkpoint.tar').exists():
        checkpoint = torch.load(Path('.')/'checkpoint.tar', weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer1.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
    else:
        epoch = 0
        loss = 0
    
    epochs = 200

    if epoch / epochs > .95:
        epochs *=2

    loss_fn = torch.nn.CrossEntropyLoss()
    
    batch_size=32
    
    
    train_dataloader = DataLoader(MLDSVideoDataset('training_label.json', 'training_data'), batch_size=batch_size)
    eval_dataloader = DataLoader(MLDSVideoDataset('testing_label.json', 'testing_data'), batch_size=batch_size)
    
    epoch_list, loss_list, eval_loss, bleu_list, train_bleu = [],[], [], [], []

    start = time.time()
    with open('training_run.txt','a') as run:
            run.write(f'Started at {start}\n')

    for epoch in range(epoch+1, epochs):
        with open('training_run.txt','a') as run:
            bleu, loss = train_loop(train_dataloader, model, loss_fn, optimizer1)
            
            if epoch%10==0:
                run.write(f'Epoch: {epoch}. Training Loss: {loss}. Training BLEU:{bleu}  Time: {time.time()}\n')
                print(f'Epoch: {epoch}. Training Loss: {loss}. Training BLEU:{bleu} Time: {time.time()}')
                loss_list.append(loss)
                epoch_list.append(epoch)
                train_bleu.append(bleu)
                
            bleu, loss, pred, actual = eval_loop(eval_dataloader, model, loss_fn)
            if epoch%10 ==0:
                run.write(f'Epoch: {epoch}. Test loss {loss}. Test BLEU: {bleu}\n')
                print(f'Epoch: {epoch}. Test loss {loss}. Test BLEU: {bleu}') 
                run.write(f'{epoch = }. {pred = }. {actual =}\n')
                print(f'{epoch = }. {pred = }. {actual =}')
                bleu_list.append(bleu)
                eval_loss.append(loss)


            if epoch !=0 and epoch % 50 ==0:
                torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer1.state_dict(),
                'loss': loss,
                }, './checkpoint.tar')

     
    end = time.time()
    with open('training_run.txt','a') as run:
        run.write(f'Ended at {end}. Total runtime is {end-start}\n')
                
            #bleu = eval_loop(eval_dataloader, model)
            #un.write(f'Epoch: {epoch}. BLEU: {bleu}\n')
            #print(f'Epoch: {epoch}. BLEU: {bleu}')

    torch.save(model.state_dict(), 's2vt.pth')
    torch.save({'epoch': epoch,'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer1.state_dict(),'loss': loss,}, './checkpoint.tar')

    if Path('raw_data.txt').exists():
        olddata = pd.read_csv('raw_data.txt')
        columns = ['epoch', 'eval_loss', 'eval_bleu','train_loss', 'train_bleu']
        df = pd.DataFrame({'epoch':epoch_list, 'eval_loss':eval_loss, 'eval_bleu':bleu_list,'train_loss':loss_list, 'train_bleu':train_bleu})
        if not df.empty:
            df = pd.concat([olddata[columns], df[columns]], ignore_index=True)
        df.to_csv('raw_data.txt')
    else:
        df = pd.DataFrame({'epoch':epoch_list, 'eval_loss':eval_loss, 'eval_bleu':bleu_list,'train_loss':loss_list, 'train_bleu':train_bleu})
        df.to_csv('raw_data.txt')