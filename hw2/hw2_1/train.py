import time
from pathlib import Path

import torch
import torch.nn as nn
from bleu_eval import BLEU
from dictionary import clean_caption, sentence_to_idx, word_to_idx, remove_other_tokens, max_length
from dataloader import MLDSVideoDataset
from torch.utils.data import DataLoader
from model_seq2seq import Seq2Seq



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
        correct += (y_pred.argmax(dim=2) == y).type(torch.float).sum().item()

        loss.backward()
        optimize.step()
    
    train_loss /= num_batches
    correct /=size

    return train_loss, correct
        
idx_to_word = {v:k for k, v in word_to_idx.items()}
def eval_loop(dataloader, model):
    model.eval()
    size= len(dataloader.dataset)

    with torch.no_grad():
        
        bleus = []
        for X,y in dataloader:
            X = X.to(device)
            y = y.to(device)
            y_pred = model(X)
            
            batch_size = y_pred.size(0)
            seq_length = y_pred.size(1)
            y_pred_view, y_view = y_pred.view(-1, y_pred.size(-1)), y.view(-1)
            
            pred_sentence = []
            actual = []
            for seq in range(seq_length):
                pred_sentence.append(idx_to_word[torch.argmax(y_pred_view[seq]).item()])
                actual.append(idx_to_word[y_view[seq].item()])

            pred_sentence = remove_other_tokens(pred_sentence)
            actual = remove_other_tokens(actual)

            bleus.append(BLEU(pred_sentence, actual) if pred_sentence else 0)

        return sum(bleus)/size, pred_sentence, actual

if __name__=='__main__':
    #model = Seq2Seq(4096, 512, 512, 512)
    model=Seq2Seq(4096, 512)
    model.to(device, dtype=torch.float64)
    lr = .001
    optimizer1 = torch.optim.Adam(model.parameters(), lr)
    
    if (Path('.')/'checkpoint.tar').exists():
        checkpoint = torch.load(Path('.'/'checkpoint.tar'), weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer1.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
    else:
        epoch = 0
        loss = 0
    
    epochs = 75

    if epoch / epochs > .95:
        epochs *=2

    loss_fn = torch.nn.CrossEntropyLoss()
    
    batch_size=10
    
    
    train_dataloader = DataLoader(MLDSVideoDataset('training_label.json', 'training_data'), batch_size=batch_size)
    eval_dataloader = DataLoader(MLDSVideoDataset('testing_label.json', 'testing_data'), batch_size=batch_size)
    
    start = time.time()
    for epoch in range(epoch, epochs):
        with open('training_run.txt','a') as run:
            run.write(f'Started at {start}\n')
            loss, correct = train_loop(train_dataloader, model, loss_fn, optimizer1)
            
            if epoch%10==0:
                run.write(f'Epoch: {epoch}. Loss: {loss}. Acurracy: {correct}. Time: {time.time()}\n')
                print(f'Epoch: {epoch}. Loss: {loss}. Acurracy: {correct}. Time: {time.time()}')

            bleu, pred, actual = eval_loop(eval_dataloader, model)
            run.write(f'Epoch: {epoch}. BLEU: {bleu}\n')
            print(f'Epoch: {epoch}. BLEU: {bleu}') 

            if epoch !=0 and epoch % 50 ==0:
                torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer1.state_dict(),
                'loss': loss,
                }, './checkpoint.tar')

                run.write(f'{epoch = }. {pred = }. {actual =}')
                print(f'{epoch = }. {pred = }. {actual =}')
     
    end = time.time()
    with open('training_run.txt','a') as run:
        run.write(f'Ended at {end}. Total runtime is {end-start}\n')
                
            #bleu = eval_loop(eval_dataloader, model)
            #un.write(f'Epoch: {epoch}. BLEU: {bleu}\n')
            #print(f'Epoch: {epoch}. BLEU: {bleu}')

    torch.save(model, 's2vt.pth')
