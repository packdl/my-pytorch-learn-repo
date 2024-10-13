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
        X = X.squeeze()
        y = y.squeeze(dim=0)
        #print(f'X.shape is {X.shape}')
        #print(f'dataloader length is {len(dataloader)}')
        optimize.zero_grad()
        y_pred = model(X)

        print(y.dtype)
        print(y_pred.dtype)
        print(y_pred)

        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        correct += (y_pred.argmax(dim=1) == y).type(torch.float).sum().item()

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
            X = X.squeeze()
            y = y.squeeze(dim=0)
            y.to(device, dtype=torch.float)

            y_pred = model(X)

            y_pred.to(dtype=torch.float)

            m = nn.LogSoftmax(dim=0)
            pred_sentence = []
            actual = []
            for y_pred_row, y_row in zip(y_pred, y):
                y_pred_idx = torch.argmax(m(y_pred_row)).item()
                pred_sentence.append(idx_to_word[y_pred_idx])
                actual.append(idx_to_word[y_row.item()])

            pred_sentence = remove_other_tokens(pred_sentence)
            actual = remove_other_tokens(actual)

            bleus.append(BLEU(pred_sentence, actual))

        return sum(bleus)/size

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
    
    epochs = 25

    if epoch / epochs > .95:
        epochs *=2

    loss_fn = torch.nn.CrossEntropyLoss()
    
    batch_size=10
    
    train_dataloader = DataLoader(MLDSVideoDataset('training_label.json', 'training_data'), batch_size=batch_size)
    #eval_dataloader = DataLoader(MLDSVideoDataset('testing_label.json', 'testing_data'))
    
    start = time.time()
    for epoch in range(epochs):
        with open('training_run.txt','a') as run:
            run.write(f'Started at {start}\n')
            loss, correct = train_loop(train_dataloader, model, loss_fn, optimizer1)
            run.write(f'Epoch: {epoch}. Loss: {loss}. Acurracy: {correct}. Time: {time.time()}\n')
            print(f'Epoch: {epoch}. Loss: {loss}. Acurracy: {correct}. Time: {time.time()}')

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

    torch.save(model, 's2vt.pth')
