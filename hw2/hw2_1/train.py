import torch
import bleu_eval
from dictionary import clean_caption, sentence_to_idx
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
device='cpu'
dtype = torch.float

def train_loop(dataloader, model, loss_fn, optimize):
    model.train()
    num_batches = len(dataloader)
    train_loss, correct = 0,0
    size = len(dataloader.dataset)

    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        optimize.zero_grad()
        y_pred = model(X)
        y = clean_caption(y)
        y = sentence_to_idx(y)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        correct += (y_pred.argmax(dim=1) == y).type(torch.float).sum().item()

        loss.backward()
        optimize.step()
    
    train_loss /= num_batches
    correct /=size

    return train_loss, correct
        
def eval_loop(dataloader, model, loss_fn, epoch):
    model.eval()
    size= len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0,0

    with torch.no_grad():
        for X,y in dataloader:
            X = X.to(device)
            y = clean_caption(y)
            y = sentence_to_idx(y)
            y_pred = model(X)
            test_loss += loss_fn(y_pred, y).item()
            correct += (y_pred.argmax(dim=1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /=size
    return test_loss, correct

if __name__=='__main__':
    model = Seq2Seq(4096, 512, 512, 512)
    
    epochs = 5
    loss_fn = torch.nn.CrossEntropyLoss

    lr = .001
    batch_size=1
    optimizer1 = torch.optim.Adam(model.parameters(), lr)
    train_dataloader = DataLoader(MLDSVideoDataset('training_label.json', 'training_data'))
    
    for epoch in range(epochs):
        loss, correct = train_loop(train_dataloader, model, loss_fn, optimizer1)
        print(f'Epoch: {epoch}. Loss: {loss}. Acurracy: {correct}')
