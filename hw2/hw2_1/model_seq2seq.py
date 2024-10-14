import torch
import torch.nn as nn 
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights
from dictionary import word_to_idx, max_length, remove_other_tokens

""" class VideoEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.dropout= nn.Dropout(p=.25)
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)

    def forward(self, x):
        x = self.linear(x)
        x = self.dropout(x)
        output, (hidden, cell) = self.lstm(x)
        return output, (hidden, cell)
    
class TextDecoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size = input_size, hidden_size=hidden_size, batch_first=True)
        
    def forward(self, x, hidden, cell):
        output, (e_hidden, e_cell) = self.lstm(x, (hidden,cell))
        return output, (e_hidden, e_cell) """

class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden, word_to_idx = word_to_idx, sentence_len = max_length):
        super().__init__()
        self.word_to_idx = word_to_idx
        self.idx_to_word = {val:key for key, val in self.word_to_idx.items()}
        self.sentence_len = sentence_len
        self.lstm1 = nn.LSTM(hidden, hidden, batch_first=True)
        self.lstm2 = nn.LSTM(hidden, hidden, batch_first=True)
        self.embedding = nn.Embedding(len(self.word_to_idx), hidden)
        self.input_lin = nn.Linear(input_size, hidden)
        self.logits = nn.Linear(hidden, len(word_to_idx))
        self.fixy = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(p = 0.25)

    def forward(self, x, training_label = None):
        x = self.dropout(self.input_lin(x))
        encoder_output, (hidden, cell) = self.lstm1(x)
        batch_size = encoder_output.size(0)


        decoder_input = torch.ones(batch_size, 1, dtype=int) * self.word_to_idx['<BOS>']
        decoder_input = self.embedding(decoder_input)
                
        sentence = []
        sentence.append(decoder_input)
        d_hidden, d_cell = hidden, cell

        for incr in range(self.sentence_len-1):
            y, (d_hidden, d_cell) = self.forward_step(decoder_input, d_hidden, d_cell)
            sentence.append(y)

            if training_label is not None:
                decoder_input = self.embedding(training_label[:, incr+1].unsqueeze(1))
            else:
                #decoder_input = y
                _, topi = y.topk(1)
                decoder_input = self.embedding(topi.squeeze(-1).detach())

            
        return self.logits(torch.cat(sentence, dim=1))
    
    def forward_step(self, decoder_input, d_hidden, d_cell):
        decoder_input = F.relu(decoder_input)
        output, (hidden, cell) =  self.lstm2(decoder_input, (d_hidden, d_cell))
        output = self.fixy(output)
        return output, (hidden, cell)



if __name__ == '__main__':
    torch.set_default_device('cuda')
    x = torch.rand(5, 80, 4096)
    x.to('cuda')
    my_model=Seq2Seq(x.size(-1), 512)
    my_model.eval()
    y_pred = my_model(x)

    print(y_pred.shape)
    #my_model.embedding.weights.data - 
