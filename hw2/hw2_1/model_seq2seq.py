import random

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
        self.gru1 = nn.GRU(hidden, hidden, batch_first=True)
        self.gru2 = nn.GRU(2*hidden, hidden, batch_first=True)
        self.attention = nn.MultiheadAttention(hidden, 2)
        self.embedding = nn.Embedding(len(self.word_to_idx), hidden)
        self.input_lin = nn.Linear(input_size, hidden)
        self.logits = nn.Linear(hidden, len(word_to_idx))
        self.fixy = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(p = 0.25)
        self.hidden = hidden

    def forward(self, x, training_label = None):
        x = self.dropout(self.input_lin(x))
        encoder_output, hidden = self.gru1(x)
        batch_size = encoder_output.size(0)
        step = encoder_output.size(1)

        decoder_input = torch.ones(batch_size, 1, dtype=int) * self.word_to_idx['<BOS>']
        decoder_input = self.embedding(decoder_input)
    
        sentence = []
        sentence.append(decoder_input)
        d_hidden = hidden

        for incr in range(self.sentence_len-1):
            y, d_hidden = self.forward_step(decoder_input, d_hidden, encoder_output, hidden)
            sentence.append(y)

            if random.choice([True, False]) and training_label is not None:
                decoder_input = self.embedding(training_label[:, incr+1].unsqueeze(1))
            else:
                #decoder_input = y
                _, topi = y.topk(1)
                decoder_input = self.embedding(topi.squeeze(1).detach())

        return self.logits(torch.cat(sentence, dim=1))
    
    def forward_step(self, decoder_input, d_hidden, encoder_output, e_hidden):
        # decoder_input = F.relu(decoder_input)
        decoder_input = self.dropout(decoder_input)
        # query = hidden.permute(1,0,2)

        e_hidden = e_hidden.permute(1,0,2)

        ctx, attn_weights = self.attention(e_hidden, e_hidden, e_hidden)
        #print(f'ddfd{decoder_input.shape = }')
        #print(f'ffff{ctx.shape = }')
        decoder_input = torch.cat((decoder_input, ctx), dim=2)
        
        output, hidden =  self.gru2(decoder_input, d_hidden)
        output = self.fixy(output)
        return output, hidden

if __name__ == '__main__':
    torch.set_default_device('cuda')
    x = torch.rand(5, 80, 4096)
    x.to('cuda')
    my_model=Seq2Seq(x.size(-1), 256)
    my_model.eval()
    y_pred = my_model(x)

    print(y_pred.shape)
    #my_model.embedding.weights.data - 
