import sys
import random
from pathlib import Path

import torch
import torch.nn as nn 
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights
from torch.utils.data import DataLoader

from dictionary import word_to_idx, max_length, remove_other_tokens
from dataloader import WeirdDataset

class Attention(nn.Module):
    """
    The Attention mechanism is based on the one found at https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html. 
    It is an implementation of the Bahdanau Attention mechanism. 
    """
    def __init__(self, hidden):
        super().__init__()
        self.hidden = hidden
        self.W = nn.Linear(hidden,hidden)
        self.V = nn.Linear(hidden, 1)
        self.U = nn.Linear(hidden, hidden)

    def forward(self, hidden, encoder_output):

        if hidden.size(0) != encoder_output.size(0):
            hidden = hidden.permute(1,0,2)
        scores = self.V(torch.tanh(self.W(hidden) + self.U(encoder_output)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, encoder_output)

        return context, weights


class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden, word_to_idx = word_to_idx, sentence_len = max_length):
        super().__init__()
        self.word_to_idx = word_to_idx
        self.idx_to_word = {val:key for key, val in self.word_to_idx.items()}
        self.sentence_len = sentence_len
        self.gru1 = nn.GRU(hidden, hidden, batch_first=True)
        self.gru2 = nn.GRU(2*hidden, hidden, batch_first=True)
        self.attention = Attention(hidden)
        self.embedding = nn.Embedding(len(self.word_to_idx), hidden)
        self.input_lin = nn.Linear(input_size, hidden)
        self.logits = nn.Linear(hidden, len(word_to_idx))
        self.fixy = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(p = 0.3)
        self.dropout2 = nn.Dropout(p = 0.2)
        self.hidden = hidden

    def forward(self, x, training_label = None):
        x = self.dropout(self.input_lin(x))
        encoder_output, hidden = self.gru1(x)
        batch_size = encoder_output.size(0)
        step = encoder_output.size(1)

        decoder_input = torch.ones(batch_size, 1, dtype=int) * self.word_to_idx['<BOS>']
        decoder_input = self.embedding(decoder_input)
        #print(f'{decoder_input.shape = }')
        
        #pad = torch.zeros(batch_size, step-1, decoder_input.size(2))
        
        #print(f'{pad.shape =}')

        #decoder_input = torch.cat([decoder_input, pad], 1)
        #print(f'{decoder_input.shape =}')
    
        sentence = []
        sentence.append(decoder_input)
        d_hidden = hidden

        for incr in range(self.sentence_len-1):
            y, d_hidden = self.forward_step(decoder_input, d_hidden, encoder_output, hidden)
            sentence.append(y)
            choice = random.choice([True, False])
            if choice and training_label is not None:
                decoder_input = self.embedding(training_label[:, incr+1].unsqueeze(1))
                #print(f'randomchoice{decoder_input.shape = }')
            else:
                #decoder_input = y
                _, topi = y.topk(k=1, dim=2)
                #_, topi2 = y.topk(k=1, dim=1)
                #print(f'{topi2.shape =}')
                decoder_input = self.embedding(topi.squeeze(2).detach())
                
                #print(f'topiversion{decoder_input.shape = }')


        final_sentence = []
        for line in sentence: # batchsize x 80 X 256
            final_sentence.append(self.logits(line.max(dim=1, keepdim=True)[0]))
        #print(f'{final_sentence[0].shape = }')
        #print(f'{final_sentence[0].dtype = }')
           
        return torch.cat(final_sentence, dim=1)
    
    def forward_step(self, decoder_input, d_hidden, encoder_output, e_hidden):
        # decoder_input = F.relu(decoder_input)
        decoder_input = self.dropout2(decoder_input)
        # query = hidden.permute(1,0,2)

        #e_hidden = e_hidden.permute(1,0,2)
        query = d_hidden.permute(1,0,2)
        encoder_output = encoder_output[:,0,:].unsqueeze(1)
        ctx, attn_weights = self.attention(query, encoder_output)
        
        #ctx = ctx.repeat(1,decoder_input.size(1), 1)
        #print(f'ffff{ctx.shape = }')
        decoder_input = torch.cat((decoder_input, ctx), dim=2)
        #print(f'ddfd{decoder_input.shape = }')
        output, hidden =  self.gru2(decoder_input, d_hidden)
        return output, hidden

#if __name__ == '__main__':
    """ torch.set_default_device('cuda')
    x = torch.rand(5, 80, 4096)
    x.to('cuda')
    my_model=Seq2Seq(x.size(-1), 256)
    my_model.eval()
    y_pred = my_model(x, torch.randint(1,25, (5,9)))

    
    print(y_pred.shape) """ 
    #my_model.embedding.weights.data - 


if __name__ == '__main__':
    idx_to_word = {v:k for k, v in word_to_idx.items()}
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    #print(f'{device} is available')
    torch.set_default_device(device)
    torch.set_default_dtype(torch.float64)
    
    if len(sys.argv) == 3:        
        _, test_dir, outputfile = sys.argv
        print("let's go create some captions")
        print("We assume the use of testing_label.json in the same directory as hw2_seq2seq.sh and TESTDIR ")

        with open(Path(test_dir)/'id.txt', 'r') as id_file:
            file_ids = id_file.readlines()
        file_ids =[vid_id.strip() for vid_id in file_ids]

        #eval_dataloader = DataLoader(MLDSVideoDataset('testing_label.json', test_dir), batch_size=1)
        model=Seq2Seq(4096, 128)
        model=model.to(device)

        if (Path('.')/'model1.pth').exists():
            print("Loading model parameters.")
            checkpoint = torch.load(Path('.')/'model1.pth', weights_only=True)
            model.load_state_dict(checkpoint)
            print("Model loaded")

        write_to_file = []
        loader = DataLoader(WeirdDataset('testing_label.json', 'testing_data'), batch_size=1)
        for X,target in loader:
            labels, data = X
            data = data.to(device)
            candidate = model(data)
            seq_length = candidate.size(1)
            for idx, label in enumerate(labels):
                result = candidate[idx]
                pred_sentence = []
                for word in range(seq_length):
                    result[word]
                    pred_sentence.append(idx_to_word[torch.argmax(result[word]).item()])
                pred_sentence = remove_other_tokens(pred_sentence)
                write_to_file.append(f'{label},{pred_sentence}')
        with open(outputfile, 'w') as out:
            out.write('\n'.join(write_to_file))
        print(f"Output file created at {outputfile}")     
    else:
        print('usage: hw2_seq2seq.sh TESTDIR OUTPUT_FILE')

