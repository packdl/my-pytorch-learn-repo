import torch
import torch.nn as nn 
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

    def forward(self, x):
        print(x.dtype)
        x = self.dropout(self.input_lin(x))
        encoder_output, (hidden, cell) = self.lstm1(x)
        batch_size = encoder_output.size(0)

        word_embedding = self.embedding(torch.tensor(self.word_to_idx['<BOS>']))
        word_embedding = self.dropout(word_embedding)
        #print(f'dropout-embedding {word_embedding}')
        decoder_input = torch.empty(batch_size, self.sentence_len, word_embedding.size(-1))
        for i, j in zip(range(batch_size), range(self.sentence_len)):
            decoder_input[i,j] = word_embedding
        
        #print(f'decoder input: {decoder_input}')
        sentence = []
        sentence.append(torch.argmax(decoder_input, 2))
        d_hidden, d_cell = hidden, cell
        # decoder_input = self.embedding(torch.ones(batch_size, dtype=torch.long) * self.word_to_idx['<BOS>'])
        for incr in range(self.sentence_len-1):
            y, (d_hidden, d_cell) = self.lstm2(decoder_input, (d_hidden, d_cell))
            
            y = self.fixy(y)
            
            sentence.append(torch.argmax(y,2))

            # decoder_input = torch.cat((self.embedding(torch.argmax(y)), decoder_input[:, self.sentence_len+incr-1, :]), 1)
            decoder_input = y
            
            #print(self.embedding(torch.argmax(y,2)).shape)
            #print(self.embedding(torch.argmax(y,2)))
            """
                TO FIX: Somehow my y variable is being turned into an int . Add a linear expression here in the hopes of switching it 
            """
            

        """         #print(word_embedding.shape)
        word_embedding = torch.reshape(word_embedding , (1, word_embedding.shape[0]))
        
        y, (d_hidden, d_cell) = self.lstm2(word_embedding, hidden, cell)
        #y, (d_hidden, d_cell) = self.decoder(word_embedding, hidden, cell)
        output = self.logits(y)
        y = torch.argmax(output).item()
        sentence.append(output)
        for _ in range(self.sentence_len -2):
            word = self.idx_to_word[y]
            word_embedding = self.embedding(torch.tensor(self.word_to_idx[word]))
            #word_embedding = torch.reshape(word_embedding , (1, word_embedding.shape[0]))
            y, (d_hidden, d_cell) = self.lstm2(word_embedding, d_hidden, d_cell)
            output = self.logits(y)
            sentence.append(output)
            y = torch.argmax(output).item() """

        """ combined_sentence = []
        for word in sentence:
            distance = torch.norm(self.embedding.weight.data - word, dim=1)
            idx = torch.argmin(distance).item()
            combined_sentence.append(self.idx_to_word[idx])
            return remove_other_tokens(sentence)
        """

        #print(torch.cat(sentence, dim=0).shape)
        #print("Let's start printing x")
        for x in sentence:
            print(x.shape)
            print(x)

        #print(f'length of sentence: {len(sentence)}')
        #print(f'shape of sentence[0]: {sentence[0].shape}')

        return torch.cat(sentence, dim=0) 
    

if __name__ == '__main__':
    """     x = torch.rand(80, 4096)
        my_model = Seq2Seq(4096, 512, 512, 512)
        my_model.eval()
        print(my_model(x))
    """
    torch.set_default_device('cuda')
    x = torch.rand(5, 80, 4096)
    x.to('cuda')
    my_model=Seq2Seq(x.size(-1), 512)
    my_model.eval()
    y_pred = my_model(x)

    # y_pred has a shape of 45, 9, 2483
    print(y_pred.shape)

    print(y_pred[0].shape)

    