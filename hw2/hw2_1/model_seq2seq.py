import torch
import torch.nn as nn 
from torchvision.models import vgg16, VGG16_Weights
from dictionary import word_to_idx, max_length, remove_other_tokens

class VideoEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size)

    def forward(self, x):
        # x = self.CNN(x)
        output, (hidden, cell) = self.lstm(x)
        return output, (hidden, cell)
    
class TextDecoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size = input_size, hidden_size=hidden_size)
        
    def forward(self, x, hidden, cell):
        output, (e_hidden, e_cell) = self.lstm(x, (hidden,cell))
        return output, (e_hidden, e_cell)

class Seq2Seq(nn.Module):
    def __init__(self, e_input_size, e_hidden_size, d_input_size, d_hidden_size, word_to_idx = word_to_idx, sentence_len = max_length):
        super().__init__()
        self.word_to_idx = word_to_idx
        self.idx_to_word = {val:key for key, val in self.word_to_idx.items()}
        self.sentence_len = sentence_len
        self.encoder = VideoEncoder(e_input_size, e_hidden_size)
        self.decoder = TextDecoder(d_input_size, d_hidden_size)
        self.embedding = nn.Embedding(len(self.word_to_idx), d_input_size)
        self.linear = nn.Linear(d_hidden_size, len(word_to_idx))

    def forward(self, x):
        _, (hidden, cell) = self.encoder(x)
        word_embedding = self.embedding(torch.tensor(self.word_to_idx['<BOS>']))
        word_embedding = torch.reshape(word_embedding , (1, word_embedding.shape[0]))
        y, (d_hidden, d_cell) = self.decoder(word_embedding, hidden, cell)
        logits = self.linear(y)
        y = torch.argmax(logits).item()
        
        sentence = []
        sentence.append(logits)
        for _ in range(self.sentence_len -1):
            word = self.idx_to_word[y]
            word_embedding = self.embedding(torch.tensor(self.word_to_idx[word]))
            word_embedding = torch.reshape(word_embedding , (1, word_embedding.shape[0]))
            y, (d_hidden, d_cell) = self.decoder(word_embedding, d_hidden, d_cell)
            logits = self.linear(y)
            sentence.append(logits)
            y = torch.argmax(logits).item()

        """ combined_sentence = []
        for word in sentence:
            distance = torch.norm(self.embedding.weight.data - word, dim=1)
            idx = torch.argmin(distance).item()
            combined_sentence.append(self.idx_to_word[idx])
            return remove_other_tokens(sentence)
        """

        #print(torch.cat(sentence, dim=0).shape)
        return torch.cat(sentence, dim=0) 
    

if __name__ == '__main__':
    x = torch.rand(80, 4096)
    my_model = Seq2Seq(4096, 512, 512, 512)
    my_model.eval()
    print(my_model(x))

    