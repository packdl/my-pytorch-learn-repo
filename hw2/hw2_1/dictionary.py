from collections import Counter
import json
from functools import cache

@cache
def create_dictionary(label_path):
    dict_cnt = Counter()
    with open(label_path,'r') as FILE_:
        labels = json.load(FILE_)
        for label in labels:
            for caption in label['caption']:
                caption = caption.replace('.','').replace('?','').replace('!','').replace(',','').replace("'",'').replace('-','').replace('"','')
                dict_cnt.update(caption.split())
    return dict_cnt

other_tokens = {'<PAD>':0,'<BOS>':1, '<EOS>':2, '<UNK>':3 }

word_dictionary = create_dictionary('training_label.json')
# print(word_dictionary)
word_dictionary = {key:val for key, val in word_dictionary.items() if val > 3}
word_to_idx = {key:(idx + len(other_tokens)) for idx, key in enumerate(word_dictionary)}
word_to_idx.update(other_tokens)


@cache
def get_max_length(label_path):
    with open(label_path,'r') as FILE_:
        labels = json.load(FILE_)
        max_length = 0
        for label in labels:
            for caption in label['caption']:
                caption = caption.replace('.','').replace('?','').replace('!','').replace(',','').replace("'",'').replace('-','').replace('"','')
                cap_len = len(caption.split())
                if cap_len > max_length:
                    max_length = cap_len
    return max_length + 2

max_length=get_max_length('training_label.json')



def sentence_to_idx(caption, max_length = max_length, word_to_idx=word_to_idx, other_tokens=other_tokens):
    caption = caption.replace('.','').replace('?','').replace('!','').replace(',','').replace("'",'').replace('-','').replace('"','')
    words = ['<BOS>'] + caption.split()
    if len(words) < max_length:
        words = words + ((max_length-1)-(len(words))) * ['<PAD>'] + ['<EOS>']
    words = [word_to_idx.get(word,word_to_idx['<UNK>']) for word in words]
    return words

def clean_caption(caption):
    caption = caption.replace('.','').replace('?','').replace('!','').replace(',','').replace("'",'').replace('-','').replace('"','')
    return caption

def remove_other_tokens(sentence:list):
    if sentence:
        return ' '.join([token for token in sentence if token not in ['<PAD>','<BOS>', '<EOS>']])
    return ''



if __name__ == '__main__':
    print(word_dictionary.most_common(10))
    print(sentence_to_idx('The man ran'))

    # vgg11