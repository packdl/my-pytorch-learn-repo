from collections import Counter
import json
from functools import cache

class BertDictionary():
    def __init__(self, content_file):
        with open(content_file,'r') as FILE_:
            self.documents = json.load(FILE_)
            self.counter = Counter()

    @cache
    def create_dictionary(self):
        # Iterate through content and return a full Counter object
        doc_list = self.documents['data']
        for doc in doc_list:
            for paragraph in doc['paragraphs']:
                context=paragraph['context']
                context = context.replace('.','').replace('?','').replace('!','').replace(',','').replace(';','')
                self.counter.update(context.split())

        return self.counter


    @cache
    def num_qapairs(self):
        totals = 0
        doc_list = self.documents['data']
        for doc in doc_list:
            for paragraph in doc['paragraphs']:
                qas = paragraph['qas']
                totals += len(qas)
        return totals
    
    @cache
    def get_data_groups(self):
        doc_list = self.documents['data']
        
        all_groups = []
        for doc in doc_list:
            for paragraph in doc['paragraphs']:
                context=paragraph['context']
                context = context.replace('.','').replace('?','').replace('!','').replace(',','').replace(';','')
                qas = paragraph['qas']
                for qa in paragraph['qas']:
                    question = qa['question']
                    answer = qa['answers'][0]
                    answer_start = answer['answer_start']
                    answer_text = answer['text']
                    start, end = self.find_start_end(context, answer_text)
                            
                    all_groups.append((context, question, answer_text, (start, end), answer_start))
        return all_groups
    
    def find_start_end(self, context, answer_text):
        context_words = context.split()
        answer_words = answer_text.split()
        if not answer_words:
            return 0,0
        len_answer = len(answer_words)

        end = start = 0
        if len_answer ==1: 
            try:
                start = end = context_words.index(answer_words[0])
            except:
                temp = [w for w in context_words if answer_words[0] in w]
                start = end = context_words.index(temp[0])
        else:
            while (((end - start) + 1) != len_answer):
                #print(f'{context_words = }')
                #print(f'{answer_words[0] = }')
                try:
                    start_one = context_words.index(answer_words[0], start)
                    #print(f'{start_one = }')
                    temp = [start + i  for i, w in enumerate(context_words[start:]) if w.endswith(answer_words[0])]
                    if temp:
                        #print('temp is true')
                        start_two = temp[0]
                    else:
                        #print('temp is not true')
                        start_two = start_one
                    
                    #print(f'{start_two = }')
                    start = start_one if start_one < start_two else start_two
                except:
                    temp = [start + i  for i, w in enumerate(context_words[start:]) if w.endswith(answer_words[0])]
                    start = temp[0]
                #print(f'{answer_words = }')
                #print(f'{start = }')
                #print(f'{context_words[start:start+(len_answer)] = }')
                if context_words[start: start + (len_answer)] == answer_words:
                    end = start + (len_answer-1)
                    #print(f'1{end = }')
                    break
                elif answer_words[-1] in context_words[start + (len_answer-1)]:
                    end = start+ (len_answer-1)
                    #print(f'2{end = }')
                    break
                else:
                    start+=1
                    #print(f'2{start = }')
                    continue
        return start, end

                




if __name__=='__main__':
    bert = BertDictionary('data/spoken_train-v1.1.json')
    c = bert.create_dictionary()
    print(c.total())
    d = bert.create_dictionary()
    print(d.total())
    print(c == d)
    
    print(bert.num_qapairs())

    ctx = 'In the beginning was the world and the dog is really awesome'
    answer = 'the dog is'

    print(bert.find_start_end(ctx, answer))