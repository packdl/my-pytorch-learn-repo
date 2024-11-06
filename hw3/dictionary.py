from collections import Counter
import json
from functools import cache
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("albert/albert-base-v2")
#tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

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
                #context = context.replace('.','').replace('?','').replace('!','').replace(',','').replace(';','')
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
                #context = context.replace('.','').replace('?','').replace('!','').replace(',','').replace(';','')
                qas = paragraph['qas']
                for qa in paragraph['qas']:
                    question = qa['question']
                    answer = qa['answers'][0]
                    answer_start = answer['answer_start']
                    answer_text = answer['text']
                    answer_list, question_list, context_list, starts_ends = self.find_start_end(context, answer_text, question)
                    # answer_list, question_list, context_list, start_ends
                    #if len(answer_list) >=2:
                    #    print((len(answer_list), len(question_list), len(context_list), len(starts_ends)))
                    #    print(answer_list, question_list, context_list, starts_ends)
                    for a,q,c,se in zip(answer_list, question_list, context_list, starts_ends):
                        all_groups.append((c, question, answer_text, (se[0], se[1]), answer_start))
        #print(f'{len(all_groups) = }')
        return all_groups
    
    def find_start_end(self, context, answer_text, question):
       
        batch_encoding = tokenizer(question, context,max_length=512,stride=64, padding='max_length', truncation='only_second', return_overflowing_tokens=True)
        #batch_encoding = tokenizer(question, context,max_length=512, padding=True, truncation=True)
               
        loops = len(batch_encoding['overflow_to_sample_mapping'])
        
        starts_ends = []
        answer_list = []
        question_list = []
        context_list = []
        
        for x in range(loops):
            #context_words = tokenizer.convert_ids_to_tokens(batch_encoding['input_ids'])
            context_words = tokenizer.convert_ids_to_tokens(batch_encoding['input_ids'][x])
            answer_words = tokenizer.convert_ids_to_tokens(tokenizer(answer_text, add_special_tokens=False)['input_ids'])

            answer_list.append(tokenizer.convert_tokens_to_string(answer_words))
            #print(context_words)
            sep = context_words.index('[SEP]')
            if '<pad>' in context_words:
                #print('This happens')
                pad = context_words.index('<pad>')
                context_list.append(tokenizer.convert_tokens_to_string(context_words[sep+1:pad-1]))
            else:
                context_list.append(tokenizer.convert_tokens_to_string(context_words[sep+1:]))
            #print(context_list[-1])
            question_list.append(tokenizer.convert_tokens_to_string(context_words[1:sep]))
            #context_list.append(tokenizer.convert_tokens_to_string(context_words[sep+1:]))
            
            if not answer_words:
                start = end = 0
                starts_ends.append((0,0))
                continue
                # return 0,0
            
        
            len_answer = len(answer_words)

            end = start = 0
            if len_answer ==1: 
                try:
                    start = end = context_words.index(answer_words[0])
                    starts_ends.append((start, end))
                    continue
                except:
                    temp = [w for w in context_words if answer_words[0] in w]
                    if temp:
                        start = end = context_words.index(temp[0])
                    else:
                        start = end = 0
                        # return 0,0
                    starts_ends.append((start, end))
                    continue
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
                        if temp:
                            start = temp[0]
                        else:
                            start=end=0
                            break

                    if (start + len_answer) > len(context_words):
                        end = len(context_words)
                        break

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
                starts_ends.append((start,end))
        return answer_list, question_list, context_list, starts_ends

                




if __name__=='__main__':
    bert = BertDictionary('data/spoken_train-v1.1.json')
    c = bert.create_dictionary()
    print(c.total())
    d = bert.create_dictionary()
    print(d.total())
    print(c == d)
    
    print(bert.num_qapairs())

    ctx = 'In the beginning was the world and the dog is really awesome'
    answer = ''
    question ='who'

    batch_encoding = tokenizer(question, ctx,max_length=512, padding=True, truncation=True)
    context_words = tokenizer.convert_ids_to_tokens(batch_encoding['input_ids'])
    print(context_words)
    answer_words = tokenizer.convert_ids_to_tokens(tokenizer(answer, add_special_tokens=False)['input_ids'])
    print(answer_words)
    print(bert.find_start_end(ctx, answer, question))