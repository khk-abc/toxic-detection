import json
import random
import time
import numpy as np

import torch
from ltp import LTP
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer



def repeat_sent(sent,length):
    raw_length = len(sent)
    raw_sent = sent
    while (length-len(sent))>int(0.5*raw_length):
        sent = "。".join([sent,raw_sent])

    return sent




class MyDataset(Dataset):
    '''
    The dataset based on Bert.
    '''

    def __init__(self, config, args, data_name, task_num=4,add_special_tokens=True, not_test=True):
        self.config = config
        self.args=args
        self.not_test = not_test
        self.data_name = data_name
        self.lexicon_base_path = config.lexicon_path
        self.max_tok_len = args.pad_size
        self.gpt_max_tok_len = args.gpt_pad_size
        self.add_special_tokens = add_special_tokens
        self.task_num =task_num

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)

        with open(data_name, 'r') as f:
            self.data_file = json.load(f)


        self.data = []
        self.preprocess_data()

        self.task_prompt = self.tokenizer(list(config.task_prompt.values()),padding='longest',truncation=False,return_tensors='pt')


    def modify_text(self,ori_text,method='naive',samples=None):
        if method=='naive':
            if ori_text[-1] in [',', '，', '。', '.']:
                ori_text = f'给定言论：“{ori_text}”，该言论是：'
            else:
                # ori_text = '。关于上述文本，可以对文本毒性、目标群体和表达方式作出以下判断：'
                ori_text = f'给定言论：“{ori_text}”，该言论是：'

        elif method=='in-context':
            try:
                assert samples is not None
            except:
                raise ValueError('在in-context方式下必须传入samples变量！')

            in_contexts = ""
            for item in samples:
                content = item['content']
                toxic_text_label = item['toxic_text_label']
                toxic_type_text_label = item['toxic_type_text_label']
                expression_text_label = item['expression_text_label']
                target_text_label = "-".join(item['target_text_label'])
                item_context = f"给定言论：“{content}”，该言论是：{toxic_text_label}、{toxic_type_text_label}、{expression_text_label}、{target_text_label}。"
                in_contexts = in_contexts + item_context
            ori_text = in_contexts + f"给定言论：“{ori_text}“，该言论是："

        return ori_text


    def in_context_sampling(self,k=2):
        sample_indexes = np.random.choice([idx for idx in range(len(self.data_file))],k).tolist()
        samples = [self.data_file[idx] for idx in sample_indexes]

        return samples


    def preprocess_data(self):
        print('Preprocessing Data {} ...'.format(self.data_name))
        data_time_start = time.time()

        count = 0
        for row in tqdm(self.data_file):
            ori_text = row['content']
            if self.args.repeatsent:
                if self.args.use_two_side:
                    ori_text = repeat_sent(ori_text, 80)
                else:
                    ori_text = repeat_sent(ori_text, 80)
            if row.get('response') is not None and self.args.concat_social_sense:
                print('使用gpt的social senses！')
                # ori_text = '给定言论：“' + ori_text + '”。 相关社会含义涉及: “' + row['response']+'”。' + "该言论是："
                # ori_text = '给定言论：“' + ori_text + '”， 相关社会含义涉及: “' + row['response']+'”。' + "根据上述信息，对该言论的毒性可以做出以下判断："
                ori_text = '给定言论：“' + ori_text + '”， 相关社会含义涉及: “' + row['response']+'”。' + "根据上述信息，该言论的毒性是："
            else:
                ori_text = '给定言论：“' + ori_text + "“，该言论的毒性是："

            # samples = self.in_context_sampling(k=1)
            # ori_text = self.modify_text(ori_text=ori_text,method='in-context',samples=samples)
            print(ori_text)

            # pos = row['pos']

            text = self.tokenizer(ori_text, add_special_tokens=self.add_special_tokens,
                                  max_length=int(self.max_tok_len), padding='max_length', truncation=True)
            # text = self.tokenizer(ori_text, add_special_tokens=self.add_special_tokens,padding='longest', truncation=False)



            if self.args.use_two_side:
                toxic_social_context = self.tokenizer(row['toxic_response'], add_special_tokens=self.add_special_tokens,
                                                max_length=int(self.gpt_max_tok_len), padding='max_length', truncation=True)
                non_toxic_social_context = self.tokenizer(row['non_toxic_response'], add_special_tokens=self.add_special_tokens,
                                                      max_length=int(self.gpt_max_tok_len), padding='max_length',
                                                      truncation=True)
            else:
                if self.args.sample_to_gpt:
                    row['response'] = "言论："+row['content']+"。分析："+row['response']
                    print(row['response'])
                social_context = self.tokenizer(row['response'], add_special_tokens=self.add_special_tokens,
                                                max_length=int(self.gpt_max_tok_len), padding='max_length',
                                                truncation=True)


            # for adding prompt tokens
            sep_index = text['input_ids'].index(self.tokenizer.convert_tokens_to_ids(self.tokenizer.special_tokens_map["sep_token"]))
            try:
                pad_index = text['input_ids'].index(self.tokenizer.convert_tokens_to_ids(self.tokenizer.special_tokens_map["pad_token"]))
                index = min(sep_index,pad_index)
            except ValueError:  # 该样本没有pad符
                index = sep_index

            prompt_index = list(range(index,index+self.task_num))
            for _ in range(self.task_num): # 一个task对应一个prompt token
                text['input_ids'].insert(index,self.tokenizer.convert_tokens_to_ids(self.tokenizer.special_tokens_map["mask_token"])) # 表示prompt token
                text['token_type_ids'].insert(index,0)
                text['attention_mask'].insert(index,1)
                # pos.insert(index,-2) # -2 用于表示[MASK]的id

            # For content of samples
            row['index'] = prompt_index
            row['input_ids'] = text['input_ids']
            row['token_type_ids'] = text['token_type_ids']
            row['attention_mask'] = text['attention_mask']



            if self.args.use_two_side:
                row['toxic_gpt_context_input_ids'] = toxic_social_context['input_ids']
                row['toxic_gpt_context_token_type_ids'] = toxic_social_context['token_type_ids']
                row['toxic_gpt_context_attention_mask'] = toxic_social_context['attention_mask']

                row['non_toxic_gpt_context_input_ids'] = non_toxic_social_context['input_ids']
                row['non_toxic_gpt_context_token_type_ids'] = non_toxic_social_context['token_type_ids']
                row['non_toxic_gpt_context_attention_mask'] = non_toxic_social_context['attention_mask']
            else:
                # for context of gpt
                row['gpt_context_input_ids'] = social_context['input_ids']
                row['gpt_context_token_type_ids'] = social_context['token_type_ids']
                row['gpt_context_attention_mask'] = social_context['attention_mask']

            # row['pos'] = pos

            if sum(row['toxic_ids'])>2:
                count +=1

            self.data.append(row)

        print('total num of containing dirty words: ',count)

        data_time_end = time.time()
        print("... finished preprocessing cost {} ".format(data_time_end - data_time_start))

    def __len__(self):
        return len(self.data_file)

    def __getitem__(self, idx, corpus=None):
        # row = self.data_file[idx]
        row = self.data[idx]

        if self.args.use_soft_alpha>0:
            if self.args.use_two_side:
                sample = {
                    # For content of samples
                    'index': row['index'],
                    'input_ids': row['input_ids'],
                    'token_type_ids': row['token_type_ids'],
                    'attention_mask': row['attention_mask'],
                    'sent_length': sum(row['attention_mask']),
                    'toxic_ids': row['toxic_ids'],
                    # 'pos':row['pos'],

                    # # for context of gpt
                    # 'gpt_context_input_ids': row['gpt_context_input_ids'],
                    # 'gpt_context_token_type_ids': row['gpt_context_token_type_ids'],
                    # 'gpt_context_attention_mask': row['gpt_context_attention_mask'],

                    # for context of gpt two side
                    'toxic_gpt_context_input_ids': row['toxic_gpt_context_input_ids'],
                    'toxic_gpt_context_token_type_ids': row['toxic_gpt_context_token_type_ids'],
                    'toxic_gpt_context_attention_mask': row['toxic_gpt_context_attention_mask'],

                    'non_toxic_gpt_context_input_ids': row['non_toxic_gpt_context_input_ids'],
                    'non_toxic_gpt_context_token_type_ids': row['non_toxic_gpt_context_token_type_ids'],
                    'non_toxic_gpt_context_attention_mask': row['non_toxic_gpt_context_attention_mask'],

                    # For label
                    'toxic': row["toxic_one_hot"],
                    'toxic_label': row["toxic"],
                    'toxic_type': row["toxic_type_one_hot"],
                    'toxic_type_label': row["toxic_type"],
                    'expression': row["expression_one_hot"],
                    'expression_label': row["expression"],
                    'target': row["target"],

                    # For soft label
                    'toxic_soft': row['toxic_soft_one_hot'],
                    'toxic_type_soft': row['toxic_type_soft_one_hot'],
                    'expression_soft': row['expression_soft_one_hot'],
                    'target_soft': row['target_soft_one_hot'],
                }

            else:
                sample = {
                # For content of samples
                'index':row['index'],
                'input_ids': row['input_ids'],
                'token_type_ids': row['token_type_ids'],
                'attention_mask': row['attention_mask'],
                'sent_length': sum(row['attention_mask']),
                'toxic_ids': row['toxic_ids'],
                # 'pos':row['pos'],

                # for context of gpt
                'gpt_context_input_ids': row['gpt_context_input_ids'],
                'gpt_context_token_type_ids': row['gpt_context_token_type_ids'],
                'gpt_context_attention_mask': row['gpt_context_attention_mask'],

                # For label
                'toxic': row["toxic_one_hot"],
                'toxic_label': row["toxic"],
                'toxic_type': row["toxic_type_one_hot"],
                'toxic_type_label': row["toxic_type"],
                'expression': row["expression_one_hot"],
                'expression_label': row["expression"],
                'target': row["target"],

                # For soft label
                'toxic_soft':row['toxic_soft_one_hot'],
                'toxic_type_soft':row['toxic_type_soft_one_hot'],
                'expression_soft':row['expression_soft_one_hot'],
                'target_soft':row['target_soft_one_hot'],
            }

        else:
            if self.args.use_two_side:
                sample = {
                    # For content of samples
                    'index': row['index'],
                    'input_ids': row['input_ids'],
                    'token_type_ids': row['token_type_ids'],
                    'attention_mask': row['attention_mask'],
                    'sent_length': sum(row['attention_mask']),
                    'toxic_ids': row['toxic_ids'],
                    # 'pos':row['pos'],

                    # # for context of gpt
                    # 'gpt_context_input_ids': row['gpt_context_input_ids'],
                    # 'gpt_context_token_type_ids': row['gpt_context_token_type_ids'],
                    # 'gpt_context_attention_mask': row['gpt_context_attention_mask'],

                    # for context of gpt two side
                    'toxic_gpt_context_input_ids': row['toxic_gpt_context_input_ids'],
                    'toxic_gpt_context_token_type_ids': row['toxic_gpt_context_token_type_ids'],
                    'toxic_gpt_context_attention_mask': row['toxic_gpt_context_attention_mask'],

                    'non_toxic_gpt_context_input_ids': row['non_toxic_gpt_context_input_ids'],
                    'non_toxic_gpt_context_token_type_ids': row['non_toxic_gpt_context_token_type_ids'],
                    'non_toxic_gpt_context_attention_mask': row['non_toxic_gpt_context_attention_mask'],

                    # For label
                    'toxic': row["toxic_one_hot"],
                    'toxic_label': row["toxic"],
                    'toxic_type': row["toxic_type_one_hot"],
                    'toxic_type_label': row["toxic_type"],
                    'expression': row["expression_one_hot"],
                    'expression_label': row["expression"],
                    'target': row["target"],
                }

            else:
                sample = {
                    # For content of samples
                    'index': row['index'],
                    'input_ids': row['input_ids'],
                    'token_type_ids': row['token_type_ids'],
                    'attention_mask': row['attention_mask'],
                    'sent_length': sum(row['attention_mask']),
                    'toxic_ids': row['toxic_ids'],
                    # 'pos':row['pos'],

                    # for context of gpt
                    'gpt_context_input_ids': row['gpt_context_input_ids'],
                    'gpt_context_token_type_ids': row['gpt_context_token_type_ids'],
                    'gpt_context_attention_mask': row['gpt_context_attention_mask'],

                    # For label
                    'toxic': row["toxic_one_hot"],
                    'toxic_label': row["toxic"],
                    'toxic_type': row["toxic_type_one_hot"],
                    'toxic_type_label': row["toxic_type"],
                    'expression': row["expression_one_hot"],
                    'expression_label': row["expression"],
                    'target': row["target"],
                }



        return sample


class MyDataloader(DataLoader):
    '''
    A batch sampler of a dataset. 
    '''

    def __init__(self, data, batch_size, shuffle=True, SEED=42):
        super().__init__(data, batch_size=batch_size, shuffle=shuffle)
        self.data = data
        self.shuffle = shuffle
        self.SEED = SEED
        random.seed(self.SEED)
        self.task_prompt=data.task_prompt

        self.indices = list(range(len(data)))
        if shuffle:
            random.shuffle(self.indices)
        self.batch_num = 0

    def __len__(self):
        return int(len(self.data) / float(self.batch_size))

    def num_batches(self):
        return len(self.data) / float(self.batch_size)

    def __iter__(self):
        self.indices = list(range(len(self.data)))
        if self.shuffle:
            random.shuffle(self.indices)
        return self

    def __next__(self):
        if self.indices != []:
            idxs = self.indices[:self.batch_size]
            batch = [self.data.__getitem__(i) for i in idxs]
            self.indices = self.indices[self.batch_size:]
            return batch
        else:
            raise StopIteration

    def get(self):
        self.reset()
        return self.__next__()

    def reset(self):
        self.indices = list(range(len(self.data)))
        if self.shuffle:
            random.shuffle(self.indices)


def to_tensor(batch):
    '''
    Convert a batch data into tensor
    '''
    args = {}
    for key in batch[0]:
        args[key] = torch.tensor([b[key] for b in batch])

    return args


