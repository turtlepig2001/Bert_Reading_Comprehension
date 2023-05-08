'''
Date: 2023-05-01 23:25:51
LastEditors: turtlepig
LastEditTime: 2023-05-08 13:38:09
'''
'''
Paddle 与 pytorch的API映射表：https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/model_convert/pytorch_api_mapping_cn.html
涉及到的库可以参照上述链接进行更改
'''
import os
import jieba
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
import torch.utils.data  
# from datasets import Dataset,Features,Value
from torch.utils.data import RandomSampler,DataLoader,BatchSampler,SequentialSampler
# from datasets import load_dataset

import matplotlib.pyplot as plt
import transformers
from transformers import BertTokenizerFast,BertForQuestionAnswering,get_linear_schedule_with_warmup

from utils import prepare_train_features, prepare_validation_features
from squad import compute_prediction,squad_evaluate
from functools import partial

import collections
import time
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(filepath):

    '''
    json 数据的处理
    '''

    train_path = os.path.join(filepath, 'train.json')
    dev_path = os.path.join(filepath, 'dev.json')

    train_file = open(train_path, 'r',encoding='utf-8')
    tr_content = train_file.read()
    tr_ds = json.loads(tr_content)
    tr_ds = tr_ds['data'] # a list

    dev_path = open(dev_path, 'r',encoding='utf-8')
    dev_content = dev_path.read()
    dev_ds = json.loads(dev_content)
    dev_ds = dev_ds['data'] # a list
    

    return tr_ds,dev_ds

def extract_data(data):
    #源数据结构复杂，重新构建一个结构简单的字典列表，方便后续处理 train和dev数据结构基本一致，可以直接调用相同函数
    results = []
    for article in data:
        paragraphs = article['paragraphs']
        for paragraph in paragraphs:
            context = paragraph['context']
            qas = paragraph['qas']
            for q in qas:
                question = q['question']
                id = q['id']
                answers = q['answers']
                for answer in answers:
                    text = answer['text']
                    answer_start = answer['answer_start']
                    results.append({'question':question, 'id':id,'context':context,'answers':text,'answer_starts':answer_start})

    #     features = Features({
    #     'id': Value('string'),
    #     'question': Value('string'),
    #     'context': Value('string'),
    #     'answers': Value('string'),
    #     'answer_starts': Value('int32'),
    # })

    
    # results = Dataset.from_list(results, features)
                    
    return results


#调用BertTokenizer进行数据处理
#tokenizer的作用是将原始输入文本转化成模型可以接受的输入数据形式。
def getTokenizer(model_name):
    tokenizer = BertTokenizerFast.from_pretrained(model_name)# 使用'bert-base-chinese'
    #这里使用BertTokennizerFast的原因是为了使用return_offsets_mapping参数 使用BertTokennizer会报错
    return tokenizer

#源代码的简化版
def map(data,trans_func):
    data = trans_func(data)
    return data

#在原案例中使用load_dataset()API默认读取到的数据集是MapDataset对象，MapDataset是paddle.io.Dataset的功能增强版本。其内置的map()方法适合用来进行批量数据集处理。map()方法传入的是一个用于数据处理的function。
#由于在本案例中没有构造MapDataset类，因此参照源代码自行书写map函数
def trans_features(data,tokenizer,tr_or_dev):
    '''
    数据集中的example将会被转换成了模型可以接收的feature，包括input_ids、token_type_ids、答案的起始位置等信息。 
    其中：
    input_ids: 表示输入文本的token ID。
    token_type_ids: 表示对应的token属于输入的问题还是答案。（Transformer类预训练模型支持单句以及句对输入）。
    overflow_to_sample: feature对应的example的编号。
    offset_mapping: 每个token的起始字符和结束字符在原文中对应的index（用于生成答案文本）。
    start_positions: 答案在这个feature中的开始位置。
    end_positions: 答案在这个feature中的结束位置。
    '''
    max_seq_length = 512
    doc_stride = 128

    train_trans_func = partial(prepare_train_features,
                           max_seq_length=max_seq_length, 
                           doc_stride=doc_stride,
                           tokenizer=tokenizer)
    
    dev_trans_func = partial(prepare_validation_features,
                           max_seq_length=max_seq_length, 
                           doc_stride=doc_stride,
                           tokenizer=tokenizer)
    
    if tr_or_dev == 'train':
        data = map(data,train_trans_func)
    elif tr_or_dev == 'dev':
        data = map(data,dev_trans_func)

    return data


def get_tr_data_loader(tr_ds,berttokenizer):
    '''
    构造训练数据集的dataloader
    '''
    batch_size = 8
    train_sampler = RandomSampler(tr_ds)

    train_batchify_fn = lambda batch:{
    'input_ids' : pad_sequence([torch.tensor(example['input_ids'][0]) for example in batch],batch_first= True,padding_value = berttokenizer.pad_token_id),

    'token_type_ids': pad_sequence([torch.tensor(example['token_type_ids'][0]) for example in batch],batch_first= True,padding_value = berttokenizer.pad_token_type_id),

    'start_positions' : torch.tensor([example['start_positions'] for example in batch],dtype=torch.int64),

    'end_positions' : torch.tensor([example['end_positions'] for example in batch], dtype=torch.int64)
    }

    train_data_loader = DataLoader(tr_ds,sampler=train_sampler,batch_size=batch_size,collate_fn=train_batchify_fn,pin_memory=True)

    return train_data_loader

def get_dev_data_loader(dev_ds,berttokenizer):

    batch_size = 8
    dev_sampler = BatchSampler(SequentialSampler(dev_ds),batch_size=batch_size , drop_last=False)

    dev_batchify_fn = lambda batch:{
    'input_ids': pad_sequence([torch.tensor(example['input_ids'][0]) for example in batch],batch_first= True,padding_value = berttokenizer.pad_token_id),
    
    'token_type_ids': pad_sequence([torch.tensor(example['token_type_ids'][0]) for example in batch],batch_first= True,padding_value = berttokenizer.pad_token_type_id)
    }

    dev_data_loader = DataLoader(dev_ds,sampler=dev_sampler,batch_size = batch_size,collate_fn=dev_batchify_fn,pin_memory=True )


    return dev_data_loader


#-----------------------------------------------------------
#模型构建

def get_model(model_name):

    model = BertForQuestionAnswering.from_pretrained(model_name)
    return model

class CrossEntropyLossForSQuAD(nn.Module):
    def __init__(self):
        super(CrossEntropyLossForSQuAD, self).__init__()

    def forward(self,y,label):
        start_logits, end_logits = y   # both shape are [batch_size, seq_len]
        start_position, end_position = label
        # start_position = start_position.unsqueeze(-1)
        # end_position = end_position.unsqueeze(-1)

        start_loss = F.cross_entropy(start_logits,start_position,reduction ='mean')

        end_loss = F.cross_entropy(end_logits,end_position,reduction = 'mean')

        total_loss =(start_loss + end_loss)/2

        return total_loss

@torch.no_grad()
def evaluate(model, raw_data,data_loader):

    model.eval()
    all_start_logits = []
    all_end_logits = []
    tic_eval = time.time()

    for batch in data_loader:
        input_ids,token_type_ids = batch
        start_logits_tensor,end_logits_tensor = model(input_ids,token_type_ids) 
    
        for idx in range(start_logits_tensor.shape[0]):
            if len(all_start_logits) % 1000 == 0 and len(all_start_logits):
                print("Processing example: %d" % len(all_start_logits))
                print('time per 1000:', time.time() - tic_eval)
                tic_eval = time.time()
            
            all_start_logits.append(start_logits_tensor.numpy()[idx])
            all_end_logits.append(end_logits_tensor.numpy()[idx])
        
    all_predictions, _ ,_ = compute_prediction(raw_data,data_loader.dataset,(all_start_logits,all_end_logits),False,20,30)

    squad_evaluate(examples=raw_data,preds= all_predictions,is_whitespace_splited=False)

    model.train()

  

def train(dev_ds_raw,model,tokenizer,tr_dataloader,dev_dataloader):


    # tr_dataloader = get_tr_data_loader(tr_ds, tokenizer)
    # dev_dataloader = get_dev_data_loader(dev_ds,tokenizer)

    # 训练过程中的最大学习率
    learning_rate = 3e-5 
    # 训练轮次
    epochs = 1
    # 学习率预热比例
    warmup_proportion = 0.1
    # 权重衰减系数，类似模型正则项策略，避免模型过拟合
    weight_decay = 0.01

    num_training_steps = len(tr_dataloader)*epochs

    num_warmup_steps = warmup_proportion*num_training_steps


    #学习率预热
    
    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_param = [
        p for n,p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])    
    ] #注意这里不要使用p.name
    optimizer = optim.AdamW(decay_param,lr = learning_rate,weight_decay=weight_decay)

    lr_schedule = get_linear_schedule_with_warmup(optimizer=optimizer,num_training_steps=num_training_steps,num_warmup_steps=num_warmup_steps)

    criterion = CrossEntropyLossForSQuAD()
    global_step = 0

    print('-------------------------------------------------')
    print("开始训练")
    for epoch in range(epochs+1):
        for step,batch in enumerate(tr_dataloader,start=1):
            global_step += 1
            # input_ids,segment_ids,start_positions,end_positions = batch
            input_ids = batch['input_ids']
            segment_ids = batch['token_type_ids']
            start_positions = batch['start_positions']
            end_positions = batch['end_positions']
            logits = model(input_ids = input_ids,token_type_ids = segment_ids)

            start_logits = logits['start_logits']
            end_logits = logits['end_logits']
            logits = (start_logits,end_logits)

            loss = criterion(logits,(start_positions,end_positions)) #y, label

            if global_step % 100 == 0 :
                print("global step %d, epoch: %d, batch: %d, loss: %.5f" % (global_step, epoch, step, loss))
            
            loss.backward()
            optimizer.step()
            lr_schedule.step()
            optimizer.zero_grad()
        
        evaluate(model = model ,raw_data = dev_ds_raw ,data_loader = dev_dataloader)
    
    
    print("正在保存模型")
    model.save_pretrained('./checkpoint')
    tokenizer.save_pretrained('./checkpoint')
    print('-------------------------------------------------')

    return model,dev_dataloader

#模型预测
# ----------------------------------------------------
@torch.no_grad()
def do_predict(model,raw_data,data_loader):

    print('------------------------------------------------')
    print("开始模型预测")
    model.eval()
    all_start_logits = []
    all_end_logits = []
    tic_eval = time.time()
    for batch in data_loader:
        # input_ids,token_type_ids = batch
        input_ids = batch['input_ids']
        token_type_ids = batch['token_type_ids']
        # start_logits_tensor,end_logits_tensor = model(input_ids,token_type_ids) 
        logits = model(input_ids,token_type_ids)
        start_logits_tensor = logits['start_logits']
        end_logits_tensor = logits['end_logits']

        for idx in range(start_logits_tensor.shape[0]):
            if len(all_start_logits) % 1000 == 0 and len(all_start_logits):
                print("Processing example: %d" % len(all_start_logits))
                print('time per 1000:', time.time() - tic_eval)
                tic_eval = time.time()

            all_start_logits.append(start_logits_tensor.numpy()[idx])
            all_end_logits.append(end_logits_tensor.numpy()[idx])

    
    all_predictions, _, _ = compute_prediction(
        raw_data, data_loader.dataset.dataset,
        (all_start_logits, all_end_logits), False, 20, 30)   
    print("模型预测结束")
    print('------------------------------------------------')

    count = 0
    for example in raw_data:
        
        count += 1
        print()
        print('问题：',example['question'])
        print('原文：',''.join(example['context']))
        print('答案：',all_predictions[example['id']])
        if count >=5:
            break

    model.train()



if __name__ == '__main__':

    filepath = './data'

    tr_ds_raw,dev_ds_raw = load_data(filepath)
    tr_ds_raw = extract_data(tr_ds_raw)
    dev_ds_raw = extract_data(dev_ds_raw)

    model_name = 'bert-base-chinese'

    berttokenizer = getTokenizer(model_name)
    model = get_model(model_name)

    tr_ds = trans_features(tr_ds_raw,tokenizer = berttokenizer,tr_or_dev='train')
    dev_ds = trans_features(dev_ds_raw,tokenizer=berttokenizer,tr_or_dev='dev')


    tr_dataloader = get_tr_data_loader(tr_ds, berttokenizer)
    dev_dataloader = get_dev_data_loader(dev_ds,berttokenizer)

    model,dev_dataloader = train(dev_ds_raw=dev_ds_raw,model = model ,tokenizer=berttokenizer, tr_dataloader=tr_dataloader ,dev_dataloader=dev_dataloader)

    do_predict(model=model , raw_data= dev_ds_raw ,data_loader = dev_dataloader)


