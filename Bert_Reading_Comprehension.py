'''
Date: 2023-05-01 23:25:51
LastEditors: turtlepig
LastEditTime: 2023-05-03 00:07:19
'''
import os
import jieba
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I
import torch.optim as optim
import matplotlib.pyplot as plt
from transformers import BertTokenizer

from utils import prepare_train_features, prepare_validation_features
from functools import partial

import collections
import time
import json

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

    
    # #test
    # for index in range(2):
        
    #     print(tr_ds[index]['context'])
    #     print(tr_ds[index]['qas'][0]['question'])
    #     print(tr_ds[index]['qas'][0]['answers'])
    #     print(tr_ds[index]['qas'][0]['answers'][0]['answer_start'])
        
    #     print()

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
                answers = q['answers']
                for answer in answers:
                    text = answer['text']
                    answer_start = answer['answer_start']
                    results.append({'question':question, 'context':context,'answers':text,'answer_starts':answer_start})
    return results


#调用BertTokenizer进行数据处理
#tokenizer的作用是将原始输入文本转化成模型可以接受的输入数据形式。
def getTokenizer(model_name):
    tokenizer = BertTokenizer.from_pretrained()# 使用'bert-base-chinese'
    return tokenizer

#源代码的简化版
def map(data,trans_func):
    data = trans_func(data)
    return

#在原案例中使用load_dataset()API默认读取到的数据集是MapDataset对象，MapDataset是paddle.io.Dataset的功能增强版本。其内置的map()方法适合用来进行批量数据集处理。map()方法传入的是一个用于数据处理的function。
#由于在本案例中没有构造MapDataset类，因此参照源代码自行书写map函数
def dataprocess(data,tokenizer,tr_or_dev):
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

path='./data'
load_data(path)