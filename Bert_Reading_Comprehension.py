'''
Date: 2023-05-01 23:25:51
LastEditors: turtlepig
LastEditTime: 2023-05-02 09:15:33
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
    tr_ds = tr_ds['data'][0]['paragraphs']

    dev_path = open(dev_path, 'r',encoding='utf-8')
    dev_content = dev_path.read()
    dev_ds = json.loads(dev_content)
    

    #test
    for index in range(2):
        
        print(tr_ds[index]['context'])
        print(tr_ds[index]['qas'][0]['question'])
        print(tr_ds[index]['qas'][0]['answers'])
        print(tr_ds[index]['qas'][0]['answers'][0]['answer_start'])
        
        print()

def json_process(js_data):
    print()


path='./data'
load_data(path)