# 导入相关包
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


def load_dataset(path):
    # 生成加载数据的地址
    train_path = os.path.join(path, "train.txt")
    test_path = os.path.join(path, "test.txt")
    dict_path = os.path.join(path, "dict.txt")
    label_path = os.path.join(path, "label_dict.txt")

    # 加载词典
    with open(dict_path, "r", encoding="utf-8") as f:
        words = [word.strip() for word in f.readlines()]
        word_dict = dict(zip(words, range(len(words))))

    # 加载标签词典
    with open(label_path, "r", encoding="utf-8") as f:
        lines = [line.strip().split() for line in f.readlines()]
        lines = [(line[0], int(line[1])) for line in lines]
        label_dict = dict(lines)

    def load_data(data_path):
        data_set = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                label, text = line.strip().split("\t", maxsplit=1)
                data_set.append((text, label))
        return data_set
    train_set = load_data(train_path)
    test_set = load_data(test_path)

    return train_set, test_set, word_dict, label_dict

# 将文本序列进行分词，然后词转换为字典ID
def convert_corpus_to_id(data_set, word_dict, label_dict):
    tmp_data_set = []
    for text, label in data_set:
        text = [word_dict.get(word, word_dict["[oov]"]) for word in jieba.cut(text)]
        tmp_data_set.append((text, label_dict[label]))

    return tmp_data_set

# 构造训练数据，每次传入模型一个batch，一个batch里面有batch_size条样本
def build_batch(data_set,  batch_size, max_seq_len, shuffle=True, drop_last=True, pad_id=1):
    batch_text = []
    batch_label = []

    if shuffle:
        random.shuffle(data_set)

    for text, label in data_set:
        # 截断数据
        text = text[:max_seq_len]
        # 填充数据到固定长度
        if len(text) < max_seq_len:
            text.extend([pad_id]*(max_seq_len-len(text)))

        assert len(text) == max_seq_len
        batch_text.append(text)
        batch_label.append([label])

        if len(batch_text) == batch_size:
            yield np.array(batch_text).astype("int64"), np.array(batch_label).astype("int64")
            batch_text.clear()
            batch_label.clear()

    # 处理是否删掉最后一个不足batch_size 的batch数据
    if (not drop_last) and len(batch_label) > 0:
        yield np.array(batch_text).astype("int64"), np.array(batch_label).astype("int64")

# 定义注意力层
class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.w = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.v = nn.Parameter(torch.empty([1, hidden_size],dtype=torch.float32))

    def forward(self, inputs):
        # inputs:  [batch_size, seq_len, hidden_size]
        last_layers_hiddens = inputs
        # transposed inputs: [batch_size, hidden_size, seq_len]
        inputs = torch.transpose(inputs,dim0=1,dim1=2)
        # inputs: [batch_size, hidden_size, seq_len]
        inputs = torch.tanh(torch.matmul(self.w,inputs))
        # attn_weights: [batch_size, seq_len]
        attn_weights = torch.matmul(self.v,inputs)
        # softmax数值归一化
        attn_weights = F.softmax(attn_weights, dim=-1)
        # 通过attention后的向量值, attn_vectors: [batch_size, hidden_size]
        attn_vectors = torch.matmul(attn_weights,last_layers_hiddens)
        attn_vectors = torch.squeeze(attn_vectors, axis=1)

        return attn_vectors


class Classifier(nn.Module):
    def __init__(self, hidden_size, embedding_size, vocab_size, n_classes=14, n_layers=1, direction="bidirectional",
                 dropout_rate=0., init_scale=0.05):
        super(Classifier, self).__init__()
        # 表示LSTM单元的隐藏神经元数量，它也将用来表示hidden和cell向量状态的维度
        self.hidden_size = hidden_size
        # 表示词向量的维度
        self.embedding_size = embedding_size
        # 表示神经元的dropout概率
        self.dropout_rate = dropout_rate
        # 表示词典的的单词数量
        self.vocab_size = vocab_size
        # 表示文本分类的类别数量
        self.n_classes = n_classes
        # 表示LSTM的层数
        self.n_layers = n_layers
        # 用来设置参数初始化范围
        self.init_scale = init_scale

        # 定义embedding层
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_size,
                                      _weight=nn.Parameter(
                                          torch.empty(self.vocab_size, self.embedding_size).uniform_(-self.init_scale, self.init_scale)))
        
        # 定义LSTM，它将用来编码网络
        self.lstm = nn.LSTM(input_size=self.embedding_size, hidden_size=self.hidden_size,
                            num_layers=self.n_layers, bidirectional=True,
                                   dropout=self.dropout_rate)


        # 对词向量进行dropout
        self.dropout_emb = nn.Dropout(p=self.dropout_rate)

        # 定义Attention层
        self.attention = AttentionLayer(hidden_size=hidden_size*2 if direction == "bidirectional" else hidden_size)

        # 定义分类层，用于将语义向量映射到相应的类别
        self.cls_fc = nn.Linear(in_features=self.hidden_size*2 if direction == "bidirectional" else hidden_size,
                                       out_features=self.n_classes)

    def forward(self, inputs):
        # 获取训练的batch_size
        batch_size = inputs.shape[0]
        # 获取词向量并且进行dropout
        embedded_input = self.embedding(inputs)
        if self.dropout_rate > 0.:
            embedded_input = self.dropout_emb(embedded_input)

        # 使用LSTM进行语义编码
        last_layers_hiddens, (last_step_hiddens, last_step_cells) = self.lstm(embedded_input)

        # 进行Attention, attn_weights: [batch_size, seq_len]
        attn_vectors = self.attention(last_layers_hiddens)

        # 将其通过分类线性层，获得初步的类别数值
        logits = self.cls_fc(attn_vectors)

        return logits

# 加载数据集
root_path = "./dataset/"
train_set, test_set, word_dict, label_dict = load_dataset(root_path)
train_set = convert_corpus_to_id(train_set, word_dict, label_dict)
test_set = convert_corpus_to_id(test_set, word_dict, label_dict)
id2label = dict([(item[1], item[0]) for item in label_dict.items()])

# 参数设置
n_epochs = 3
vocab_size = len(word_dict.keys())
print(vocab_size)
batch_size = 128
hidden_size = 128
embedding_size = 128
n_classes = 14
max_seq_len = 32
n_layers = 1
dropout_rate = 0.2
learning_rate = 0.0001
direction = "bidirectional"

# 检测是否可以使用GPU，如果可以优先使用GPU
use_gpu = torch.cuda.is_available()
if use_gpu:
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

# 实例化模型
classifier = Classifier(hidden_size, embedding_size, vocab_size, n_classes=n_classes, n_layers=n_layers,
                        direction=direction, dropout_rate=dropout_rate)

# 指定优化器
optimizer = optim.Adam(classifier.parameters(), lr=learning_rate, betas=(0.9, 0.99))

# 定义模型评估类
class Metric:
    def __init__(self, id2label):
        self.id2label = id2label
        self.reset()

    def reset(self):
        self.total_samples = 0
        self.total_corrects = 0

    def update(self, real_labels, pred_labels):
        self.total_samples += real_labels.shape[0]
        self.total_corrects += np.sum(real_labels == pred_labels)

    def get_result(self):
        accuracy = self.total_corrects / self.total_samples
        return {"accuracy": accuracy}

    def format_print(self, result):
        print(f"Accuracy: {result['accuracy']:.4f}")


# 模型评估代码
def evaluate(model):
    model.eval()
    metric = Metric(id2label)

    for batch_texts, batch_labels in build_batch(test_set, batch_size, max_seq_len, shuffle=False, pad_id=word_dict["[pad]"]):
        # 将数据转换为Tensor类型
        batch_texts = torch.tensor(batch_texts).to(device)
        batch_labels = torch.tensor(batch_labels).to(device)

        # 执行模型的前向计算
        logits = model(batch_texts)

        # 使用softmax进行归一化
        probs = F.softmax(logits, dim=1)

        preds = probs.argmax(dim=1).cpu().numpy()
        batch_labels = batch_labels.squeeze().cpu().numpy()
        
        metric.update(real_labels=batch_labels, pred_labels=preds)
    
    result = metric.get_result()
    metric.format_print(result)

# 模型训练代码
# 记录训练过程中的中间变量
loss_records = []
def train(model):
    global_step = 0
    for epoch in range(n_epochs):
        model.train()
        for step, (batch_texts, batch_labels) in enumerate(build_batch(train_set, batch_size, max_seq_len, shuffle=True, pad_id=word_dict["[pad]"])):
            # 将数据转换为Tensor类型
            batch_texts = torch.tensor(batch_texts)
            batch_labels = torch.tensor(batch_labels)

            # 执行模型的前向计算
            logits = model(batch_texts)

            # 计算损失
            batch_labels_flat = batch_labels.view(-1)
            loss = F.cross_entropy(input=logits, target=batch_labels_flat, reduction='mean')

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % 200 == 0:
                loss_records.append((global_step, loss.item()))
                print(f"Epoch: {epoch+1}/{n_epochs} - Step: {step} - Loss: {loss.item()}")

            global_step += 1
        
        # 模型评估
        evaluate(model)

# 训练模型
train(classifier)

# 开始画图，横轴是训练step，纵轴是损失
loss_records =  np.array(loss_records)
steps, losses = loss_records[:, 0], loss_records[:, 1]

plt.plot(steps, losses, "-o")
plt.xlabel("step")
plt.ylabel("loss")
plt.savefig("./loss.png")
plt.show()

# 模型保存的名称
model_name = "classifier"
# 保存模型
torch.save(classifier.state_dict(), "{}.pdparams".format(model_name))
torch.save(optimizer.state_dict(), "{}.optparams".format(model_name))

# 模型预测代码
def infer(model, text):
    model.eval()
    # 数据处理
    tokens = [word_dict.get(word, word_dict["[oov]"]) for word in jieba.cut(text)]

    # 构造输入模型的数据
    tokens = torch.LongTensor(tokens).unsqueeze(0)

    # 计算发射分数
    with torch.no_grad():
        logits = model(tokens)
        probs = F.softmax(logits, dim=1)

    # 解析出分数最大的标签
    max_label_id = torch.argmax(logits, dim=1).item()
    pred_label = id2label[max_label_id]

    print("Label: ", pred_label)

title = "习近平主席对职业教育工作作出重要指示"
infer(classifier, title)