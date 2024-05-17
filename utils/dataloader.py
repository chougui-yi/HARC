import json
import random

import pandas as pd
from gensim.models import Word2Vec, KeyedVectors
from gensim.utils import simple_preprocess
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim

import torch
from torchvision import transforms

from transformers import AutoTokenizer


# 自定义数据集类
class MalwareDataset(Dataset):
    def __init__(self, csv_file, model_path, word2vec_path, is_onehot=True, tokern_type = "ALL"):
        self.total_number = None
        self.word2Vec_path = word2vec_path
        self.model_path = model_path
        self.data = []
        self.is_onehot = is_onehot
        self.tokenizer = None
        self.model = None
        self.load_parameter()
        self.set_gan()
        # print("use token:", tokern_type)
        self.tokern_type = tokern_type
        self.load_data(csv_file)

    def setToken(self, x):
        self.tokern_type = x

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx >= self.total_number:
            raise StopIteration
        try:
            row = self.data[idx]
            parameters = row['parameters']
            source_code = row['source_code']
            ast = row['ast']
            node_type_statistics = row['node_type_statistics']
            ast_deep = row['ast_deep']
            ast_total = row['ast_total']
            ast_width = row['ast_width']
            entropy = row['entropy']
            label = row['label']
            if self.tokern_type == "ALL":
                return parameters, source_code, ast, node_type_statistics, ast_deep, ast_total, ast_width, entropy, label
            
            if self.tokern_type == "parameters":
                return parameters, label
            if self.tokern_type == "source_code":
                return source_code, label
            if self.tokern_type == "ast":
                return ast, label
            if self.tokern_type == "node_type":
                return node_type_statistics, label

        except Exception as e:
            print("发现异常")
            print(e.__class__.__name__)
            print(e)

    def one_hot(self, x):
        label = torch.zeros(6)
        label[x] = 1
        return label

    def set_gan(self):
        self.datagan = [0, 0, 0, 0]
        self.datagan[0] = transforms.Compose([
            transforms.Resize((1, 32)),
        ])
        self.datagan[1] = transforms.Compose([
            transforms.Resize((1, self.tokenizer.model_max_length)),
        ])
        self.datagan[2] = transforms.Compose([
            transforms.Resize((1, self.tokenizer.model_max_length)),
        ])
        self.datagan[3] = transforms.Compose([
            transforms.Resize((1, self.tokenizer.model_max_length)),
        ])

    def load_data(self, file_path):
        preloaded_data = pd.read_csv(file_path)
        for idx, row in preloaded_data.iterrows():
            parameters = self.text_to_vectors(row['new_parameters']).unsqueeze(0)
            source_code = self.text_to_vectors(row['source_code']).unsqueeze(0)
            ast = self.text_to_vectors(row['ast']).unsqueeze(0)
            node_type_statistics = self.text_to_vectors(row['node_type_statistics']).unsqueeze(0)

            ast_deep = torch.tensor(row['ast_deep']).unsqueeze(0)
            ast_total = torch.tensor(row['ast_total']).unsqueeze(0)
            ast_width = torch.tensor(row['ast_width']).unsqueeze(0)
            entropy = torch.tensor(row['entropy']).unsqueeze(0)
            label = self.one_hot(row['label'])
            # src = self.datagan[1](source_code.float()/self.tokenizer.vocab_size)
            # print(src.shape, source_code.shape)
            self.data.append({
                # 'parameters': self.datagan[0](parameters.float() ).squeeze(0),
                # 'source_code': self.datagan[1](source_code.float() ).squeeze(0),
                # 'ast': self.datagan[2](ast.float()).squeeze(0),
                # 'node_type_statistics': self.datagan[3](
                #     node_type_statistics.float()).squeeze(0),
                
                'parameters': self.datagan[0](parameters.float() / self.tokenizer.vocab_size).squeeze(0),
                'source_code': self.datagan[1](source_code.float() / self.tokenizer.vocab_size).squeeze(0),
                'ast': self.datagan[2](ast.float() / self.tokenizer.vocab_size).squeeze(0),
                'node_type_statistics': self.datagan[3](
                    node_type_statistics.float() / self.tokenizer.vocab_size).squeeze(0),

                'ast_deep': ast_deep.float(),
                'ast_total': ast_total.float(),
                'ast_width': ast_width.float(),
                'entropy': entropy.float(),
                'label': label.float()
            })
        self.total_number = len(self.data)

    def load_parameter(self, type1=None):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.model is None:
            if type1 is None:
                return
            word2vec_model_path = ""
            if type1 == 'parameters':
                word2vec_model_path = self.word2Vec_path + "word2vec_model_parameters.bin"
            elif type1 == 'node_type':
                word2vec_model_path = self.word2Vec_path + "word2vec_model_node_type.bin"
            # 加载预训练好的Word2Vec模型
            model = Word2Vec.load(word2vec_model_path)
            self.model = model

    def text_to_vectors(self, string):

        wait_decode_str = string
        index_iter = len(wait_decode_str) // self.tokenizer.vocab_size + 1
        code_list = []

        for i in range(index_iter):
            crop_str = wait_decode_str[i * self.tokenizer.vocab_size: (i + 1) * self.tokenizer.vocab_size]
            code_list.append(
                self.tokenizer(crop_str, return_tensors="pt", truncation=True, padding='longest')['input_ids'][:, 1:-1])
        code_feature = torch.cat(code_list, 1)

        return code_feature

    def word2vec_vectors(self, string, type):
        self.load_parameter(type)
        # 分词并过滤掉不在词汇表中的词语
        words = string.split()
        words = [word for word in words if word in self.model.wv]

        # 计算每个词语的向量并对它们取平均作为字符串的向量表示
        if len(words) > 0:
            vector = sum(self.model.wv[word] for word in words) / len(words)
        else:
            vector = None

        return vector

    def _is_hex(self, token):
        # 检查token是否完全由16进制数字组成
        try:
            int(token, 16)
            return True
        except ValueError:
            return False


# 指定模型路径
# 数据集
def setDataLoader(train_path, test_path = None, batch_size = 1, token = "ALL"):
    model_path = "./models/BERT"
    word2vec_path = "./utils/"
    print("set token:", token, ",batch size:", batch_size)
    All_dataloader = MalwareDataset(
        train_path, model_path, word2vec_path, tokern_type = token
    )
    if test_path is None:
        train_size = int(All_dataloader.__len__() * 0.8)
        validate_size = All_dataloader.__len__() - train_size
        train_dataset, validate_dataset = torch.utils.data.random_split(All_dataloader
                                                                        , [train_size, validate_size])
    else:
        train_dataset = All_dataloader
        validate_dataset = MalwareDataset(
            test_path, model_path, word2vec_path, tokern_type = token
        )
        train_size, validate_size = len(train_dataset), len(validate_dataset)

    print("训练集大小: {} 测试集大小: {} , ".format(train_size, validate_size))
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    validate_loader = DataLoader(
        dataset=validate_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    return train_loader, validate_loader


