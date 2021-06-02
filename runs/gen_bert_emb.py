import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import argparse
import json
import pickle
from tqdm import tqdm

from flair.embeddings import BertEmbeddings
from flair.data import Token, Sentence

def form_sentence(tokens):
    s = Sentence()
    for w in tokens:
        s.add_token(Token(w))
    return s

def get_embs(s):
    ret = []
    for t in s:
        ret.append(t.get_embedding().cpu().numpy())
    return np.stack(ret, axis=0)



parser = argparse.ArgumentParser(description='Arguments for training.')

parser.add_argument('--dataset',
                    default='ACE05',
                    action='store',)

parser.add_argument('--model_name',
                    default='bert-base-multilingual-cased',
                    action='store',)

parser.add_argument('--lm_emb_save_path',
                    default='../wv/lm.emb.pkl',
                    action='store',)

parser.add_argument('--layers',
                    default='mean',
                    action='store',)

args = parser.parse_args()


if args.layers == 'mean':
    embedding = BertEmbeddings(args.model_name, layers='-1,-2,-3,-4', use_scalar_mix=True, pooling_operation="mean")
else:
    embedding = BertEmbeddings(args.model_name, layers=args.layers, pooling_operation="mean")
    
if 'pubmed' in args.model_name.lower():
    embedding.tokenizer.basic_tokenizer.do_lower_case = False


flag = args.dataset
dataset = []
with open(f'./datasets/unified/train.{flag}.json') as f:
    dataset += json.load(f)
with open(f'./datasets/unified/valid.{flag}.json') as f:
    dataset += json.load(f)
with open(f'./datasets/unified/test.{flag}.json') as f:
    dataset += json.load(f)
    
    
bert_emb_dict = {}
for item in tqdm(dataset):
    tokens = tuple(item['tokens'])
    s = form_sentence(tokens)
    embedding.embed(s)
    emb = get_embs(s)
    bert_emb_dict[tokens] = emb.astype('float16')
    
    
with open(args.lm_emb_save_path, 'wb') as f:
    pickle.dump(bert_emb_dict, f)