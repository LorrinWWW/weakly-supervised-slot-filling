import os, sys, pickle
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from utils import *
from layers import *
from functions import *

import copy

class Config:
    def __init__(
        self,
        token_emb_dim=100,
        char_encoder='lstm',
        char_emb_dim=0,
        cased=False,
        hidden_dim=512,
        num_layers=1,
        crf='',
        loss_reduction='sum',
        maxlen=None,
        dropout=0.5,
        optimizer='adam',
        lr=1e-3,
        vocab_size=50000,
        vocab_file='./datasets/vocab.txt',
        tag_vocab_size=100,
        cls_vocab_size=100,
        lm_emb_dim=0,
        lm_emb_path=None,
        tag_form='iobes',
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        *args, **kargs,
    ):
        self.cased = cased
        self.token_emb_dim = token_emb_dim
        self.char_encoder = char_encoder
        self.char_emb_dim = char_emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.crf = crf
        self.loss_reduction = loss_reduction
        self.dropout = dropout
        self.lr = lr
        self.optimizer = optimizer
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.vocab_file = vocab_file
        self.tag_vocab_size = tag_vocab_size
        self.cls_vocab_size = cls_vocab_size
        self.tag_form = tag_form
        self.device = device
        self.lm_emb_dim = lm_emb_dim
        self.lm_emb_path = lm_emb_path
        
    def __call__(self, **kargs):
        
        obj = copy.copy(self)
        for k, v in kargs.items():
            setattr(obj, k, v)
        return obj
    
    def copy(self, deep=False):
        if deep:
            return copy.deepcopy(self)
        else:
            return copy.copy(self)
        

from .basic_taggers import *
from .weak import *