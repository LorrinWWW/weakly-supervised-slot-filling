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

from .basic_taggers import *
from .basic_joints import *
from .pnet_joints import *
from .pnet_attn_joints import *
from .pnet_route_joints import *
from .winpnet_attn_joints import *
from .pnet_caps_joints import *

from .rnet_attn_joints import *
from .rnet_caps_joints import *

from .caps_joints import *
from .trans_joints import *

from .pyramid_nest_ner import *
from .nest_ner import *

import copy

class Config:
    def __init__(
        self,
        token_emb_dim=100,
        char_encoder='lstm',
        char_emb_dim=0,
        tag_emb_dim=50,
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
        tag_form='iobes',
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        *args, **kargs,
    ):
        self.cased = cased
        self.token_emb_dim = token_emb_dim
        self.char_encoder = char_encoder
        self.char_emb_dim = char_emb_dim
        self.tag_emb_dim = tag_emb_dim
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
        self.tag_form = tag_form
        self.cls_vocab_size = cls_vocab_size
        self.device = device
        
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