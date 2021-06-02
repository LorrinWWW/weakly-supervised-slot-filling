

import os, sys
import numpy as np
import torch
import six
import json
import random
import time
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from utils import *
from itertools import combinations 

from .basics import *
from .base import *


class SlotFillingDataLoader(DataLoader):
    
    def __init__(self, json_path, 
                 model=None, num_workers=0, tag_form='iob2', 
                 skip_empty=False, *args, **kargs):
        self.model = model
        self.num_workers = num_workers
        self.dataset = JsonDataset(json_path, tag_form=tag_form, skip_empty=skip_empty)
        super().__init__(dataset=self.dataset, collate_fn=self._collect_fn, num_workers=num_workers, *args, **kargs)
        
        if self.num_workers == 0:
            pass # does not need warm indexing
        elif self.model is not None:
            print("warm indexing...")
            tmp = self.num_workers
            self.num_workers = 0
            for batch in self:
                pass
            self.num_workers = tmp
        else:
            print("warn: model is not set, skip warming.")
            print("note that if num_worker>0, vocab will be reset after each batch step,")
            print("thus a warming of indexing is required!")
        
    def _collect_fn(self, batch):
        tokens, tags = [], []
        for item in batch:
            tokens.append(item['tokens'])
            tags.append(item['tags'])
        
        rets = {
            'tokens': tokens,
            'tags': tags,
        }
        
        if self.model is not None:
            tokens = self.model.token_indexing(tokens)
            tags = self.model.tag_indexing(tags)
        
        rets['_tokens'] = tokens
        rets['_tags'] = tags
        
        return rets
    
    
class SlotFillingTrainer(Trainer):
    def __init__(self, train_path, test_path, valid_path,
                 batch_size=128, shuffle=True, model=None, num_workers=0, tag_form='iobes', 
                 *args, **kargs):
        self.batch_size = batch_size
        self.model = model
        self.train = SlotFillingDataLoader(train_path, model=model, batch_size=batch_size, 
                                       shuffle=shuffle, num_workers=num_workers, tag_form=tag_form)
        self.test = SlotFillingDataLoader(test_path, model=model, batch_size=batch_size, 
                                      num_workers=num_workers, tag_form=tag_form)
        self.valid = SlotFillingDataLoader(valid_path, model=model, batch_size=batch_size, 
                                       num_workers=num_workers, tag_form=tag_form)
    
    def evaluate_model(self, model=None, verbose=0, test_type='valid'):
        
        if model is None:
            model = self.model
        
        if test_type == 'valid':
            g = self.valid
        elif test_type == 'test':
            g = self.test
        else:
            g = []
            
        sents = []
        preds = []
        labels = []
        for i, inputs in enumerate(g):
            rets = model.predict_step(inputs)
            preds += rets['preds']
            labels += inputs['tags']
            sents += inputs['tokens']
        
        return get_seq_metrics(sents=sents, preds=preds, labels=labels, verbose=verbose)
    
    
    def _evaluate_during_train(self, model=None, trainer_target=None, args=None):
        
        if not hasattr(self, 'max_f1'):
            self.max_f1 = 0.0
        
        rets = trainer_target.evaluate_model(model, verbose=0, test_type='test')
        precision, recall, f1, confusion_dict = rets['precision'], rets['recall'], rets['f1'], rets['confusion_dict']
        print(f">> test prec:{precision:.4f}, rec:{recall:.4f}, f1:{f1:.4f}")
        rets = trainer_target.evaluate_model(model, verbose=0, test_type='valid')
        precision, recall, f1, confusion_dict = rets['precision'], rets['recall'], rets['f1'], rets['confusion_dict']
        print(f">> valid prec:{precision:.4f}, rec:{recall:.4f}, f1:{f1:.4f}")
        if f1 > self.max_f1:
            self.max_f1 = f1
            print('new max f1 on valid!')
            if args.model_write_ckpt:
                model.save(args.model_write_ckpt)