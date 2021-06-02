

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


class WeakSlotFillingDataLoader(DataLoader):
    
    def __init__(self, json_path, 
                 model=None, num_workers=0, tag_form='iob2', 
                 skip_empty=False, max_depth=None, *args, **kargs):
        self.model = model
        self.num_workers = num_workers
        self.max_depth = max_depth
        self.dataset = SimpleJsonDataset(json_path)
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
        tokens, tags, chunks_tags, chunks, mentions = [], [], [], [], []
        for item in batch:
            tokens.append(item['tokens'])
            if 'mentions' in item:
                mentions.append(item['mentions'])
            if 'tags' in item:
                tags.append(item['tags'])
            if 'chunk' in item:
                chunks.append(item['chunk'])
            if 'chunks_tags' in item:
                chunks_tags.append(item['chunks_tags'])
        
        rets = {
            'tokens': tokens,
            'mentions': mentions,
            'tags': tags,
            'chunks_tags': chunks_tags,
            'chunks': chunks,
        }
        
        if self.model is not None:
            rets['_tokens'] = self.model.token_indexing(tokens) # (B, T)
            if len(tags):
                rets['_tags'] = self.model.tag_indexing(tags)
            if len(chunks):
                rets['_chunks'] = self.model.chunk_tag_indexing(chunks)
            if len(chunks_tags):
                rets['_chunks_tags'] = self.model.chunk_tag_indexing(chunks_tags)
        
        return rets
    
    
class WeakTaggingTrainer(Trainer):
    def __init__(self, train_path, train_weak_path, test_path, valid_path,
                 batch_size=32, shuffle=True, model=None, num_workers=0, tag_form='iobes', 
                 max_depth=None, ratios={},
                 *args, **kargs):
        self.batch_size = batch_size
        self.ratios = ratios
        self.model = model
        self.train = WeakSlotFillingDataLoader(
            train_path, model=model, batch_size=batch_size, shuffle=shuffle, 
            num_workers=num_workers, tag_form=tag_form, max_depth=max_depth)
        self.train_weak = WeakSlotFillingDataLoader(
            train_weak_path, model=model, batch_size=batch_size, shuffle=shuffle, 
            num_workers=num_workers, tag_form=tag_form, max_depth=max_depth)
        self.test = WeakSlotFillingDataLoader(
            test_path, model=model, batch_size=batch_size, num_workers=num_workers, 
            tag_form=tag_form, max_depth=max_depth)
        self.valid = WeakSlotFillingDataLoader(
            valid_path, model=model, batch_size=batch_size, num_workers=num_workers, 
            tag_form=tag_form, max_depth=max_depth)
        
    def __iter__(self, *args, **kargs):
        
        class _Iter:
            def __init__(self, parent, supervised, weak_supervised):
                self.parent = parent
                self.supervised = supervised
                self.weak_supervised = weak_supervised
                
            def __next__(self):
                inputs_supervised = next(self.supervised)
                inputs_weak_supervised = next(self.weak_supervised)
                
                inputs = inputs_supervised
        
                inputs['supervised'] = self.parent.ratios.get('supervised', 1.)
                inputs['tokens_supervised'] = inputs_supervised['tokens']
                if '_tokens' in inputs_supervised:
                    inputs['_tokens_supervised'] = inputs_supervised['_tokens']
                inputs['tags_supervised'] = inputs_supervised['tags']
                if '_tags' in inputs_supervised:
                    inputs['_tags_supervised'] = inputs_supervised['_tags']
                    
                inputs['weak_supervised'] = self.parent.ratios.get('weak_supervised', 0.)
                inputs['tokens_weak_supervised'] = inputs_weak_supervised['tokens']
                if '_tokens' in inputs_weak_supervised:
                    inputs['_tokens_weak_supervised'] = inputs_weak_supervised['_tokens']
                inputs['chunks_weak_supervised'] = inputs_weak_supervised['chunks']
                if '_chunks' in inputs_weak_supervised:
                    inputs['_chunks_weak_supervised'] = inputs_weak_supervised['_chunks']
                    
                return inputs
        
        iter_obj = _Iter(
            self, 
            self.train.__iter__(*args, **kargs), 
            self.train_weak.__iter__(*args, **kargs),
        )
            
        return iter_obj
        
    def get_metrics(self, sents, pred_set_list, mentions_list, verbose=0):
        
        assert len(pred_set_list) == len(mentions_list)
        
        n_recall = n_pred = n_correct = 0
        for b in range(len(mentions_list)):
            mentions = mentions_list[b]

            _preds_set = pred_set_list[b]

            _labels_set = {
                (mention[0], tuple(mention[1])) for mention in mentions
            }

            n_recall += len(_labels_set)
            n_pred += len(_preds_set)
            n_correct += len(_labels_set & _preds_set)
            
            if verbose:
                print('====')
                print(sents[b])
                print(sorted(list(_labels_set)))
                print('--')
                print(sorted(list(_preds_set)))
                print('====')
            
        rec = n_correct / (n_recall + 1e-8)
        prec = n_correct / (n_pred + 1e-8)
        f1 = 2 / (1/(rec+1e-8) + 1/(prec+1e-8))
        return {  
            'precision' : prec,
            'recall' : rec,
            'f1' : f1,
            'confusion_dict' : None,
            'sents': sents,
            'pred_set_list': pred_set_list,
            'mentions_list': mentions_list,
        }
        
    
    def evaluate_model(self, model=None, verbose=0, test_type='valid'):
        
        if model is None:
            model = self.model
        
        if test_type == 'valid':
            g = self.valid
        elif test_type == 'test':
            g = self.test
        elif test_type == 'train_all':
            g = self.train_all
        else:
            g = []
            
        sents = []
        pred_set_list = []
        mentions_list = []
        for i, inputs in enumerate(g):
            rets = model.predict_step(inputs)
            pred_set_list += list(rets['pred_set'])
            mentions_list += [set((etype, tuple(espan)) for etype, espan in x) for x in rets['mentions']]
            sents += inputs['tokens']
        
        return self.get_metrics(sents=sents, pred_set_list=pred_set_list, mentions_list=mentions_list, verbose=verbose)
    
    
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
                
