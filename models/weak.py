import os, sys, pickle

import re
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
from data import *

from .base import *
from .basic_taggers import *

import copy


class _Encoder(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config 
        
        if self.config.tag_form.lower() == 'iob2':
            self.one_entity_n_tags = 2
        elif self.config.tag_form.lower() == 'iobes':
            self.one_entity_n_tags = 4
        else:
            raise Exception('no such tag form.')
        
        self.token_embedding = AllEmbedding(self.config)
        self.token_indexing = self.token_embedding.preprocess_sentences
        
        self.tag_indexing = get_tag_indexing(self.config)
        self.chunk_tag_indexing = get_tag_indexing(self.config)

        emb_dim = self.config.token_emb_dim + self.config.char_emb_dim + self.config.lm_emb_dim
        
        self.reduce_dim = nn.Linear(emb_dim, self.config.hidden_dim)
        init_linear(self.reduce_dim)
        
        self.sentence_encoding = LSTMEncoding(self.config)
        
        self.dropout_layer = nn.Dropout(self.config.dropout)
        
    def forward(self, inputs):
        
        if 'embeddings' in inputs:
            # skip
            return inputs
        
        if '_tokens' in inputs:
            sents = inputs['_tokens']
        else:
            sents = inputs['tokens']
            
        embeddings_list, masks = self.token_embedding(sents, return_list=True)
        embeddings = torch.cat(embeddings_list, dim=-1)
        embeddings = self.dropout_layer(embeddings)
        embeddings = self.reduce_dim(embeddings)
        embeddings = self.sentence_encoding(embeddings, mask=masks)
        embeddings = self.dropout_layer(embeddings)
        
        inputs['embeddings'] = embeddings
        inputs['masks'] = masks
        
        return inputs
    
    
class _Tagger(nn.Module):
    
    def __init__(self, config, encoder):
        super().__init__()
        
        self.config = config 
        self.encoder = encoder
        
        self.tagging_encoding = LSTMEncoding(self.config)
        
        self.dropout_layer = nn.Dropout(self.config.dropout)
        
        self.logits_layer = nn.Linear(self.config.hidden_dim, self.config.tag_vocab_size)
        init_linear(self.logits_layer)
        
        # optional
        self.chunk_logits_layer = nn.Linear(self.config.hidden_dim, 10)
        init_linear(self.chunk_logits_layer)
        
    def forward(self, inputs):
        
        self.encoder(inputs)
        
        embeddings = inputs['embeddings']
        masks = inputs['masks']
        
        embeddings = self.tagging_encoding(embeddings, mask=masks)
        embeddings = self.dropout_layer(embeddings)
        
        logits = self.logits_layer(embeddings)
        chunk_logits = self.chunk_logits_layer(embeddings)
        
        inputs['tagging_logits'] = logits
        inputs['chunking_logits'] = chunk_logits
        
        return inputs

    
class _Classifier(nn.Module):
    
    def __init__(self, config, encoder):
        super().__init__()
        
        self.config = config 
        self.encoder = encoder
        
        self.bio_embedding = nn.Embedding(10, self.config.hidden_dim)
        
        self.classifying_encoding = LSTMEncoding(self.config)
        
        self.dropout_layer = nn.Dropout(self.config.dropout)
        
        self.classifying_attention = AttentionEncoding(self.config)
        
        self.logits_layer = nn.Linear(self.config.hidden_dim, (self.config.tag_vocab_size-1)//self.encoder.one_entity_n_tags)
        init_linear(self.logits_layer)
        
    def forward(self, inputs, mode='sup'):
        
        self.encoder(inputs)
        
        embeddings = inputs['embeddings']
        masks = inputs['masks']
        
        B, N, H = embeddings.shape
        T = self.config.tag_vocab_size
        C = (T-1) // self.encoder.one_entity_n_tags
        
        if mode == 'sup':
            
            if '_tags' in inputs:
                tags = inputs['_tags']
            else:
                tags = self.encoder.tag_indexing(inputs['tags'])

            new_tags = tags[:, None].repeat(1, C, 1) # B, C, N
            m = torch.arange(0, C)[None, :, None].repeat(1, 1, N) * self.encoder.one_entity_n_tags # B, C, N (00000, 22222, 44444, ...)
            m_i = (m>=new_tags)|(new_tags>m+2)
            new_tags = (new_tags - m)
            new_tags[m_i] = 0
            new_tags = new_tags.to(embeddings.device)
            
            inputs['tags_BIO'] = new_tags

            phrase_embeddings = self.bio_embedding(new_tags) # B, C, N
            embeddings = embeddings[:, None] + phrase_embeddings
            embeddings = self.classifying_encoding(embeddings.view(B*C, N, self.config.hidden_dim)) # B*C, N
            embeddings = self.dropout_layer(embeddings)
            embeddings = self.classifying_attention(embeddings, mask=masks[:, None].repeat(1, C, 1).view(B*C, N)) # B*C, H
            logits = self.logits_layer(embeddings) # B*C, C
            logits = logits.view(B, C, C)
        
        elif mode == 'weak':
            
            if '_chunks' in inputs:
                chunks = inputs['_chunks'].to(embeddings.device)
            else:
                chunks = self.encoder.chunk_tag_indexing(inputs['chunks']).to(embeddings.device) # B, N
                inputs['_chunks'] = chunks
                
            phrase_embeddings = self.bio_embedding(chunks) # B, N
            embeddings = embeddings + phrase_embeddings
            embeddings = self.classifying_encoding(embeddings)
            embeddings = self.dropout_layer(embeddings)
            embeddings = self.classifying_attention(embeddings) # B, H
            logits = self.logits_layer(embeddings) # B, C
            
        inputs['classifying_logits'] = logits
        
        return inputs
    

class WeakDualModel(LSTMTagger):
    
    def set_embedding_layer(self):
        
        self.encoder = _Encoder(self.config)
        self.one_entity_n_tags = self.encoder.one_entity_n_tags
        self.token_embedding = self.encoder.token_embedding
        self.token_indexing = self.encoder.token_indexing
        
        self.tag_indexing = self.encoder.tag_indexing
        self.chunk_tag_indexing = self.encoder.chunk_tag_indexing
    
    def set_encoding_layer(self):
        
        self.tagger = _Tagger(self.config, self.encoder)
        self.classifier = _Classifier(self.config, self.encoder)
        
    def set_loss_layer(self):
        
        if self.config.crf and self.config.crf != 'None':
            self.crf_layer = eval(self.config.crf)(self.config)
        
        self.loss_layer = nn.CrossEntropyLoss(reduction='none')
        self.bloss_layer = nn.BCELoss(reduction='none')
        
    def check_attrs(self):
        # indexing
        assert hasattr(self, 'tag_indexing')
        assert hasattr(self, 'chunk_tag_indexing')
        assert hasattr(self, 'token_indexing')
        
    def get_default_trainer_class(self):
        return WeakTaggingTrainer
        
    def forward(self, inputs):
        '''
        L = L^{sup} + \lambda * L^{weak}
        '''
        
        loss = 0
        
        if inputs.get('supervised', 1.) > 0:
            tmp = inputs.copy()
            
            tmp['tokens'] = tmp['tokens_supervised']
            if '_tokens_supervised' in tmp:
                tmp['_tokens'] = tmp['_tokens_supervised']
                
            tmp['tags'] = tmp['tags_supervised']
            if '_tags_supervised' in tmp:
                tmp['tags'] = tmp['_tags_supervised']
            
            loss_sup = self.forward_supervised(tmp)
                
            loss = loss + loss_sup * inputs.get('supervised', 1.)
            
        if inputs.get('weak_supervised', 0.) > 0:
            tmp = inputs.copy()
            
            tmp['tokens'] = tmp['tokens_weak_supervised']
            if '_tokens_weak_supervised' in tmp:
                tmp['_tokens'] = tmp['_tokens_weak_supervised']
                
            tmp['chunks'] = tmp['chunks_weak_supervised']
            if '_chunks_weak_supervised' in tmp:
                tmp['_chunks'] = tmp['_chunks_weak_supervised']
            
            loss_wsup = self.forward_weak_supervised(tmp)
                
            loss = loss + loss_wsup * inputs.get('weak_supervised', 0.)

        inputs['loss'] = loss
            
        return inputs
    
    def forward_chunk_supervised(self, inputs):
        
        self.tagger(inputs)
        masks = inputs['masks']
        logits = inputs['chunking_logits']
        
        if '_tags' in inputs:
            tags = inputs['_tags'].to(logits.device)
        else:
            tags = self.tag_indexing(inputs['tags']).to(logits.device)
            
        loss = self.loss_layer(logits.permute(0, 2, 1), tags) * masks.float() # B, N
        loss = loss.sum(-1).mean(0)
        
        return loss
    
    def forward_weak_supervised(self, inputs):
        '''
        L^{weak}
        '''
        
        self.tagger(inputs)
        self.classifier(inputs, mode='weak')
        masks = inputs['masks']
        tagging_logits = inputs['tagging_logits'] # B, N, T
        classifying_logits = inputs['classifying_logits'] # B, C
        B, N, T = tagging_logits.shape
        C = (T-1) // self.one_entity_n_tags
        
        if '_chunks' in inputs:
            chunks = inputs['_chunks'].to(tagging_logits.device)
        else:
            chunks = self.chunk_tag_indexing(inputs['chunks']).to(tagging_logits.device) # B, N
        
        logits_O = tagging_logits[:, :, 0]
        logits_BI = tagging_logits[:, :, 1:].view(B, N, C, self.one_entity_n_tags) # B, N, C, 2
        logits_BIO = torch.cat([logits_O[:,:,None,None].repeat(1,1,C,1), logits_BI], -1) # B, N, C, 3
        
        logits_BIO_ = logits_BIO.permute(0, -1, 1, 2) # B, 3, N, C
        
        loss = self.loss_layer(logits_BIO_, chunks[:, :, None].repeat(1,1,C)) * masks[:, :, None].float() # B, N, C
        
        loss = loss.sum(1) # B, C
        
        weight = classifying_logits.softmax(-1)
        
        inputs['_tagging_loss'] = loss
        
        inputs['_classifying_prob'] = weight
        
        loss0 = (weight * loss).sum(-1).mean(0) 
        
        loss1 = (- (-loss).softmax(-1) * (weight+1e-6).log()).sum(-1).mean(0)
        
        loss = loss0 + loss1
        
        return loss
    
    def forward_supervised(self, inputs):
        '''
        L^{sup}
        '''
                
        self.tagger(inputs)
        self.classifier(inputs, mode='sup')
        masks = inputs['masks']
        tagging_logits = inputs['tagging_logits'] # B, N, T
        classifying_logits = inputs['classifying_logits'] # B, C
        B, N, T = tagging_logits.shape
        C = (T-1) // self.one_entity_n_tags
        
        # share parameters for O tag
        logits_O = tagging_logits[:, :, 0]
        logits_BI = tagging_logits[:, :, 1:].view(B, N, C, self.one_entity_n_tags) # B, N, C, 2
        logits_BIO = torch.cat([logits_O[:,:,None,None].repeat(1,1,C,1), logits_BI], -1) # B, N, C, 3
        
        tags_BIO = inputs['tags_BIO'] # B, C, N
        
        tagging_loss = self.loss_layer(logits_BIO.permute(0, 3, 2, 1), tags_BIO) * masks[:, None].float() # B, C, N
        tagging_loss = tagging_loss.sum(-1).sum(-1).mean(0)
        
        aux_labels = torch.arange(0, C)[None].repeat(B, 1).to(classifying_logits.device) # B, C
        aux_masks = (tags_BIO!=0).any(-1).float() # B, C. mask all-O-tag sequences
        
        classifying_loss = self.loss_layer(classifying_logits.permute(0, 2, 1), aux_labels) * aux_masks
        classifying_loss = classifying_loss.sum(-1).mean(0)
        
        loss = tagging_loss + classifying_loss
        
        return loss
    
    def predict_step(self, inputs):
        
        self.tagger(inputs)
        
        logits = inputs['tagging_logits']
        masks = inputs['masks']
        
        B, N, T = logits.shape
        C = (T-1) // self.one_entity_n_tags
        
        logits_O = logits[:, :, 0]
        logits_BI = logits[:, :, 1:].view(B, N, C, self.one_entity_n_tags) # B, N, C, 2
        logits_BIO = torch.cat([logits_O[:,:,None,None].repeat(1,1,C,1), logits_BI], -1) # B, N, C, 3
        
        preds = (logits_BIO.argmax(dim=-1) * masks[:, :, None]).cpu().detach().numpy() # B, N, C
        preds_scores = (logits_BIO.max(dim=-1).values * masks[:, :, None]).cpu().detach().numpy() # B, N, C 
        preds_scores = preds_scores.sum(1) # B, C
        
        preds = np.array(preds)
        preds = self.chunk_tag_indexing.inv(preds.reshape([B, N*C]))
        preds = np.array(preds).reshape(B, N, C)
        
        ### very slow implementation, optimize it in the future
        pred_set = []
        for b in range(B): # N, C
            
            pred_tags = preds[b]
            pred_scores_ = preds_scores[b]
            current_pred_dict = {}
            current_token_dict = {}
            
            for c in range(C):
                pred_score = pred_scores_[c]
                spans = tag2span(pred_tags[:, c], False, True)
                etype = self.tag_indexing.idx2token(c*2+1).split('-')[-1]
                
                if etype == 'O' or etype == '[UNK]':
                    continue
                    
                for span in spans:
                    span = tuple(span)
                    
                    # when two spans are overlapping, retain the one with higger confidence.
                    skip = False
                    for i in range(*span):
                        if i in current_token_dict and pred_score < current_pred_dict[current_token_dict[i]][0]:
                            skip = True
                            break
                    if skip:
                        continue
                        
                    for i in range(*span):
                        current_token_dict[i] = span
                        
                    current_pred_dict[span] = (pred_score, etype)
                    
            pred_set.append({
                (etype, span) for span, (_, etype) in current_pred_dict.items()
            })
            
        inputs['pred_set'] = pred_set
                
        return inputs
    
    
    def train_step(self, inputs):
        
        rets = self(inputs)
        loss = rets['loss']
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 5.)
        
        return rets
    
    
    def save_ckpt(self, path):
        torch.save(self.state_dict(), path+'.pt')
        with open(path+'.vocab.pkl', 'wb') as f:
            pickle.dump(self.token_embedding.token_indexing.vocab, f)
        with open(path+'.char_vocab.pkl', 'wb') as f:
            pickle.dump(self.token_embedding.char_indexing.vocab, f)
        with open(path+'.tag_vocab.pkl', 'wb') as f:
            pickle.dump(self.tag_indexing.vocab, f)
        with open(path+'.chunk_tag_vocab.pkl', 'wb') as f:
            pickle.dump(self.chunk_tag_indexing.vocab, f)
            
    def load_ckpt(self, path):
        self.load_state_dict(torch.load(path+'.pt'))
        with open(path+'.vocab.pkl', 'rb') as f:
            self.token_embedding.token_indexing.vocab = pickle.load(f)
            self.token_embedding.token_indexing.update_inv_vocab()
        with open(path+'.char_vocab.pkl', 'rb') as f:
            self.token_embedding.char_indexing.vocab = pickle.load(f)
            self.token_embedding.char_indexing.update_inv_vocab()
        with open(path+'.tag_vocab.pkl', 'rb') as f:
            self.tag_indexing.vocab = pickle.load(f)
            self.tag_indexing.update_inv_vocab()
        with open(path+'.chunk_tag_vocab.pkl', 'rb') as f:
            self.chunk_tag_indexing.vocab = pickle.load(f)
            self.chunk_tag_indexing.update_inv_vocab()
    

    