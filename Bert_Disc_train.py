#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 15:10:45 2019

@author: r17935avinash
"""

# #############################################
# #  gl: uncomment for better cuda debug
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# #############################################

# ############################### IMPORT LIBRARIES ###############################################################
import torch
import numpy as np
import pykp.io
import torch.nn as nn
from utils.statistics import RewardStatistics
from utils.time_log import time_since
import time
from sequence_generator import SequenceGenerator
from utils.report import export_train_and_valid_loss, export_train_and_valid_reward
import sys
import logging
import os
from evaluate import evaluate_reward
from pykp.reward import *
import math
EPS = 1e-8
import argparse
import config
import logging
import os
import json
from pykp.io import KeyphraseDataset
from pykp.model import Seq2SeqModel
from torch.optim import Adam
import pykp
from pykp.model import Seq2SeqModel
import train_ml
import train_rl

from utils.time_log import time_since
from utils.data_loader import load_data_and_vocab
from utils.string_helper import convert_list_to_kphs
import time
import numpy as np
import random
from torch import device 
from hierarchal_attention_Discriminator import Discriminator
from torch.nn import functional as F
from BERT_Discriminator import NetModel, NLPModel, NLP_MODELS
from transformers import AdamW
# ####################################################################################################
# def Check_Valid_Loss(valid_data_loader,D_model,batch,generator,opt,perturb_std):
    
# #### TUNE HYPERPARAMETERS ##############

# #  batch_reward_stat, log_selected_token_dist = train_one_batch(batch, generator, optimizer_rl, opt, perturb_std)
#########################################################


def build_kps_idx_list(kps, bert_tokenizer, opt):
    """
    evaluate the BERT tokenization for list of KPs
    :param kps: list of indexes of each word of KPs, for each src in batch
    :param bert_tokenizer: BERT tokenizer
    :param opt: optins values of the project
    :return: list of BERT indexes of each word of KPs, for each src in batch
    """
    str_kps_list = [[[opt.idx2word[w] for w in kp] for kp in b] for b in kps]
    # print(str_kps_list)
    str_list = []
    idx_list = []
    for b in str_kps_list:
        str_kps = '[SEP]'
        for kp in b:
            for w in kp:
                if w == pykp.io.UNK_WORD:
                    w = '[UNK]'  # gl: token inserito nel vocabolario di Bert corrispondente a <digit>
                # elif w == pykp.io.DIGIT:
                #     w = '<digit>'  # gl: token inserito nel vocabolario di Bert corrispondente a <digit>
                # else:  # gl: tutte le altre parole sono invalide e rendono la KP sbagliata (lasciare i valori
                #     1 == 1  # lasciando la parola inalterata e classificandola come errata anche il suo valore sarà correttamente appreso come valore 'sbagliato' in quanto presente in una fake KP
                str_kps += w + ' '
            str_kps += ';'  # gl: was '<eos>'
        str_kps += '[SEP]'
        # print(str_kps)
        bert_str_kps = bert_tokenizer.tokenize(str_kps)
        str_list.append(bert_str_kps)
        idx_list.append(bert_tokenizer.convert_tokens_to_ids(bert_str_kps))  # rivedere bene
    # print(str_list)
    # print(idx_list)

    return idx_list


def build_src_idx_list(src_str_list, bert_tokenizer):
    """
    evaluates the BERT tokenization of src
    :param src_str_list: batch of input documents
    :param bert_tokenizer: BERT tokenizer
    :return: list of BERT indexes of each word of src, for each src in batch
    """
    # print(src_str_list)
    str_list = []
    idx_list = []
    for src in src_str_list:
        str_src = '[CLS]'
        for w in src:
            if w == pykp.io.UNK_WORD:
                w = '[UNK]'  # gl: token inserito nel vocabolario di Bert corrispondente a <digit>
            # elif w == pykp.io.DIGIT:
            #     w = '<digit>'  # gl: token inserito nel vocabolario di Bert corrispondente a <digit>
            # else:  # gl: tutte le altre parole sono invalide e rendono la KP sbagliata (lasciare i valori
            #     1 == 1  # lasciando la parola inalterata e classificandola come errata anche il suo valore sarà correttamente appreso come valore 'sbagliato' in quanto presente in una fake KP
            str_src += w + ' '
        # str_src += '[SEP]'
        # print(str_src)

        bert_str_src = bert_tokenizer.tokenize(str_src)
        bert_idx_src = bert_tokenizer.convert_tokens_to_ids(bert_str_src)
        str_list.append(bert_str_src)
        idx_list.append(bert_idx_src)  # rivedere bene
    # print(str_list)
    # print(idx_list)

    return idx_list


def build_training_batch(src, kps, bert_tokenizer, opt, label):
    """
    build the final training data batch
    :param src: batch of input documents
    :param kps: batch of KPs
    :param bert_tokenizer: BERT tokenizer
    :param opt: project options
    :param label: 1 for true KPs; 0 for fake KPs
    :return:
    """
    training_batch = []
    mask_batch = []
    segment_batch = []
    label_batch = [label]*len(kps)
    pad_list = [bert_tokenizer.pad_token_id]*opt.bert_max_length
    for i in range(len(kps)):
        t = src[i][:opt.bert_max_length - len(kps[i])] + kps[i] + pad_list
        t = t[:opt.bert_max_length]
        m = [1 if k != bert_tokenizer.pad_token_id else 0 for k in t]
        s = [1]*(min(opt.bert_max_length-len(kps[i]), len(src[i])) + 1) + [0]*opt.bert_max_length
        s = s[:opt.bert_max_length]
        training_batch.append(t)
        mask_batch.append(m)
        segment_batch.append(s)
    # print(training_batch)
    # print(mask_batch)
    # print(segment_batch)
    # print(label_batch)

    return training_batch, mask_batch, segment_batch, label_batch


def train_one_batch(D_model, one2many_batch, generator, opt, perturb_std, bert_tokenizer):
    # torch.save(one2many_batch, 'prova/one2many_batch.pt')  # gl saving tensors
    # gl: one2many è una lista di 16 tensori o liste, ciascuno con 32 elementi (i tensori con una dimensione pari a 32)
    src, src_lens, src_mask, src_oov, oov_lists, src_str_list, trg_str_2dlist, trg, trg_oov, trg_lens, trg_mask, \
        _, title, title_oov, title_lens, title_mask, b_src, b_trg, b_src_str, b_trg_str, b_tok_map = one2many_batch
    one2many = opt.one2many
    one2many_mode = opt.one2many_mode
    if one2many and one2many_mode > 1:
        num_predictions = opt.num_predictions
    else:
        num_predictions = 1
    if torch.cuda.is_available():
        src = src.to(opt.device)
        src_mask = src_mask.to(opt.device)
        src_oov = src_oov.to(opt.device)
    if opt.title_guided:
        title = title.to(opt.device)
        title_mask = title_mask.to(opt.device)

    eos_idx = opt.word2idx[pykp.io.EOS_WORD]
    # print('eos_idx=' + str(eos_idx))  # gl: eos_idx=2
    delimiter_word = opt.delimiter_word
    # print('delimiter_word=' + str(delimiter_word))  # gl: delimiter_word=<sep>; sep_idx=4
    batch_size = src.size(0)
    topk = opt.topk
    reward_type = opt.reward_type
    reward_shaping = opt.reward_shaping
    baseline = opt.baseline
    match_type = opt.match_type
    regularization_type = opt.regularization_type  # # DNT
    regularization_factor = opt.regularization_factor  # # DNT
    devices = opt.device
    if regularization_type == 2:
        entropy_regularize = True
    else:
        entropy_regularize = False
    start_time = time.time()
    sample_list, log_selected_token_dist, output_mask, pred_eos_idx_mask, entropy, location_of_eos_for_each_batch, location_of_peos_for_each_batch = generator.sample(
        src, src_lens, src_oov, src_mask, oov_lists, opt.max_length, greedy=False, one2many=one2many,
        one2many_mode=one2many_mode, num_predictions=num_predictions, perturb_std=perturb_std,
        entropy_regularize=entropy_regularize, title=title, title_lens=title_lens, title_mask=title_mask)
    # torch.save(sample_list, 'prova/sample_list.pt')  # gl saving tensors

    pred_str_2dlist = sample_list_to_str_2dlist(sample_list, oov_lists, opt.idx2word, opt.vocab_size, eos_idx,
                                                delimiter_word, opt.word2idx[pykp.io.UNK_WORD], opt.replace_unk,
                                                src_str_list, opt.separate_present_absent, pykp.io.PEOS_WORD)
    # torch.save(pred_str_kps, 'prova/pred_str_kps.pt')  # gl saving tensors
    # torch.save(pred_str_2dlist, 'prova/pred_str_2dlist.pt')  # gl saving tensors
    # print(len(pred_str_2dlist))

    target_str_2dlist = convert_list_to_kphs(trg)
    # torch.save(target_str_2dlist, 'prova/target_str_2dlist.pt')  # gl saving tensors

    # gl: 1. verificare se nelle 2 liste di KPs ce ne sono uguali ed eventualmente metterle all'inizio nello stesso ordine (così il Discriminator capisce che sono simili)
    # gl: alla fine G dovrebbe creare samples uguali a quelli veri ma non necessariamente nello steso ordine;
    # gl: dandogli lo steso ordine lo aiuto a emettere un output = 0,5 perché non ci capisce più nulla (?)
    # for i in range(opt.batch_size):
    #     # print(i)
    #     if all(a in target_str_2dlist[i] for a in pred_str_2dlist[i]):
    #         print('same true and fake KPS at index=' + str(i))

    # gl: 2. Bert tokens indexes (NOTA: probabilmente non ho più bisogno di creare le variabili BERT nel dataloader, perché tanto qui devo farlo per forza)
    pred_idx_list = build_kps_idx_list(pred_str_2dlist, bert_tokenizer, opt)
    target_idx_list = build_kps_idx_list(target_str_2dlist, bert_tokenizer, opt)
    src_idx_list = build_src_idx_list(src_str_list, bert_tokenizer)
    # torch.save(pred_idx_list, 'prova/pred_idx_list.pt')  # gl saving tensors
    # torch.save(target_idx_list, 'prova/target_idx_list.pt')  # gl saving tensors

    # gl: 3. creare il batch di addestramento concatenando src sia con le fake che con le true KPs, limitando la lunghezza a 512 tokens e paddando
    # nota: sia per le true che per le fake KPS, src dovrebbe essere identico e quindi troncato alla stessa lunghezza
    pred_train_batch, pred_mask_batch, pred_segment_batch, pred_label_batch = \
        build_training_batch(src_idx_list, pred_idx_list, bert_tokenizer, opt, label=0)
    target_train_batch, target_mask_batch, target_segment_batch, target_label_batch = \
        build_training_batch(src_idx_list, target_idx_list, bert_tokenizer, opt, label=1)

    # gl: 4. transform to torch.tensor
    pred_input_ids = torch.tensor(pred_train_batch, dtype=torch.long).to(devices)
    target_input_ids = torch.tensor(target_train_batch, dtype=torch.long).to(devices)
    pred_input_mask = torch.tensor(pred_mask_batch, dtype=torch.float).to(devices)  # gl: was dtype=torch.float32
    target_input_mask = torch.tensor(target_mask_batch, dtype=torch.float).to(devices)  # gl: was dtype=torch.float32
    pred_input_segment = torch.tensor(pred_segment_batch, dtype=torch.long).to(devices)
    target_input_segment = torch.tensor(target_segment_batch, dtype=torch.long).to(devices)
    pred_input_labels = torch.tensor(pred_label_batch, dtype=torch.long).to(devices)
    target_input_labels = torch.tensor(target_label_batch, dtype=torch.long).to(devices)

    # torch.save(pred_input_ids, 'prova/pred_input_ids.pt')  # gl saving tensors
    # torch.save(pred_input_mask, 'prova/pred_input_mask.pt')  # gl saving tensors
    # torch.save(pred_input_segment, 'prova/pred_input_segment.pt')  # gl saving tensors
    # torch.save(pred_input_labels, 'prova/pred_input_labels.pt')  # gl saving tensors

    # gl: 5. forward pass
    # print('pred_input_ids.shape:     ' + str(pred_input_ids.shape))
    # print('pred_input_mask.shape:    ' + str(pred_input_mask.shape))
    # print('pred_input_segment.shape: ' + str(pred_input_segment.shape))
    # print('pred_input_labels.shape:  ' + str(pred_input_labels.shape))
    pred_output = D_model(pred_input_ids,
                          attention_mask=pred_input_mask,
                          token_type_ids=pred_input_segment,
                          labels=pred_input_labels)
    # torch.save(pred_output, 'prova/pred_output.pt')  # gl saving tensors

    target_output = D_model(target_input_ids,
                            attention_mask=target_input_mask,
                            token_type_ids=target_input_segment,
                            labels=target_input_labels)
    # torch.save(target_output, 'prova/target_output.pt')  # gl saving tensors

    avg_batch_loss = (pred_output[0] + target_output[0])
    avg_real = torch.mean(target_output[1])
    avg_fake = torch.mean(pred_output[1])
    # print(avg_batch_loss)  # gl
    # print(avg_real)  # gl
    # print(avg_fake)  # gl

    """
     src = [batch_size,abstract_seq_len]
     target_str_2dlist = list of list of true keyphrases
     pred_str_2dlist = list of list of false keyphrases
    
    """
    # total_abstract_loss = 0
    # batch_mine = 0
    # abstract_t = torch.Tensor([]).to(devices)
    # abstract_f = torch.Tensor([]).to(devices)
    # kph_t = torch.Tensor([]).to(devices)
    # kph_f = torch.Tensor([]).to(devices)
    # h_kph_t_size = 0
    # h_kph_f_size = 0
    # len_list_t, len_list_f = [], []
    # for idx, (src_list, pred_str_list, target_str_list) in enumerate(zip(src, pred_str_2dlist, target_str_2dlist)):  # gl: idx è l'indice del batch (0, ..., 31)
    #     # torch.save(src_list, 'prova/src_list-idx0.pt')  # gl saving tensors
    #     # torch.save(pred_str_list, 'prova/pred_str_list-idx0.pt')  # gl saving tensors
    #     # torch.save(target_str_list, 'prova/target_str_list-idx0.pt')  # gl saving tensors
    #     batch_mine += 1
    #     if len(target_str_list) == 0 or len(pred_str_list) == 0:
    #         continue
    #     h_abstract_t, h_kph_t = D_model.get_hidden_states(src_list, target_str_list)
    #     h_abstract_f, h_kph_f = D_model.get_hidden_states(src_list, pred_str_list)
    #     len_list_t.append(h_kph_t.size(1))
    #     len_list_f.append(h_kph_f.size(1))
    #     h_kph_t_size = max(h_kph_t_size, h_kph_t.size(1))
    #     h_kph_f_size = max(h_kph_f_size, h_kph_f.size(1))
    #
    # for idx, (src_list, pred_str_list, target_str_list) in enumerate(zip(src, pred_str_2dlist, target_str_2dlist)):  # gl: idx cicla sui batch
    #     batch_mine += 1
    #     if len(target_str_list) == 0 or len(pred_str_list) == 0:
    #         continue
    #     h_abstract_t, h_kph_t = D_model.get_hidden_states(src_list, target_str_list)
    #     # torch.save(h_abstract_t, 'prova/h_abstract_t-idx0.pt')  # gl saving tensors
    #     # torch.save(h_kph_t, 'prova/h_kph_t-idx0.pt')  # gl saving tensors
    #     h_abstract_f, h_kph_f = D_model.get_hidden_states(src_list, pred_str_list)
    #     # torch.save(h_abstract_f, 'prova/h_abstract_f-idx0.pt')  # gl saving tensors
    #     # torch.save(h_kph_f, 'prova/h_kph_f-idx0.pt')  # gl saving tensors
    #     p1d = (0, 0, 0, h_kph_t_size - h_kph_t.size(1))
    #     p2d = (0, 0, 0, h_kph_f_size - h_kph_f.size(1))
    #     h_kph_t = F.pad(h_kph_t, p1d)
    #     h_kph_f = F.pad(h_kph_f, p2d)
    #     abstract_t = torch.cat((abstract_t, h_abstract_t), dim=0)  # gl: qui gli hidden states estratti vengono concatenati in modo che alla fine si abbia comunque un tesnore con dim(0) pari a 32 cioè batch size
    #     abstract_f = torch.cat((abstract_f, h_abstract_f), dim=0)
    #     kph_t = torch.cat((kph_t, h_kph_t), dim=0)
    #     kph_f = torch.cat((kph_f, h_kph_f), dim=0)
    # _, real_rewards, abstract_loss_real = D_model.calculate_context(abstract_t, kph_t, 1, len_list_t)
    # _, fake_rewards, abstract_loss_fake = D_model.calculate_context(abstract_f, kph_f, 0, len_list_f)
    # # torch.save(real_rewards, 'prova/real_rewards.pt')  # gl saving tensors
    # # torch.save(abstract_loss_real, 'prova/abstract_loss_real.pt')  # gl saving tensors
    # # torch.save(fake_rewards, 'prova/fake_rewards.pt')  # gl saving tensors
    # # torch.save(abstract_loss_fake, 'prova/abstract_loss_fake.pt')  # gl saving tensors
    # avg_batch_loss = (abstract_loss_real + abstract_loss_fake)
    # avg_real = real_rewards
    # avg_fake = fake_rewards

    return avg_batch_loss, avg_real, avg_fake


def main(opt):
    clip = 5
    start_time = time.time()
    train_data_loader, valid_data_loader, word2idx, idx2word, vocab = load_data_and_vocab(opt, load_train=True)
    # torch.save(train_data_loader, 'prova/train_data_loader.pt')  # gl
    # torch.save(valid_data_loader, 'prova/valid_data_loader.pt')  # gl saving tensors
    load_data_time = time_since(start_time)
    logging.info('Time for loading the data: %.1f' % load_data_time)
    
    print("Data Successfully Loaded __.__.__.__.__.__.__.__.__.__.__.__.__.__.")
    model = Seq2SeqModel(opt)
    
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(opt.model_path))
        model = model.to(opt.gpuid)
    else:
        model.load_state_dict(torch.load(opt.model_path, map_location="cpu"))
    
    print("___________________ Generator Initialised and Loaded _________________________")
    generator = SequenceGenerator(model,
                                  bos_idx=opt.word2idx[pykp.io.BOS_WORD],
                                  eos_idx=opt.word2idx[pykp.io.EOS_WORD],
                                  pad_idx=opt.word2idx[pykp.io.PAD_WORD],
                                  peos_idx=opt.word2idx[pykp.io.PEOS_WORD],
                                  beam_size=1,
                                  max_sequence_length=opt.max_length,
                                  copy_attn=opt.copy_attention,
                                  coverage_attn=opt.coverage_attn,
                                  review_attn=opt.review_attn,
                                  cuda=opt.gpuid > -1
                                  )
    
    init_perturb_std = opt.init_perturb_std
    final_perturb_std = opt.final_perturb_std
    perturb_decay_factor = opt.perturb_decay_factor
    perturb_decay_mode = opt.perturb_decay_mode
    # hidden_dim = opt.D_hidden_dim  # gl: modificare con i dati BERT
    # embedding_dim = opt.D_embedding_dim  # gl: modificare con i dati BERT
    # n_layers = opt.D_layers  # gl: modificare con i dati BERT
    bert_model = NLP_MODELS[opt.bert_model].choose()  # gl

    # D_model = NetModel.from_pretrained(bert_model.pretrained_weights, num_labels=opt.bert_labels)  # gl
    D_model = bert_model.model.from_pretrained('bert-base-uncased', output_hidden_states=True, num_labels=opt.bert_labels)  # gl: prova semplificata

    bert_tokenizer = bert_model.tokenizer
    if torch.cuda.is_available():
        # D_model = Discriminator(opt.vocab_size, embedding_dim, hidden_dim, n_layers, opt.word2idx[pykp.io.PAD_WORD], opt.gpuid)
        D_model = D_model.cuda()  # gl
    # else:
    #     D_model = Discriminator(opt.vocab_size, embedding_dim, hidden_dim, n_layers, opt.word2idx[pykp.io.PAD_WORD], "cpu")
    print("The Discriminator Description is ", D_model)
    if opt.pretrained_Discriminator:
        if torch.cuda.is_available():
            D_model.load_state_dict(torch.load(opt.Discriminator_model_path))
            D_model = D_model.to(opt.gpuid)
        else:
            D_model.load_state_dict(torch.load(opt.Discriminator_model_path, map_location="cpu"))
    else:
        if torch.cuda.is_available():
            D_model = D_model.to(opt.gpuid)
        else:
            D_model.load_state_dict(torch.load(opt.Discriminator_model_path, map_location="cpu"))

    FULL_FINETUNING = True  # gl: che significa? eventualmente chiedere a Cancian
    if FULL_FINETUNING:
        param_optimizer = list(D_model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        # no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
    else:
        param_optimizer = list(D_model.classifier.named_parameters())
        optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer]}]
    D_optimizer = AdamW(optimizer_grouped_parameters, opt.bert_learning_rate, correct_bias=False)

    # # gl: it works, but input data and shapes are not the same as we need
    # choices = ["Hello, my dog is cute", "Hello, my cat is amazing", "Hello, my name is Joe"]
    # input_ids = torch.tensor([bert_tokenizer.encode(s, add_special_tokens=True) for s in choices]).unsqueeze(0).to(opt.device)  # Batch size 1, 2 choices
    # labels = torch.tensor(2).unsqueeze(0).to(opt.device)  # Batch size 1
    # outputs = D_model(input_ids, labels=labels)
    # loss, classification_scores = outputs[:2]

    # # from transformers import BertForSequenceClassification
    # # p_model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    # gl: nota importante: <eos> e <digit> vengono correttamente tokenizzati, ma poi fanno esplodere tutto quando si chiama la forward() (errori cuda)
    # prova = bert_tokenizer.eos_token
    # input_ids = torch.tensor(bert_tokenizer.encode("Hello, my dog is cute. [SEP] Its name is [UNK]. Not Dick", add_special_tokens=True)).unsqueeze(0).to(opt.device)  # Batch size 1
    # # input_ids = torch.tensor(bert_tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
    # labels = torch.tensor([1]).unsqueeze(0).to(opt.device)  # Batch size 1
    # outputs = D_model(input_ids, labels=labels)
    # loss, classification_scores = outputs[:2]

    # D_optimizer = torch.optim.Adam(D_model.parameters(), opt.learning_rate)

    print("Beginning with training Discriminator")
    print("########################################################################################################")
    total_epochs = 5
    for epoch in range(total_epochs):
        total_batch = 0
        print("Starting with epoch:", epoch)
        for batch_i, batch in enumerate(train_data_loader):
            print('batch: ' + str(batch_i))  # gl: debug
            # torch.save(batch, 'prova/batch.pt')  # gl
            best_valid_loss = 1000
            D_model.train()
            D_optimizer.zero_grad()
            
            if perturb_decay_mode == 0:  # do not decay
                perturb_std = init_perturb_std
            elif perturb_decay_mode == 1:  # exponential decay
                perturb_std = final_perturb_std + (init_perturb_std - final_perturb_std) * math.exp(-1. * total_batch * perturb_decay_factor)
            elif perturb_decay_mode == 2:  # steps decay
                perturb_std = init_perturb_std * math.pow(perturb_decay_factor, math.floor((1+total_batch)/4000))
            # torch.save(batch, 'prova/batch.pt')  # gl saving tensors
            avg_batch_loss, _, _ = train_one_batch(D_model, batch, generator, opt, perturb_std, bert_tokenizer)
            torch.nn.utils.clip_grad_norm_(D_model.parameters(), clip)
            avg_batch_loss.backward()
            
            D_optimizer.step()
            D_model.eval()

            if batch_i % 100 == 0:  # gl
                print('batch: ' + str(batch_i))

            if batch_i % 4000 == 0:
                print('validation pass ')
                total = 0
                valid_loss_total, valid_real_total, valid_fake_total = 0, 0, 0
                for batch_j, valid_batch in enumerate(valid_data_loader):
                    print(batch_j)
                    # torch.save(valid_batch, 'prova/valid_batch_error.pt')  # gl saving tensors
                    total += 1
                    valid_loss, valid_real, valid_fake = train_one_batch(D_model, valid_batch, generator, opt, perturb_std, bert_tokenizer)
                    valid_loss_total += valid_loss.cpu().detach().numpy()
                    valid_real_total += valid_real.cpu().detach().numpy()
                    valid_fake_total += valid_fake.cpu().detach().numpy()
                    D_optimizer.zero_grad()

                print("Currently loss is ", valid_loss_total.item() / total)
                print("Currently real loss is ", valid_real_total.item() / total)
                print("Currently fake loss is ", valid_fake_total.item() / total)
                
                if best_valid_loss > valid_loss_total.item() / total:
                    print("Loss Decreases so saving the file ...............----------->>>>>")
                    state_dfs = D_model.state_dict()
                    torch.save(state_dfs, "Discriminator_checkpts/Bert_Discriminator_" + str(epoch) + ".pth.tar")
                    best_valid_loss = valid_loss_total.item() / total

######################################
