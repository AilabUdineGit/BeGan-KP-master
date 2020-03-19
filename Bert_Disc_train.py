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
                elif w == pykp.io.PEOS_WORD:
                    w = '.'
                # elif w == pykp.io.DIGIT:
                #     w = '<digit>'  # gl: token inserito nel vocabolario di Bert corrispondente a <digit>
                # else:  # gl: tutte le altre parole sono invalide e rendono la KP sbagliata (lasciare i valori
                #     1 == 1  # lasciando la parola inalterata e classificandola come errata anche il suo valore sarà correttamente appreso come valore 'sbagliato' in quanto presente in una fake KP
                str_kps += w + ' '
            # str_kps += ';'  # gl: was '<eos>'
            if str_kps[-2] != '.':  # nota: se w=pykp.io.PEOS_WORD allora ho già messo il punto, non mettere il punto e virgola
                str_kps += ';'
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


def train_one_batch(D_model, one2many_batch, generator, opt, perturb_std, bert_tokenizer, bert_model_name):
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
    # torch.save(pred_train_batch, 'prova/pred_train_batch.pt')  # gl saving tensors
    # torch.save(pred_mask_batch, 'prova/pred_mask_batch.pt')  # gl saving tensors
    # torch.save(pred_segment_batch, 'prova/pred_segment_batch.pt')  # gl saving tensors
    # torch.save(pred_label_batch, 'prova/pred_label_batch.pt')  # gl saving tensors
    # torch.save(target_train_batch, 'prova/target_train_batch.pt')  # gl saving tensors
    # torch.save(target_mask_batch, 'prova/target_mask_batch.pt')  # gl saving tensors
    # torch.save(target_segment_batch, 'prova/target_segment_batch.pt')  # gl saving tensors
    # torch.save(target_label_batch, 'prova/target_label_batch.pt')  # gl saving tensors

    # print(bert_model_name)
    if bert_model_name == 'BertForMultipleChoice':

        input_ids = []
        input_mask = []
        input_segment = []
        labels = []
        for i in range(opt.batch_size):  # batch_size
            # print(i)
            if i > len(pred_train_batch) - 1 or i > len(target_train_batch) - 1:  # gl: batch could be not complete
                break
            # print(pred_train_batch[i])
            value = random.random()
            appo_i = []
            appo_m = []
            appo_s = []
            if value < 0.5:
                appo_i.append(pred_train_batch[i])
                appo_i.append(target_train_batch[i])
                appo_m.append(pred_mask_batch[i])
                appo_m.append(target_mask_batch[i])
                appo_s.append(pred_segment_batch[i])
                appo_s.append(target_segment_batch[i])
                labels.append(1)
            else:
                appo_i.append(target_train_batch[i])
                appo_i.append(pred_train_batch[i])
                appo_m.append(target_mask_batch[i])
                appo_m.append(pred_mask_batch[i])
                appo_s.append(target_segment_batch[i])
                appo_s.append(pred_segment_batch[i])
                labels.append(0)
            input_ids.append(appo_i)
            input_mask.append(appo_m)
            input_segment.append(appo_s)

        input_ids = torch.tensor(input_ids, dtype=torch.long).to(devices)
        labels = torch.tensor(labels, dtype=torch.long).to(devices)
        input_mask = torch.tensor(input_mask, dtype=torch.long).to(devices)
        input_segment = torch.tensor(input_segment, dtype=torch.long).to(devices)

        output = D_model(input_ids,
                         attention_mask=input_mask,
                         token_type_ids=input_segment,
                         labels=labels)

        avg_batch_loss = output[0]
        # torch.save(output, 'prova/output.pt')  # gl saving tensors
        sum_real = 0
        sum_fake = 0
        positives = torch.tensor([0])
        for i in range(opt.batch_size):  # batch_size
            if i > len(pred_train_batch) - 1 or i > len(target_train_batch) - 1:  # gl: batch could be not complete
                break
            if labels[i].item() == 0:
                sum_real += output[1][i][0]
                sum_fake += output[1][i][1]
                if output[1][i][0] > output[1][i][1]:
                    positives += 1
            else:
                sum_real += output[1][i][1]
                sum_fake += output[1][i][0]
                if output[1][i][1] > output[1][i][0]:
                    positives += 1

        avg_real = sum_real / (i + 1)
        avg_fake = sum_fake / (i + 1)

    else:

        # gl: 4. transform to torch.tensor
        pred_input_ids = torch.tensor(pred_train_batch, dtype=torch.long).to(devices)
        target_input_ids = torch.tensor(target_train_batch, dtype=torch.long).to(devices)
        pred_input_mask = torch.tensor(pred_mask_batch, dtype=torch.float).to(devices)  # gl: was dtype=torch.float32
        target_input_mask = torch.tensor(target_mask_batch, dtype=torch.float).to(devices)  # gl: was dtype=torch.float32
        pred_input_segment = torch.tensor(pred_segment_batch, dtype=torch.long).to(devices)
        target_input_segment = torch.tensor(target_segment_batch, dtype=torch.long).to(devices)
        if opt.bert_labels == 1:
            target_input_labels = torch.tensor(target_label_batch, dtype=torch.float).to(devices)
            pred_input_labels = torch.tensor(pred_label_batch, dtype=torch.float).to(devices)
        else:
            target_input_labels = torch.tensor(target_label_batch, dtype=torch.long).to(devices)
            pred_input_labels = torch.tensor(pred_label_batch, dtype=torch.long).to(devices)

        # torch.save(pred_input_ids, 'prova/pred_input_ids.pt')  # gl saving tensors
        # torch.save(pred_input_mask, 'prova/pred_input_mask.pt')  # gl saving tensors
        # torch.save(pred_input_segment, 'prova/pred_input_segment.pt')  # gl saving tensors
        # torch.save(pred_input_labels, 'prova/pred_input_labels.pt')  # gl saving tensors
        # torch.save(target_input_ids, 'prova/target_input_ids.pt')  # gl saving tensors
        # torch.save(target_input_mask, 'prova/target_input_mask.pt')  # gl saving tensors
        # torch.save(target_input_segment, 'prova/target_input_segment.pt')  # gl saving tensors
        # torch.save(target_input_labels, 'prova/target_input_labels.pt')  # gl saving tensors

        # gl: 5. forward pass
        # print('pred_input_ids.shape:     ' + str(pred_input_ids.shape))
        # print('pred_input_mask.shape:    ' + str(pred_input_mask.shape))
        # print('pred_input_segment.shape: ' + str(pred_input_segment.shape))
        # print('pred_input_labels.shape:  ' + str(pred_input_labels.shape))
        # print('target_input_ids.shape:     ' + str(target_input_ids.shape))
        # print('target_input_mask.shape:    ' + str(target_input_mask.shape))
        # print('target_input_segment.shape: ' + str(target_input_segment.shape))
        # print('target_input_labels.shape:  ' + str(target_input_labels.shape))
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
        positives = torch.tensor([0.])
        for i in range(opt.batch_size):  # gl: evaluating a (custom) accuracy
            if i > len(pred_train_batch) - 1 or i > len(target_train_batch) - 1:  # gl: batch could be not complete
                break
            if target_output[1][i] > 0.5:
                positives += 0.5
            if pred_output[1][i] < 0.5:
                positives += 0.5
        # print(avg_batch_loss)  # gl
        # print(avg_real)  # gl
        # print(avg_fake)  # gl

    return avg_batch_loss, avg_real, avg_fake, positives


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
    bert_model_name = bert_model.model.__class__.__name__
    print(bert_model_name)

    # D_model = NetModel.from_pretrained(bert_model.pretrained_weights, num_labels=opt.bert_labels, output_hidden_states=True)  # gl
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

    # D_optimizer = torch.optim.Adam(D_model.parameters(), opt.learning_rate)

    print("Beginning with training Discriminator")
    print("########################################################################################################")
    total_epochs = 5
    for epoch in range(total_epochs):
        total_batch = 0
        print("Starting with epoch:", epoch)
        for batch_i, batch in enumerate(train_data_loader):
            # print('batch: ' + str(batch_i))  # gl: debug
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
            avg_batch_loss, _, _, _ = train_one_batch(D_model, batch, generator, opt, perturb_std, bert_tokenizer, bert_model_name)
            torch.nn.utils.clip_grad_norm_(D_model.parameters(), clip)
            avg_batch_loss.backward()
            
            D_optimizer.step()
            D_model.eval()

            if batch_i % 100 == 0:  # gl
                print('batch: ' + str(batch_i))

            if batch_i % 4000 == 0:
                print('validation pass ')
                total = 0
                valid_loss_total, valid_real_total, valid_fake_total, valid_positives_total = 0, 0, 0, 0
                for batch_j, valid_batch in enumerate(valid_data_loader):
                    # print(batch_j)
                    # torch.save(valid_batch, 'prova/valid_batch_error.pt')  # gl saving tensors
                    total += 1
                    valid_loss, valid_real, valid_fake, valid_positives = train_one_batch(D_model, valid_batch, generator, opt, perturb_std, bert_tokenizer, bert_model_name)
                    # print(len(valid_data_loader))
                    # print(valid_positives.item())
                    valid_loss_total += valid_loss.cpu().detach().numpy()
                    valid_real_total += valid_real.cpu().detach().numpy()
                    valid_fake_total += valid_fake.cpu().detach().numpy()
                    valid_positives_total += valid_positives.cpu().detach().numpy()
                    D_optimizer.zero_grad()

                print("Currently loss is ", valid_loss_total.item() / total)
                print("Currently real score is ", valid_real_total.item() / total)
                print("Currently fake score is ", valid_fake_total.item() / total)
                print("Currently accuracy is ", valid_positives_total.item() / (total * opt.batch_size))

                if best_valid_loss > valid_loss_total.item() / total:
                    # print(best_valid_loss)
                    print("Loss Decreases so saving the file ...............----------->>>>>")
                    state_dfs = D_model.state_dict()
                    torch.save(state_dfs, "Discriminator_checkpts/Bert_Discriminator_" + str(epoch) + ".pth.tar")
                    best_valid_loss = valid_loss_total.item() / total
                    # print(best_valid_loss)
                else:  # gl
                    print("Loss doesn't decrease so go on without saving the file")

    print("End of the Discriminator training")  # gl

######################################
