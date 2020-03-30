#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 15:10:45 2019
@author: r17935avinash
"""

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
import numpy as np

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
from Bert_Disc_train import build_kps_idx_list,build_src_idx_list,build_training_batch

#####################################################################################################

torch.autograd.set_detect_anomaly(True)


# #  batch_reward_stat, log_selected_token_dist = train_one_batch(batch, generator, optimizer_rl, opt, perturb_std)
#########################################################


def train_one_batch(D_model, one2many_batch, generator, opt, bert_tokenizer, perturb_std):
    src, src_lens, src_mask, src_oov, oov_lists, src_str_list, trg_str_2dlist, trg, trg_oov, trg_lens, trg_mask, _, title, title_oov, title_lens, title_mask, _, _, _, _,_ = one2many_batch
    one2many = opt.one2many
    one2many_mode = opt.one2many_mode
    if one2many and one2many_mode > 1:
        num_predictions = opt.num_predictions
    else:
        num_predictions = 1

    src = src.to(opt.device)
    src_mask = src_mask.to(opt.device)
    src_oov = src_oov.to(opt.device)
    if opt.title_guided:
        title = title.to(opt.device)
        title_mask = title_mask.to(opt.device)
    eos_idx = opt.word2idx[pykp.io.EOS_WORD]
    delimiter_word = opt.delimiter_word
    batch_size = src.size(0)
    topk = opt.topk
    reward_type = opt.reward_type
    reward_shaping = opt.reward_shaping
    baseline = opt.baseline
    match_type = opt.match_type
    regularization_type = opt.regularization_type  # # DNT
    regularization_factor = opt.regularization_factor  # # DNT
    if regularization_type == 2:
        entropy_regularize = True
    else:
        entropy_regularize = False

    start_time = time.time()
    # print('title = ', title)
    # print('title_lens = ', title_lens)
    # print('title_mask = ', title_mask)
    sample_list, log_selected_token_dist, output_mask, pred_eos_idx_mask, entropy, location_of_eos_for_each_batch, location_of_peos_for_each_batch = generator.sample(
        src, src_lens, src_oov, src_mask, oov_lists, opt.max_length, greedy=False, one2many=one2many,
        one2many_mode=one2many_mode, num_predictions=num_predictions, perturb_std=perturb_std,
        entropy_regularize=entropy_regularize, title=title, title_lens=title_lens, title_mask=title_mask)
    # opt.max_length=6; one2many=True; one2many_mode=1; num_predictions=1; perturb_std=0; entropy_regularize=False; title=None; title_lens=None; title_mask=None;

    pred_str_2dlist = sample_list_to_str_2dlist(sample_list, oov_lists, opt.idx2word, opt.vocab_size, eos_idx,
                                                delimiter_word, opt.word2idx[pykp.io.UNK_WORD], opt.replace_unk,
                                                src_str_list, opt.separate_present_absent, pykp.io.PEOS_WORD)
    # print('pred_str_2dlist = ', pred_str_2dlist)
    target_str_2dlist = convert_list_to_kphs(trg)
    # print('target_str_2dlist = ', target_str_2dlist)
    # print('src = ', src, src.size())
    """
     src = [batch_size,abstract_seq_len]
     target_str_2dlist = list of list of true keyphrases
     pred_str_2dlist = list of list of false keyphrases

    """
    if torch.cuda.is_available():
        devices = opt.gpuid
    else:
        devices = "cpu"

    total_abstract_loss = 0
    batch_mine = 0
    abstract_f = torch.Tensor([]).to(devices)
    kph_f = torch.Tensor([]).to(devices)
    h_kph_f_size = 0
    len_list_t, len_list_f = [], []

    for idx, (src_list, pred_str_list, target_str_list) in enumerate(zip(src, pred_str_2dlist, target_str_2dlist)):
        batch_mine += 1
        if len(pred_str_list) == 0:
            continue

        pred_idx_list = build_kps_idx_list(pred_str_2dlist, bert_tokenizer, opt)
        target_idx_list = build_kps_idx_list(target_str_2dlist, bert_tokenizer, opt)
        src_idx_list = build_src_idx_list(src_str_list, bert_tokenizer)
        # torch.save(pred_idx_list, 'prova/pred_idx_list.pt')  # gl saving tensors
        # torch.save(target_idx_list, 'prova/target_idx_list.pt')  # gl saving tensors

        pred_train_batch, pred_mask_batch, pred_segment_batch, pred_label_batch = \
            build_training_batch(src_idx_list, pred_idx_list, bert_tokenizer, opt, label=0)
        target_train_batch, target_mask_batch, target_segment_batch, target_label_batch = \
            build_training_batch(src_idx_list, target_idx_list, bert_tokenizer, opt, label=1)

        # gl: 4. transform to torch.tensor
        pred_input_ids = torch.tensor(pred_train_batch, dtype=torch.long).to(devices)
        target_input_ids = torch.tensor(target_train_batch, dtype=torch.long).to(devices)
        pred_input_mask = torch.tensor(pred_mask_batch, dtype=torch.float).to(devices)  # gl: was dtype=torch.float32
        target_input_mask = torch.tensor(target_mask_batch, dtype=torch.float).to(
            devices)  # gl: was dtype=torch.float32
        pred_input_segment = torch.tensor(pred_segment_batch, dtype=torch.long).to(devices)
        target_input_segment = torch.tensor(target_segment_batch, dtype=torch.long).to(devices)
        pred_input_labels = torch.tensor(pred_label_batch, dtype=torch.long).to(devices)
        target_input_labels = torch.tensor(target_label_batch, dtype=torch.long).to(devices)

        torch.save(pred_input_ids, 'prova/pred_input_ids.pt')
        torch.save(pred_input_mask, 'prova/pred_input_mask.pt')
        torch.save(pred_input_segment, 'prova/pred_input_segment.pt')
        torch.save(pred_input_labels, 'prova/pred_input_labels.pt')
        print('pred_input_ids.shape:     ' + str(pred_input_ids.shape))
        print('pred_input_mask.shape:    ' + str(pred_input_mask.shape))
        print('pred_input_segment.shape: ' + str(pred_input_segment.shape))
        print('pred_input_labels.shape:  ' + str(pred_input_labels.shape))

        print('pred_input_ids.shape:     ' + str(pred_input_ids[0,:512]))
        print((pred_input_ids[1,:512] == 102).nonzero())
        print('pred_input_mask.shape:    ' + str(pred_input_mask[0,:512]))
        print('pred_input_segment.shape: ' + str(pred_input_segment[0,:512]))
        print('pred_input_labels.shape:  ' + str(pred_input_labels.shape))

        #f_output = D_model(pred_input_ids,
         #                      attention_mask=pred_input_mask,
         #                      token_type_ids=pred_input_segment,
         #                      labels=pred_input_labels)
        # torch.save(f_output,'prova/f_out.pt')
        # print('f_output.shape'+str(len(f_output)))
        # loss, logits =f_output[:2]
        # print("loss",loss.shape)
        # print("logits",logits[12][0])
        # a = (pred_input_ids[:, :512] == 102).nonzero()
        # #b = (pred_input_ids[1,:512] == 102).nonzero()
        # print("a",a,a.shape)
        # print(a[0,1].item())
        # print(a[2,1].item())
        # t1 = a[0,1].item()
        # t2 = a[2, 1].item()
        # #t1 = a[0, 0].item()
        # #t2 = b[0, 0].item()
        # h_abstract_f1 = logits[12][0, :t1, :780]
        # h_kph_f1 = logits[12][0, t1:512, :780]
        # h_abstract_f2 = logits[12][1, :t2, :780]
        # h_kph_f2 = logits[12][1, t2:512, :780]
        #
        # h_kph_f_size1 = max(512-t1, 512-t2)
        # h_abstract_f_size1 = max(t1, t2)
        #
        # h_abstract_f_size = h_abstract_f_size1
        # h_kph_f_size = h_kph_f_size1
        #
        #
        # print(h_abstract_f1.shape, type(h_abstract_f1))
        # print(h_abstract_f2.shape, type(h_abstract_f2))
        # print(h_kph_f1.shape)
        # print(h_kph_f2.shape)
        # print('pred_input_ids.shape:     ' + str(pred_input_ids.shape))

        h_abstract_f, h_kph_f = D_model.get_hidden_states(pred_input_ids,
                            attention_mask=pred_input_mask,
                            token_type_ids=pred_input_segment,
                            labels=pred_input_labels)
        #h_abstract_f=logits[12][:2,:505,:780]
        #h_kph_f = logits[12][:2,506]
        #h_abstract_f, h_kph_f = D_model.get_hidden_states(src_list, pred_str_list)
        len_list_f.append(h_kph_f.size(1))
        h_kph_f_size = max(h_kph_f_size, h_kph_f.size(1))

    pred_str_new2dlist = []
    log_selected_token_total_dist = torch.Tensor([]).to(devices)
    print("src",src.shape)
    for idx, (src_list, pred_str_list, target_str_list) in enumerate(zip(src, pred_str_2dlist, target_str_2dlist)):
    #for idx in range(pred_input_ids.shape[0]):
        batch_mine += 1
        if len(target_str_list) == 0 or len(pred_str_list) == 0:
            continue
        pred_str_new2dlist.append(pred_str_list)
        log_selected_token_total_dist = torch.cat((log_selected_token_total_dist, log_selected_token_dist[idx]), dim=0)
        #h_abstract_f, h_kph_f = D_model.get_hidden_states(src_list, pred_str_list)
        #f_output = D_model(pred_input_ids,
         #                  attention_mask=pred_input_mask,
          #                 token_type_ids=pred_input_segment,
           #                labels=pred_input_labels)

        h_abstract_f, h_kph_f = D_model.get_hidden_states(pred_input_ids,
                                                          attention_mask=pred_input_mask,
                                                          token_type_ids=pred_input_segment,
                                                          labels=pred_input_labels)
        p2d = (0, 0, 0, h_kph_f_size - h_kph_f.size(1))
        h_kph_f = F.pad(h_kph_f, p2d)
        print("abstract_f", abstract_f.shape)
        print("h_abstract_f",h_abstract_f.shape)
        abstract_f = h_abstract_f
        kph_f = h_kph_f
        #abstract_f = torch.cat((abstract_f, h_abstract_f), dim=0)
        #kph_f = torch.cat((kph_f, h_kph_f), dim=0)

    # print('idx = ', idx)  # idx=31: è l'indice che cicla sul batch da 0 a 31
    # print('abstract_f = ', abstract_f, abstract_f.size())  # torch.Size([32, 385, 150]) = [batch, max_length, hidden_dim]
    # print('kph_f = ', kph_f, kph_f.size())  # torch.Size([32, 3, 150]) = [batch, padded_kp_length, hidden_dim]
    # print('pred_str_new2dlist = ', pred_str_new2dlist)  # lista di liste, lista delle KP predette
    opt.multiple_rewards = True  # gl: necessario perché sia single che multiple sono False in opt.
    if opt.multiple_rewards:
        # print('multiple reward!')  # gl
        print(abstract_f.shape)
        len_abstract = abstract_f.size(1)
        total_len = log_selected_token_dist.size(1)
        log_selected_token_total_dist = log_selected_token_total_dist.reshape(-1, total_len)
        all_rewards = D_model.calculate_rewards(abstract_f, kph_f, len_abstract, len_list_f, pred_str_new2dlist,
                                                total_len)
        all_rewards = all_rewards.reshape(-1, total_len)
        calculated_rewards = log_selected_token_total_dist * all_rewards.detach()
        individual_rewards = torch.sum(calculated_rewards, dim=1)
        J = torch.mean(individual_rewards)
        return J

    elif opt.single_rewards:
        # print('single reward!')  # gl
        len_abstract = abstract_f.size(1)
        total_len = log_selected_token_dist.size(1)
        log_selected_token_total_dist = log_selected_token_total_dist.reshape(-1, total_len)
        all_rewards = D_model.calculate_rewards(abstract_f, kph_f, len_abstract, len_list_f, pred_str_new2dlist,
                                                total_len)
        calculated_rewards = all_rewards.detach() * log_selected_token_total_dist
        individual_rewards = torch.sum(calculated_rewards, dim=1)
        J = torch.mean(individual_rewards)
        return J


def main(opt):
    print("agsnf efnghrrqthg")
    clip = 5
    start_time = time.time()
    train_data_loader, valid_data_loader, word2idx, idx2word, vocab = load_data_and_vocab(opt, load_train=True)
    load_data_time = time_since(start_time)
    logging.info('Time for loading the data: %.1f' % load_data_time)

    print("______________________ Data Successfully Loaded ______________")
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
    hidden_dim = opt.D_hidden_dim
    embedding_dim = opt.D_embedding_dim
    n_layers = opt.D_layers

    # hidden_dim = opt.D_hidden_dim
    # embedding_dim = opt.D_embedding_dim
    # n_layers = opt.D_layers
    #D_model = Discriminator(opt.vocab_size, embedding_dim, hidden_dim, n_layers, opt.word2idx[pykp.io.PAD_WORD],
    #                       opt.gpuid)  # gl
    bert_model = NLP_MODELS[opt.bert_model].choose()  # gl
    D_model = NetModel.from_pretrained(bert_model.pretrained_weights, num_labels=opt.bert_labels, output_hidden_states=True,hidden_dim = opt.D_hidden_dim, n_layers = opt.D_layers)  # gl
    print("The Discriminator Description is ", D_model)
    PG_optimizer = torch.optim.Adagrad(model.parameters(), opt.learning_rate_rl)
    if torch.cuda.is_available():
        D_model.load_state_dict(torch.load(opt.Discriminator_model_path))
        D_model = D_model.to(opt.gpuid)
    else:
        D_model.load_state_dict(torch.load(opt.Discriminator_model_path, map_location="cpu"))

    # D_model.load_state_dict(torch.load("Discriminator_checkpts/D_model_combined1.pth.tar"))
    total_epochs = opt.epochs
    bert_tokenizer = bert_model.tokenizer
    for epoch in range(total_epochs):

        total_batch = 0
        print("Starting with epoch:", epoch)
        for batch_i, batch in enumerate(train_data_loader):

            model.train()
            PG_optimizer.zero_grad()

            # gl: poiché init_perturb_std = final_perturb_std = 0 => perturb_std = 0
            if perturb_decay_mode == 0:  # do not decay
                perturb_std = init_perturb_std
            elif perturb_decay_mode == 1:  # exponential decay
                perturb_std = final_perturb_std + (init_perturb_std - final_perturb_std) * math.exp(
                    -1. * total_batch * perturb_decay_factor)
            elif perturb_decay_mode == 2:  # steps decay
                perturb_std = init_perturb_std * math.pow(perturb_decay_factor, math.floor((1 + total_batch) / 4000))
            # print('perturb_std = ', perturb_std)

            # print('batch = ', batch)
            avg_rewards = train_one_batch(D_model, batch, generator, opt, bert_tokenizer,perturb_std)
            # print('avg_rewards.size(): ' + str(avg_rewards.size()) + '; dtype: ' + str(avg_rewards.dtype))  # gl
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            avg_rewards.backward()
            PG_optimizer.step()

            if batch_i % 4000 == 0:
                print("Saving the file ...............----------->>>>>")
                print("The avg reward is", -avg_rewards.item())
                state_dfs = model.state_dict()
                torch.save(state_dfs, "RL_Checkpoints/Attention_Generator_" + str(epoch) + ".pth.tar")

######################################