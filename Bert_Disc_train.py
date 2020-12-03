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

import math

# ############################### IMPORT LIBRARIES ###############################################################
# import numpy as np
import pykp.io
from pykp.reward import *
from sequence_generator import SequenceGenerator

EPS = 1e-8
# import logging
# import os
# from torch.optim import Adam
import pykp
from pykp.model import Seq2SeqModel

from utils.time_log import time_since
from utils.data_loader import load_sampled_data_and_vocab
import time
import random
from BERT_Discriminator import NetModel, NetModelMC, NLP_MODELS
from transformers import AdamW, get_linear_schedule_with_warmup
from utils.report import export_train_and_valid_loss


# from evaluate import reward_function


# ####################################################################################################
# def Check_Valid_Loss(valid_data_loader,D_model,batch,generator,opt,perturb_std):

# #### TUNE HYPERPARAMETERS ##############

# #  batch_reward_stat, log_selected_token_dist = train_one_batch(batch, generator, optimizer_rl, opt, perturb_std)
#########################################################

def move_absent(kps_batch):
    """
    :param kps_batch: batch of kps
    :return: a kps batch of the same shape of input, with each list of KPs rearranged with absent at the beginning
    """
    # print(kps_batch)
    peos_kp = [pykp.io.PEOS_WORD]
    kps_batch_rearranged = []
    for idx, kps in enumerate(kps_batch):
        separator = kps.index(peos_kp)
        # print(separator)
        absent = kps[separator + 1:]
        present = kps[:separator]
        # print(kps)
        # print(absent)
        # print(present)
        new = absent + present
        # print(new)
        kps_batch_rearranged.append(new)

    # print(kps_batch_rearranged)
    # print('move_absent!')
    return kps_batch_rearranged


def remove_present(kps_batch):
    """
    :param kps_batch: batch of kps
    :return: a kps batch of the same shape of input, with present KPs removed
    """
    # print(kps_batch)
    peos_kp = [pykp.io.PEOS_WORD]
    kps_batch_only_absent = []
    for idx, kps in enumerate(kps_batch):
        separator = kps.index(peos_kp)
        # print(separator)
        absent = kps[separator + 1:]
        # present = kps[:separator]
        # print(kps)
        # print(absent)
        # print(present)
        kps_batch_only_absent.append(absent)

    # print('kps_batch_only_absent : ', kps_batch_only_absent)
    return kps_batch_only_absent


def build_kps_idx_list(kps, bert_tokenizer, separate_present_absent):
    """
    evaluate the BERT tokenization for list of KPs
    :param kps: list of (indexes of each word) words of KPs, for each src in batch
    :param bert_tokenizer: BERT tokenizer
    :param separate_present_absent: if True present and absent KPs will be separated by a '.' token
    :return: list of BERT indexes of each word of KPs, for each src in batch
    """
    # str_kps_list = [[[opt.idx2word[w] for w in kp] for kp in b] for b in kps]
    # print()  # gl: debug
    str_list = []
    idx_list = []
    # for b in str_kps_list:
    for b in kps:
        # print(b)  # gl: debug
        # str_kps = '[SEP]'  # original
        str_kps = '[SEP] List of Keyphrases is: '
        # str_kps = 'List of Keyphrases is: '
        for kp in b:
            for w in kp:
                if w == pykp.io.UNK_WORD:
                    w = '[UNK]'
                    # w = pykp.io.UNK_WORD  # gl: prova, semmai ripristinare quello sopra
                elif w == pykp.io.PEOS_WORD:
                    # separate_present_absent = False  # gl: only for experiment8; to remove
                    if separate_present_absent:
                        w = '.'
                    else:
                        w = ''
                str_kps += w + ' '
            if str_kps[-2] != '.':  # gl: if w=pykp.io.PEOS_WORD then no need for ;
                str_kps += ';'  # original
                # str_kps += ','
        str_kps += '[SEP]'
        str_kps = str_kps.replace('[SEP] ;', '[SEP] ')  # original
        str_kps = str_kps.replace(' ; ;', ' ;')  # original
        str_kps = str_kps.replace(' ;. ', ' . ')  # original
        str_kps = str_kps.replace(' ;[SEP]', ' .[SEP]')  # original
        str_kps = str_kps.replace(':  ;', ': ')
        # str_kps = str_kps.replace('[SEP] ,', '[SEP] ')
        # str_kps = str_kps.replace(' , ,', ' ,')
        # str_kps = str_kps.replace(' ,. ', ' . ')
        # str_kps = str_kps.replace(' ,[SEP]', ' .[SEP]')
        # print(str_kps)  # gl: debug
        bert_str_kps = bert_tokenizer.tokenize(str_kps)
        str_list.append(bert_str_kps)
        idx_list.append(bert_tokenizer.convert_tokens_to_ids(bert_str_kps))  # rivedere bene
    # print(str_list)  # gl: debug
    # print(idx_list)  # gl: debug

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
                w = '[UNK]'
                # w = pykp.io.UNK_WORD  # gl: prova, semmai ripristinare quello sopra
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
    # print(str_list)  # gl: debug
    # print(idx_list)  # gl: debug

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
    # print([len(d) for d in src])
    # print([len(k) for k in kps])
    training_batch = []
    mask_batch = []
    segment_batch = []
    label_batch = [label] * len(kps)
    pad_list = [bert_tokenizer.pad_token_id] * opt.bert_max_length
    for i in range(len(kps)):
        t = src[i][:opt.bert_max_length - len(kps[i])] + kps[i] + pad_list
        t = t[:opt.bert_max_length]
        m = [1 if k != bert_tokenizer.pad_token_id else 0 for k in t]
        s = [1] * (min(opt.bert_max_length - len(kps[i]), len(src[i])) + 1) + [0] * opt.bert_max_length
        s = s[:opt.bert_max_length]
        training_batch.append(t)
        mask_batch.append(m)
        segment_batch.append(s)
    # print(training_batch)
    # print(mask_batch)
    # print(segment_batch)
    # print(label_batch)

    return training_batch, mask_batch, segment_batch, label_batch


def shuffle_input_samples(batch_size, pred_train_batch, target_train_batch,
                          pred_mask_batch, target_mask_batch,
                          pred_segment_batch, target_segment_batch):
    """
    Random shuffles samples to use as input in BertForMultipleChoice model
    :param: batch_size, int, the size of the current batch of samples; could be less than opt.batch_size
    :param: pred_train_batch, list of ids of src and predicted KPs
    :param: target_train_batch, list of ids of src and target KPs
    :param: pred_mask_batch, list of [0, 1] where 1 denotes non-zero tokens (predicted samples)
    :param: target_mask_batch, list of [0, 1] where 1 denotes non-zero tokens (target samples)
    :param: pred_segment_batch, list of [0, 1] to split segment A (src) from segment B (KPs) (predicted samples)
    :param: target_segment_batch, list of [0, 1] to split segment A (src) from segment B (KPs) (target samples)
    :return: same lists as input, but samples are randomly shuffled
    """

    input_ids = []
    input_mask = []
    input_segment = []
    labels = []
    # for i in range(opt.batch_size):  # batch_size
    for i in range(batch_size):  # batch_size
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

    return input_ids, input_mask, input_segment, labels


def train_one_batch(D_model, one2many_batch, generator, opt, perturb_std, bert_tokenizer, bert_model_name):
    src, src_lens, src_mask, src_oov, oov_lists, src_str_list, trg_str_2dlist, trg, trg_oov, trg_lens, trg_mask, \
    _, title, title_oov, title_lens, title_mask = one2many_batch
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

    with torch.no_grad():
        sample_list, log_selected_token_dist, output_mask, pred_eos_idx_mask, entropy, location_of_eos_for_each_batch, location_of_peos_for_each_batch = \
            generator.sample(
                src, src_lens, src_oov, src_mask, oov_lists, opt.max_length, greedy=False, one2many=one2many,
                one2many_mode=one2many_mode, num_predictions=num_predictions, perturb_std=perturb_std,
                entropy_regularize=entropy_regularize, title=title, title_lens=title_lens, title_mask=title_mask)
    # print(sample_list[0]['prediction'])
    pred_str_2dlist = sample_list_to_str_2dlist(sample_list, oov_lists, opt.idx2word, opt.vocab_size, eos_idx,
                                                delimiter_word, opt.word2idx[pykp.io.UNK_WORD], opt.replace_unk,
                                                src_str_list, opt.separate_present_absent, pykp.io.PEOS_WORD)

    # gl: 1. verificare se nelle 2 liste di KPs ce ne sono uguali ed eventualmente metterle all'inizio nello stesso ordine (così il Discriminator capisce che sono simili)
    # gl: alla fine G dovrebbe creare samples uguali a quelli veri ma non necessariamente nello steso ordine;
    # gl: dandogli lo steso ordine lo aiuto a emettere un output = 0,5 perché non ci capisce più nulla (?)
    if opt.absent_first:
        trg_str_2dlist = move_absent(trg_str_2dlist)  # gl: change the order of present/absent kps in target file
    if opt.only_absent:
        trg_str_2dlist = remove_present(trg_str_2dlist)  # gl: change the order of present/absent kps in target file
    # gl: 2. Bert tokens indexes
    # print('pred_str_2dlist : ', pred_str_2dlist)
    pred_idx_list = build_kps_idx_list(pred_str_2dlist, bert_tokenizer, opt.separate_present_absent)
    # print('pred_idx_list   : ', pred_idx_list)
    # print('trg_str_2dlist  : ', trg_str_2dlist)
    target_idx_list = build_kps_idx_list(trg_str_2dlist, bert_tokenizer, opt.separate_present_absent)
    # print('target_idx_list : ', target_idx_list)
    # print('src_str_list    : ', src_str_list)
    src_idx_list = build_src_idx_list(src_str_list, bert_tokenizer)
    # print('src_idx_list    : ', src_idx_list)
    # print()

    # gl: 3. creare il batch di addestramento concatenando src sia con le fake che con le true KPs, limitando la lunghezza a 512 tokens e paddando
    # nota: sia per le true che per le fake KPS, src dovrebbe essere identico e quindi troncato alla stessa lunghezza
    pred_train_batch, pred_mask_batch, pred_segment_batch, pred_label_batch = \
        build_training_batch(src_idx_list, pred_idx_list, bert_tokenizer, opt, label=0)
    target_train_batch, target_mask_batch, target_segment_batch, target_label_batch = \
        build_training_batch(src_idx_list, target_idx_list, bert_tokenizer, opt, label=1)

    if bert_model_name == 'BertForMultipleChoice':

        input_ids, input_mask, input_segment, labels = \
            shuffle_input_samples(batch_size, pred_train_batch, target_train_batch,
                                  pred_mask_batch, target_mask_batch,
                                  pred_segment_batch, target_segment_batch)
        input_ids = torch.tensor(input_ids, dtype=torch.long).to(devices)
        labels = torch.tensor(labels, dtype=torch.long).to(devices)
        input_mask = torch.tensor(input_mask, dtype=torch.long).to(devices)
        input_segment = torch.tensor(input_segment, dtype=torch.long).to(devices)

        output = D_model(input_ids,
                         attention_mask=input_mask,
                         token_type_ids=input_segment,
                         labels=labels)

        avg_batch_loss = output[0]
        positives = sum(torch.argmax(output[1], dim=1) == labels)

        sum_real = 0
        sum_fake = 0

        for i, out in enumerate(output[1]):
            if labels[i].item() == 0:
                sum_real += out[0]
                sum_fake += out[1]
            elif labels[i].item() == 1:
                sum_real += out[1]
                sum_fake += out[0]

        avg_real = sum_real / (i + 1)
        avg_fake = sum_fake / (i + 1)

    elif bert_model_name == 'BertForSequenceClassification':

        # gl: 4. transform to torch.tensor
        pred_input_ids = torch.tensor(pred_train_batch, dtype=torch.long).to(devices)
        target_input_ids = torch.tensor(target_train_batch, dtype=torch.long).to(devices)
        pred_input_mask = torch.tensor(pred_mask_batch, dtype=torch.float).to(devices)  # gl: was dtype=torch.float32
        target_input_mask = torch.tensor(target_mask_batch, dtype=torch.float).to(
            devices)  # gl: was dtype=torch.float32
        pred_input_segment = torch.tensor(pred_segment_batch, dtype=torch.long).to(devices)
        target_input_segment = torch.tensor(target_segment_batch, dtype=torch.long).to(devices)
        if opt.bert_labels == 1:
            target_input_labels = torch.tensor(target_label_batch, dtype=torch.float).to(devices)
            pred_input_labels = torch.tensor(pred_label_batch, dtype=torch.float).to(devices)
        else:
            target_input_labels = torch.tensor(target_label_batch, dtype=torch.long).to(devices)
            pred_input_labels = torch.tensor(pred_label_batch, dtype=torch.long).to(devices)

        # gl: 5. forward pass
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

        # positives = torch.tensor([0.])
        # if opt.bert_labels == 1:  # regression
        #     avg_real = torch.mean(target_output[1])
        #     avg_fake = torch.mean(pred_output[1])
        #
        #     for i, (target, prediction) in enumerate(zip(target_output[1], pred_output[1])):
        #         if target > 0.5 > prediction:
        #             positives += 1
        # else:  # 2 classes classification
        #     avg_real = sum(torch.argmax(target_output[1], dim=1) == target_input_labels) / target_input_labels.shape[0]
        #     avg_fake = sum(torch.argmax(pred_output[1], dim=1) == pred_input_labels) / pred_input_labels.shape[0]
        #
        #     for i, (target, prediction) in enumerate(zip(target_output[1], pred_output[1])):
        #         if (target[1] > target[0]) and (prediction[0] > prediction[1]):
        #             positives += 1

        positives = 0
        p = pred_output[1].detach().clone()
        p = p.cpu().numpy()
        t = target_output[1].detach().clone()
        t = t.cpu().numpy()
        # print('p: ', p)
        if opt.bert_labels == 1:  # regression
            avg_real = np.mean(t)
            avg_fake = np.mean(p)

            for i, (target, prediction) in enumerate(zip(t, p)):
                if target > 0.5 > prediction:
                    positives += 1
        else:  # 2 classes classification
            avg_real = sum(np.argmax(t, dim=1) == target_input_labels) / target_input_labels.shape[0]
            avg_fake = sum(np.argmax(p, dim=1) == pred_input_labels) / pred_input_labels.shape[0]

            for i, (target, prediction) in enumerate(zip(t, p)):
                if (target[1] > target[0]) and (prediction[0] > prediction[1]):
                    positives += 1

    return avg_batch_loss, avg_real, avg_fake, positives


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main(opt):
    clip = 5  # gl: was 5
    start_time = time.time()
    # train_data_loader, valid_data_loader, word2idx, idx2word, vocab = load_data_and_vocab(opt, load_train=True)  # gl: old
    train_data_loader, valid_data_loader, word2idx, idx2word, vocab = load_sampled_data_and_vocab(opt, load_train=True,
                                                                                                  sampled=True)
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

    train_loss_list = []
    valid_loss_list = []

    bert_model = NLP_MODELS[opt.bert_model].choose()  # gl
    bert_model_name = bert_model.model.__class__.__name__
    print(bert_model_name)

    if bert_model_name == 'BertForSequenceClassification' or bert_model_name == 'AutoModelForSequenceClassification':
        D_model = NetModel.from_pretrained(bert_model.pretrained_weights,
                                           num_labels=opt.bert_labels,
                                           output_hidden_states=True,
                                           output_attentions=False,
                                           hidden_dropout_prob=0.1,  # gl: default=0.1
                                           )
    elif bert_model_name == 'BertForMultipleChoice':
        D_model = NetModelMC.from_pretrained(bert_model.pretrained_weights,
                                             output_hidden_states=True,
                                             output_attentions=False,
                                             hidden_dropout_prob=0.1,  # gl: default=0.1
                                             )

    bert_tokenizer = bert_model.tokenizer

    if torch.cuda.is_available():
        D_model = D_model.cuda()  # gl
    print("The Discriminator Description is ", D_model)
    print("Number of trainable parameters is ", count_parameters(D_model))
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

    param_optimizer = list(D_model.named_parameters())
    # optimizer_trainable_parameters = [p for p in param_optimizer if p.requires_grad is True]
    # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']  # gl: il weight_decay riduce l'overfitting
    no_decay = ['bias', 'LayerNorm.weight']  # gl: il weight_decay riduce l'overfitting
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    accumulation_steps = opt.bert_accumulated_steps  # gl:
    rescaled_lr = opt.bert_learning_rate * math.sqrt(accumulation_steps)
    D_optimizer = AdamW(optimizer_grouped_parameters, lr=rescaled_lr, eps=1e-8)
    total_steps = len(train_data_loader) * opt.epochs
    total_steps = int(
        total_steps / accumulation_steps)  # gl: if accumulation steps, then optimizer step is every batch_size * accumulation_step
    warmup_steps = 0  # gl: warmup is 10% of total steps: int(total_steps*0.1) ; put 0 for no warmup
    scheduler = get_linear_schedule_with_warmup(optimizer=D_optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)

    print()
    print("Training samples %d; Validation samples: %d" % (
        len(train_data_loader.sampler), len(valid_data_loader.sampler)))
    print("Beginning with training Discriminator")
    print("########################################################################################################")
    total_epochs = opt.epochs
    best_valid_loss = 1000
    num_stop_increasing = 0
    early_stop = False
    generator.model.eval()
    D_model.zero_grad()  # gl: insieme all'accumulation_steps
    D_optimizer.zero_grad()
    cum_train_loss = 0
    total_batch = -1
    for epoch in range(total_epochs):
        print("Starting with epoch:", epoch)
        for batch_i, batch in enumerate(train_data_loader):
            total_batch += 1
            D_model.train()
            # D_optimizer.zero_grad()  # gl: old

            if perturb_decay_mode == 0:  # do not decay
                perturb_std = init_perturb_std
            elif perturb_decay_mode == 1:  # exponential decay
                perturb_std = final_perturb_std + (init_perturb_std - final_perturb_std) * math.exp(
                    -1. * total_batch * perturb_decay_factor)
            elif perturb_decay_mode == 2:  # steps decay
                perturb_std = init_perturb_std * math.pow(perturb_decay_factor, math.floor((1 + total_batch) / 4000))

            avg_batch_loss, _, _, _ = train_one_batch(D_model, batch, generator, opt, perturb_std, bert_tokenizer,
                                                      bert_model_name)

            # print('1. ********************************************')
            # for name, param in D_model.named_parameters():
            #     if param.grad is not None:
            #         print(name, param.grad.sum())
            #     else:
            #         print(name, param.grad)

            # accumulo la loss per il plot
            cum_train_loss += avg_batch_loss.item()

            avg_batch_loss = avg_batch_loss / accumulation_steps
            avg_batch_loss.backward()

            if (batch_i + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(D_model.parameters(), clip)

                # y = avg_batch_loss.detach().clone()
                train_loss_list.append(cum_train_loss / accumulation_steps)
                cum_train_loss = 0
                print('train batch: ' + str(batch_i) + '; loss: ' + str(avg_batch_loss.item()))

                # print('2. ###################################################################')
                # for name, param in D_model.named_parameters():
                #     if param.grad is not None:
                #         print(name, param.grad.sum())
                #     else:
                #         print(name, param.grad)

                D_optimizer.step()
                scheduler.step()
                D_model.zero_grad()
                D_optimizer.zero_grad()

            # old
            # avg_batch_loss.backward()
            # D_optimizer.step()
            # scheduler.step()

            if (total_batch + 1) % (
                    accumulation_steps * 32) == 0 and epoch > 0:  # gl: entra solo quando ha svuotato i gradienti
                # if (total_batch + 1) % (accumulation_steps * 32) == 0:  # gl: entra solo quando ha svuotato i gradienti
                valid_start_time = time.time()
                D_model.eval()
                with torch.no_grad():
                    print()
                    print('**********************************************************************')
                    print('validation:')
                    total = 0
                    valid_loss_total, valid_real_total, valid_fake_total, valid_positives_total = 0, 0, 0, 0
                    for batch_j, valid_batch in enumerate(valid_data_loader):
                        total += 1
                        valid_loss, valid_real, valid_fake, valid_positives = \
                            train_one_batch(D_model, valid_batch, generator, opt, perturb_std, bert_tokenizer,
                                            bert_model_name)
                        valid_loss_total += valid_loss.item()
                        valid_real_total += valid_real
                        valid_fake_total += valid_fake
                        valid_positives_total += valid_positives
                        # D_optimizer.zero_grad()

                    valid_loss_list.append(valid_loss_total / len(valid_data_loader))

                    print("Time elapsed for validation is ", time_since(valid_start_time))
                    print("Average loss is ", valid_loss_total / total)
                    print("Average real score is ", valid_real_total / total)
                    print("Average fake score is ", valid_fake_total / total)
                    print("Currently accuracy is ", valid_positives_total / len(valid_data_loader.sampler))

                    if best_valid_loss > valid_loss_total / total:
                        print("Loss Decreases so saving the file ...............----------->>>>>")
                        state_dfs = D_model.state_dict()
                        torch.save(state_dfs, "Discriminator_checkpts/Bert_Discriminator.epoch=" + str(epoch)
                                   + ".batch=" + str(batch_i) + ".total_batch=" + str(total_batch)
                                   + ".pth.tar")
                        best_valid_loss = valid_loss_total / total
                        num_stop_increasing = 0
                    else:  # gl
                        print("Loss doesn't decrease so go on without saving the file")
                        num_stop_increasing += 1
                        if num_stop_increasing >= opt.bert_early_stop_tolerance and epoch > 2:
                            print(
                                'Loss did not decrease for %d check points, early stop training' % num_stop_increasing)
                            early_stop = True
                            break

                    print('**********************************************************************')
                    print()

            if early_stop:
                break

        if early_stop:
            break

    # train_curve_path = opt.exp_path + '/discriminator_train_loss'
    # export_train_and_valid_loss(train_loss_list, [0.]*len(train_loss_list), [0.]*len(train_loss_list),  [0.]*len(train_loss_list), 1, train_curve_path)
    # valid_curve_path = opt.exp_path + '/discriminator_valid_loss'
    # export_train_and_valid_loss([0.]*len(valid_loss_list), valid_loss_list, [0.]*len(valid_loss_list),  [0.]*len(valid_loss_list), 1, valid_curve_path)
    train_curve_path = opt.exp_path + '/discriminator_loss'
    export_train_and_valid_loss(train_loss_list, valid_loss_list, [0.] * len(train_loss_list),
                                [0.] * len(valid_loss_list), 1, train_curve_path)

    print()
    print('Discriminator training time: ', time_since(start_time))
    print("End of the Discriminator training")

######################################
