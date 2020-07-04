#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 15:10:45 2019
@author: r17935avinash
"""

import math
import sys

import torch.nn as nn

# ############################### IMPORT LIBRARIES ###############################################################
import pykp.io
from evaluate import evaluate_valid_reward, reward_function
from pykp.reward import *
from sequence_generator import SequenceGenerator
from utils.statistics import RewardStatistics

EPS = 1e-8
import logging
import pykp
from pykp.model import Seq2SeqModel

from utils.time_log import time_since
from utils.data_loader import load_data_and_vocab, load_sampled_data_and_vocab
from utils.report import export_train_and_valid_reward

import time
from BERT_Discriminator import NetModel, NetModelMC, NLP_MODELS
from Bert_Disc_train import build_kps_idx_list, build_src_idx_list, build_training_batch, shuffle_input_samples

#####################################################################################################

torch.autograd.set_detect_anomaly(True)


# #  batch_reward_stat, log_selected_token_dist = train_one_batch(batch, generator, optimizer_rl, opt, perturb_std)
#########################################################
def train_one_batch(D_model, one2many_batch, generator, opt, perturb_std, bert_tokenizer, optimizer, bert_model_name):
    src, src_lens, src_mask, src_oov, oov_lists, src_str_list, trg_str_2dlist, trg, trg_oov, trg_lens, trg_mask, _, \
    title, title_oov, title_lens, title_mask = one2many_batch
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
    regularization_factor = opt.regularization_factor  # #DNT
    if regularization_type == 2:
        entropy_regularize = True
    else:
        entropy_regularize = False

    start_time = time.time()
    sample_list, log_selected_token_dist, output_mask, pred_eos_idx_mask, entropy, location_of_eos_for_each_batch, location_of_peos_for_each_batch = generator.sample(
        src, src_lens, src_oov, src_mask, oov_lists, opt.max_length, greedy=False, one2many=one2many,
        one2many_mode=one2many_mode, num_predictions=num_predictions, perturb_std=perturb_std,
        entropy_regularize=entropy_regularize, title=title, title_lens=title_lens, title_mask=title_mask)
    pred_str_2dlist = sample_list_to_str_2dlist(sample_list, oov_lists, opt.idx2word, opt.vocab_size, eos_idx,
                                                delimiter_word, opt.word2idx[pykp.io.UNK_WORD], opt.replace_unk,
                                                src_str_list, opt.separate_present_absent, pykp.io.PEOS_WORD)
    # target_str_2dlist = convert_list_to_kphs(trg)
    sample_time = time_since(start_time)

    # gl: new; log_selected_token_dist is related to eq. 1 in RL project paper
    max_pred_seq_len = log_selected_token_dist.size(1)
    # print('perturb_std      :', perturb_std)  # gl: debug
    # print('opt.max_length   :', opt.max_length)  # gl: debug
    # print('num_predictions  :', num_predictions)  # gl: debug
    print('max_pred_seq_len :', max_pred_seq_len)  # gl: debug

    if entropy_regularize:
        entropy_array = entropy.data.cpu().numpy()
    else:
        entropy_array = None

    if opt.perturb_baseline:
        baseline_perturb_std = perturb_std
    else:
        baseline_perturb_std = 0

    # if use self critical as baseline, greedily decode a sequence from the model
    if baseline == 'self':
        generator.model.eval()
        with torch.no_grad():
            start_time = time.time()
            greedy_sample_list, _, _, greedy_eos_idx_mask, _, _, _ = generator.sample(src, src_lens, src_oov, src_mask,
                                                                                      oov_lists, opt.max_length,
                                                                                      greedy=True, one2many=one2many,
                                                                                      one2many_mode=one2many_mode,
                                                                                      num_predictions=num_predictions,
                                                                                      perturb_std=baseline_perturb_std,
                                                                                      title=title,
                                                                                      title_lens=title_lens,
                                                                                      title_mask=title_mask)
            greedy_str_2dlist = sample_list_to_str_2dlist(greedy_sample_list, oov_lists, opt.idx2word, opt.vocab_size,
                                                          eos_idx,
                                                          delimiter_word, opt.word2idx[pykp.io.UNK_WORD],
                                                          opt.replace_unk,
                                                          src_str_list, opt.separate_present_absent, pykp.io.PEOS_WORD)
            # print(greedy_str_2dlist)  # gl: debug
        generator.model.train()

    if torch.cuda.is_available():
        devices = opt.gpuid
    else:
        devices = "cpu"

    # gl: new part, Bert Discriminator
    # print(src_str_list)  # gl: debug
    D_model.eval()
    with torch.no_grad():

        if bert_model_name == 'BertForSequenceClassification':

            pred_idx_list = build_kps_idx_list(pred_str_2dlist, bert_tokenizer, opt.separate_present_absent)
            greedy_idx_list = build_kps_idx_list(greedy_str_2dlist, bert_tokenizer, opt.separate_present_absent)
            src_idx_list = build_src_idx_list(src_str_list, bert_tokenizer)

            pred_train_batch, pred_mask_batch, pred_segment_batch, _ = \
                build_training_batch(src_idx_list, pred_idx_list, bert_tokenizer, opt, label=0)
            greedy_train_batch, greedy_mask_batch, greedy_segment_batch, _ = \
                build_training_batch(src_idx_list, greedy_idx_list, bert_tokenizer, opt, label=0)

            pred_train_batch = torch.tensor(pred_train_batch, dtype=torch.long).to(devices)
            pred_mask_batch = torch.tensor(pred_mask_batch, dtype=torch.long).to(devices)
            pred_segment_batch = torch.tensor(pred_segment_batch, dtype=torch.long).to(devices)
            # # pred_rewards = np.zeros(batch_size)
            # output_pre = D_model(pred_train_batch,
            #                      attention_mask=pred_mask_batch,
            #                      token_type_ids=pred_segment_batch,
            #                      )
            # # idx = 0
            # # for val_p in output_pre[0]:
            # #     pred_rewards[idx] = val_p.item()
            # #     idx += 1
            #
            # pred_rewards = reward_function(output_pre[0], batch_size, bert_model_name)

            greedy_train_batch = torch.tensor(greedy_train_batch, dtype=torch.long).to(devices)
            greedy_mask_batch = torch.tensor(greedy_mask_batch, dtype=torch.long).to(devices)
            greedy_segment_batch = torch.tensor(greedy_segment_batch, dtype=torch.long).to(devices)
            # # baseline_rewards = np.zeros(batch_size)
            # output_bas = D_model(greedy_train_batch,
            #                      attention_mask=greedy_mask_batch,
            #                      token_type_ids=greedy_segment_batch,
            #                      )
            # # idx = 0
            # # for val_b in output_bas[0]:
            # #     baseline_rewards[idx] = val_b.item()
            # #     idx += 1
            #
            # baseline_rewards = reward_function(output_bas[0], batch_size, bert_model_name)

        elif bert_model_name == 'BertForMultipleChoice':

            pred_idx_list = build_kps_idx_list(pred_str_2dlist, bert_tokenizer, opt.separate_present_absent)
            greedy_idx_list = build_kps_idx_list(greedy_str_2dlist, bert_tokenizer, opt.separate_present_absent)
            trg_idx_list = build_kps_idx_list(trg_str_2dlist, bert_tokenizer, opt.separate_present_absent)
            src_idx_list = build_src_idx_list(src_str_list, bert_tokenizer)

            pred_train_batch, pred_mask_batch, pred_segment_batch, _ = \
                build_training_batch(src_idx_list, pred_idx_list, bert_tokenizer, opt, label=0)
            greedy_train_batch, greedy_mask_batch, greedy_segment_batch, _ = \
                build_training_batch(src_idx_list, greedy_idx_list, bert_tokenizer, opt, label=0)
            trg_train_batch, trg_mask_batch, trg_segment_batch, _ = \
                build_training_batch(src_idx_list, trg_idx_list, bert_tokenizer, opt, label=1)

            pred_train_batch, pred_mask_batch, pred_segment_batch, pred_labels = \
                shuffle_input_samples(batch_size, pred_train_batch, trg_train_batch,
                                      pred_mask_batch, trg_mask_batch,
                                      pred_segment_batch, trg_segment_batch)

            greedy_train_batch, greedy_mask_batch, greedy_segment_batch, greedy_labels = \
                shuffle_input_samples(batch_size, greedy_train_batch, trg_train_batch,
                                      greedy_mask_batch, trg_mask_batch,
                                      greedy_segment_batch, trg_segment_batch)

            pred_train_batch = torch.tensor(pred_train_batch, dtype=torch.long).to(devices)
            # labels = torch.tensor(labels, dtype=torch.long).to(devices)
            pred_mask_batch = torch.tensor(pred_mask_batch, dtype=torch.long).to(devices)
            pred_segment_batch = torch.tensor(pred_segment_batch, dtype=torch.long).to(devices)

            greedy_train_batch = torch.tensor(greedy_train_batch, dtype=torch.long).to(devices)
            # labels = torch.tensor(labels, dtype=torch.long).to(devices)
            greedy_mask_batch = torch.tensor(greedy_mask_batch, dtype=torch.long).to(devices)
            greedy_segment_batch = torch.tensor(greedy_segment_batch, dtype=torch.long).to(devices)

        output_pre = D_model(pred_train_batch,
                             attention_mask=pred_mask_batch,
                             token_type_ids=pred_segment_batch,
                             )
        pred_rewards = reward_function(output_pre[0], batch_size, bert_model_name, pred_labels)
        # pred_rewards = reward_function(output_pre[0], batch_size, bert_model_name)

        output_bas = D_model(greedy_train_batch,
                             attention_mask=greedy_mask_batch,
                             token_type_ids=greedy_segment_batch,
                             )
        baseline_rewards = reward_function(output_bas[0], batch_size, bert_model_name, greedy_labels)
        # baseline_rewards = reward_function(output_bas[0], batch_size, bert_model_name)

    # for doc, target, prediction, reward in zip(src_str_list, trg_str_2dlist, pred_str_2dlist, pred_rewards):  # debug
    #     print('doc        : ', doc)
    #     print('target     : ', target)
    #     print('prediction : ', prediction)
    #     print('reward     : ', reward)
    #     print()

    cumulative_reward_sum = pred_rewards.sum(0)
    # cumulative_bas_reward_sum = baseline_rewards.sum(0)
    batch_rewards = pred_rewards - baseline_rewards
    # torch.save(batch_rewards, 'prova/batch_rewards.pt')  # gl saving tensors
    # print('pred_labels      : ', pred_labels)
    # print('greedy_labels    : ', greedy_labels)
    # print('pred_rewards     : ', pred_rewards)
    # print('baseline_rewards : ', baseline_rewards)
    # print('batch_rewards    : ', batch_rewards)
    q_value_estimate_array = np.tile(batch_rewards.reshape([-1, 1]), [1, max_pred_seq_len])  # [batch, max_pred_seq_len]

    q_value_estimate = torch.from_numpy(q_value_estimate_array).type(torch.FloatTensor).to(src.device)
    q_value_estimate.requires_grad_(True)
    q_estimate_compute_time = time_since(start_time)

    # compute the policy gradient objective
    # print('log_selected_token_dist  :' + str(log_selected_token_dist))
    # print('output_mask              :' + str(output_mask))
    # print('q_value_estimate         :' + str(q_value_estimate))
    # print('q_value_estimate_array   :', q_value_estimate_array)  # gl: debug
    print('cumulative_reward_sum    :' + str(cumulative_reward_sum))
    # print('cumulative_bas_reward_sum:' + str(cumulative_bas_reward_sum))
    pg_loss = compute_pg_loss(log_selected_token_dist, output_mask, q_value_estimate)
    print('pg_loss                  :' + str(pg_loss.item()))

    # back propagation to compute the gradient
    start_time = time.time()
    pg_loss.backward()
    backward_time = time_since(start_time)

    if opt.max_grad_norm > 0:  # gl: 1
        grad_norm_before_clipping = nn.utils.clip_grad_norm_(generator.model.parameters(), opt.max_grad_norm)

    # take a step of gradient descent
    optimizer.step()

    stat = RewardStatistics(cumulative_reward_sum, pg_loss.item(), batch_size, sample_time, q_estimate_compute_time,
                            backward_time)
    # (final_reward=0.0, pg_loss=0.0, n_batch=0, sample_time=0, q_estimate_compute_time=0, backward_time=0)
    # reward=0.0, pg_loss=0.0, n_batch=0, sample_time=0, q_estimate_compute_time=0, backward_time=0

    return stat, log_selected_token_dist.detach()
    # return pg_loss


def main(opt):
    total_batch = -1
    early_stop_flag = False
    num_stop_increasing = 0
    best_valid_reward = float('-inf')

    # print("agsnf efnghrrqthg")
    # clip = 5
    report_train_reward_statistics = RewardStatistics()
    total_train_reward_statistics = RewardStatistics()
    report_train_reward = []
    report_valid_reward = []
    start_time = time.time()
    # train_data_loader, valid_data_loader, word2idx, idx2word, vocab = load_data_and_vocab(opt, load_train=True)
    train_data_loader, valid_data_loader, word2idx, idx2word, vocab = load_sampled_data_and_vocab(opt, load_train=True, sampled=True)
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

    bert_model = NLP_MODELS[opt.bert_model].choose()  # gl
    bert_model_name = bert_model.model.__class__.__name__
    print(bert_model_name)

    if bert_model_name == 'BertForSequenceClassification':
        D_model = NetModel.from_pretrained(bert_model.pretrained_weights,
                                           num_labels=opt.bert_labels,
                                           output_hidden_states=True,
                                           output_attentions=False,
                                           hidden_dropout_prob=0.1,
                                           )
    elif bert_model_name == 'BertForMultipleChoice':
        D_model = NetModelMC.from_pretrained(bert_model.pretrained_weights,
                                             output_hidden_states=True,
                                             output_attentions=False,
                                             hidden_dropout_prob=0.1,
                                             )

    bert_tokenizer = bert_model.tokenizer

    print("The Generator Description is ", model)

    # PG_optimizer = torch.optim.Adagrad(model.parameters(), opt.learning_rate_rl)  # gl: GAN code
    PG_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),
                                    lr=opt.learning_rate_rl)  # gl: RL code
    if torch.cuda.is_available():
        D_model.load_state_dict(torch.load(opt.Discriminator_model_path))
        D_model = D_model.to(opt.gpuid)
    else:
        D_model.load_state_dict(torch.load(opt.Discriminator_model_path, map_location="cpu"))

    # D_model.load_state_dict(torch.load("Discriminator_checkpts/D_model_combined1.pth.tar"))

    print()
    print("Training samples %d; Validation samples: %d" % (len(train_data_loader.sampler), len(valid_data_loader.sampler)))
    print("Beginning with training Generator")
    print("########################################################################################################")
    total_epochs = opt.epochs
    model.train()

    for epoch in range(opt.start_epoch, opt.epochs + 1):  # gl: 1, 20
        if early_stop_flag:  # gl: False
            break

        # total_batch = 0
        print("Starting with epoch:", epoch)
        for batch_i, batch in enumerate(train_data_loader):
            total_batch += 1

            # model.train()
            PG_optimizer.zero_grad()

            if perturb_decay_mode == 0:  # do not decay
                perturb_std = init_perturb_std
            elif perturb_decay_mode == 1:  # exponential decay
                perturb_std = final_perturb_std + (init_perturb_std - final_perturb_std) * math.exp(
                    -1. * total_batch * perturb_decay_factor)
            elif perturb_decay_mode == 2:  # steps decay
                perturb_std = init_perturb_std * math.pow(perturb_decay_factor, math.floor((1 + total_batch) / 4000))

            batch_reward_stat, pg_loss = train_one_batch(D_model, batch, generator, opt, perturb_std, bert_tokenizer,
                                                         PG_optimizer, bert_model_name)
            # pg_loss = train_one_batch(D_model, batch, generator, opt, perturb_std, bert_tokenizer, bert_model_name)
            report_train_reward_statistics.update(batch_reward_stat)
            total_train_reward_statistics.update(batch_reward_stat)

            # if opt.max_grad_norm > 0:
            #     torch.nn.utils.clip_grad_norm_(model.parameters(), opt.max_grad_norm)
            # pg_loss.backward()
            # PG_optimizer.step()

            # if batch_i % 4000 == 0:
            #     print("Saving the file ...............----------->>>>>")
            #     print("The avg reward is", -avg_rewards.item())
            #     state_dfs = model.state_dict()
            #     torch.save(state_dfs, "RL_Checkpoints/Attention_Generator_" + str(epoch) + ".pth.tar")

            # model.eval()

            if total_batch % 20 == 0:  # gl: was 4000 with full dataset and batch_size=32 (=128k samples)
                print("Epoch %d; batch: %d; total batch: %d" % (epoch, batch_i, total_batch))
                sys.stdout.flush()

            if epoch >= opt.start_checkpoint_at:
                if (opt.checkpoint_interval == -1 and batch_i == len(train_data_loader) - 1) or \
                        (
                                opt.checkpoint_interval > -1 and total_batch > 1 and total_batch % opt.checkpoint_interval == 0):

                    valid_reward_stat = evaluate_valid_reward(valid_data_loader, generator, opt, D_model,
                                                              bert_tokenizer, bert_model_name)
                    model.train()
                    current_valid_reward = valid_reward_stat.reward()
                    print("Enter check point!")
                    sys.stdout.flush()

                    current_train_reward = report_train_reward_statistics.reward()
                    current_train_pg_loss = report_train_reward_statistics.loss()

                    if current_valid_reward > best_valid_reward:
                        print("Valid reward increases")
                        sys.stdout.flush()
                        best_valid_reward = current_valid_reward
                        num_stop_increasing = 0

                        # check_pt_model_path = os.path.join(opt.model_path, '%s.epoch=%d.batch=%d.total_batch=%d' % (
                        #     opt.exp, epoch, batch_i, total_batch) + '.model')
                        model_folder = 'RL_Checkpoints'
                        check_pt_model_path = os.path.join(model_folder, '%s.epoch=%d.batch=%d.total_batch=%d' % (
                            opt.exp, epoch, batch_i, total_batch) + '.model')
                        torch.save(  # save model parameters
                            model.state_dict(),
                            open(check_pt_model_path, 'wb')
                        )
                        logging.info('Saving checkpoint to %s' % check_pt_model_path)
                    else:
                        print("Valid reward does not increase")
                        sys.stdout.flush()
                        num_stop_increasing += 1
                        # decay the learning rate by the factor specified by opt.learning_rate_decay
                        if opt.learning_rate_decay_rl:
                            for i, param_group in enumerate(PG_optimizer.param_groups):
                                old_lr = float(param_group['lr'])
                                new_lr = old_lr * opt.learning_rate_decay
                                if old_lr - new_lr > EPS:
                                    param_group['lr'] = new_lr

                    logging.info('Epoch: %d; batch idx: %d; total batches: %d' % (epoch, batch_i, total_batch))
                    logging.info(
                        'avg training reward: %.4f; avg training loss: %.4f; avg validation reward: %.4f; best validation reward: %.4f' % (
                            current_train_reward, current_train_pg_loss, current_valid_reward, best_valid_reward))

                    report_train_reward.append(current_train_reward)
                    report_valid_reward.append(current_valid_reward)

                    if not opt.disable_early_stop_rl:
                        if num_stop_increasing >= opt.early_stop_tolerance:
                            logging.info(
                                'Have not increased for %d check points, early stop training' % num_stop_increasing)
                            early_stop_flag = True
                            break
                    report_train_reward_statistics.clear()

    # export the training curve
    train_valid_curve_path = opt.exp_path + '/train_valid_curve'
    # train_valid_curve_path = 'exp/train_valid_curve'
    export_train_and_valid_reward(report_train_reward, report_valid_reward, opt.checkpoint_interval,
                                  train_valid_curve_path)

    print()
    print("End of the Generator training")  # gl

######################################
