#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 15:10:45 2019
@author: r17935avinash
"""

import math

# ############################### IMPORT LIBRARIES ###############################################################
import pykp.io
from pykp.reward import *
from sequence_generator import SequenceGenerator
from utils.statistics import RewardStatistics

EPS = 1e-8
import logging
import pykp
from pykp.model import Seq2SeqModel

from utils.time_log import time_since
from utils.data_loader import load_data_and_vocab
import time
from BERT_Discriminator import NetModel, NetModelMC, NLP_MODELS
from Bert_Disc_train import build_kps_idx_list, build_src_idx_list, build_training_batch

#####################################################################################################

torch.autograd.set_detect_anomaly(True)


# #  batch_reward_stat, log_selected_token_dist = train_one_batch(batch, generator, optimizer_rl, opt, perturb_std)
#########################################################
def train_one_batch(D_model, one2many_batch, generator, opt, perturb_std, bert_tokenizer, bert_model_name):
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
    # print()  # gl: debug
    # print()  # gl: debug
    # print(pred_str_2dlist)  # gl: debug
    # print(trg_str_2dlist)  # gl: debug
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
    pred_idx_list = build_kps_idx_list(pred_str_2dlist, bert_tokenizer, opt.separate_present_absent)
    greedy_idx_list = build_kps_idx_list(greedy_str_2dlist, bert_tokenizer, opt.separate_present_absent)
    src_idx_list = build_src_idx_list(src_str_list, bert_tokenizer)
    # print(pred_idx_list)
    # print(greedy_idx_list)
    # print(src_idx_list)
    # torch.save(pred_idx_list, 'prova/pred_idx_list.pt')  # gl saving tensors
    # torch.save(target_idx_list, 'prova/target_idx_list.pt')  # gl saving tensors

    pred_train_batch, pred_mask_batch, pred_segment_batch, _ = \
        build_training_batch(src_idx_list, pred_idx_list, bert_tokenizer, opt, label=0)
    greedy_train_batch, greedy_mask_batch, greedy_segment_batch, _ = \
        build_training_batch(src_idx_list, greedy_idx_list, bert_tokenizer, opt, label=1)

    # gl: only for seq. class; for multi choice add targets variables
    pred_train_batch = torch.tensor(pred_train_batch, dtype=torch.long).to(devices)
    pred_mask_batch = torch.tensor(pred_mask_batch, dtype=torch.long).to(devices)
    pred_segment_batch = torch.tensor(pred_segment_batch, dtype=torch.long).to(devices)
    pred_rewards = np.zeros(batch_size)
    for idx, (input_ids, input_mask, input_segment) in enumerate(
            zip(pred_train_batch, pred_mask_batch, pred_segment_batch)):
        # print(idx)
        # print(input_ids)
        # print(input_mask)
        # print(input_segment)
        output = D_model(input_ids.unsqueeze(0),
                         attention_mask=input_mask.unsqueeze(0),
                         token_type_ids=input_segment.unsqueeze(0),
                         )
        # print(output[0].item())  # gl: debug
        pred_rewards[idx] = output[0]
    # torch.save(pred_rewards, 'prova/pred_rewards.pt')  # gl saving tensors

    # print()  # gl: debug
    greedy_train_batch = torch.tensor(greedy_train_batch, dtype=torch.long).to(devices)
    greedy_mask_batch = torch.tensor(greedy_mask_batch, dtype=torch.long).to(devices)
    greedy_segment_batch = torch.tensor(greedy_segment_batch, dtype=torch.long).to(devices)
    baseline_rewards = np.zeros(batch_size)
    for idx, (input_ids, input_mask, input_segment) in enumerate(
            zip(greedy_train_batch, greedy_mask_batch, greedy_segment_batch)):
        # print(idx)
        output = D_model(input_ids.unsqueeze(0),
                         attention_mask=input_mask.unsqueeze(0),
                         token_type_ids=input_segment.unsqueeze(0),
                         )
        # print(output[0].item())  # gl: debug
        baseline_rewards[idx] = output[0]
    # torch.save(baseline_rewards, 'prova/baseline_rewards.pt')  # gl saving tensors





    # pred = np.zeros(batch_size)
    # gred = np.zeros(batch_size)
    # for idx, (p_input_ids, p_input_mask, p_input_segment, g_input_ids, g_input_mask, g_input_segment) in enumerate(
    #     zip(pred_train_batch, pred_mask_batch, pred_segment_batch, greedy_train_batch, greedy_mask_batch, greedy_segment_batch)
    # ):
    #
    #     if (idx + 1) % 3 == 0:
    #
    #     output_p = D_model(p_input_ids.unsqueeze(0),
    #                        attention_mask=p_input_mask.unsqueeze(0),
    #                        token_type_ids=p_input_segment.unsqueeze(0),
    #                        )
    #     output_g = D_model(g_input_ids.unsqueeze(0),
    #                        attention_mask=g_input_mask.unsqueeze(0),
    #                        token_type_ids=g_input_segment.unsqueeze(0),
    #                        )
    #
    #     pred[idx] = output_p[0]
    #     gred[idx] = output_g[0]
    #
    # print('pred_rewards : ', pred_rewards)
    # print('pred         : ', pred)
    # print('baseline_rewards : ', baseline_rewards)
    # print('gred             : ', gred)





    cumulative_reward_sum = pred_rewards.sum(0)
    # cumulative_bas_reward_sum = baseline_rewards.sum(0)
    batch_rewards = pred_rewards - baseline_rewards
    # torch.save(batch_rewards, 'prova/batch_rewards.pt')  # gl saving tensors
    # print('pred_rewards     :' + str(pred_rewards))
    # print('baseline_rewards :' + str(baseline_rewards))
    # print('batch_rewards    :' + str(batch_rewards))
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

    stat = RewardStatistics(cumulative_reward_sum, pg_loss.item(), batch_size, sample_time, q_estimate_compute_time, 0)

    # return stat, pg_loss
    return pg_loss


def main(opt):
    # print("agsnf efnghrrqthg")
    # clip = 5
    report_train_reward_statistics = RewardStatistics()
    total_train_reward_statistics = RewardStatistics()
    report_train_reward = []
    report_valid_reward = []
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
    # hidden_dim = opt.D_hidden_dim
    # embedding_dim = opt.D_embedding_dim
    # n_layers = opt.D_layers
    #
    # hidden_dim = opt.D_hidden_dim
    # embedding_dim = opt.D_embedding_dim
    # n_layers = opt.D_layers

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
    print("Beginning with training Generator")
    print("########################################################################################################")
    total_epochs = opt.epochs
    for epoch in range(total_epochs):

        total_batch = 0
        print("Starting with epoch:", epoch)
        for batch_i, batch in enumerate(train_data_loader):

            model.train()
            PG_optimizer.zero_grad()

            if perturb_decay_mode == 0:  # do not decay
                perturb_std = init_perturb_std
            elif perturb_decay_mode == 1:  # exponential decay
                perturb_std = final_perturb_std + (init_perturb_std - final_perturb_std) * math.exp(
                    -1. * total_batch * perturb_decay_factor)
            elif perturb_decay_mode == 2:  # steps decay
                perturb_std = init_perturb_std * math.pow(perturb_decay_factor, math.floor((1 + total_batch) / 4000))

            # batch_reward_stat, avg_rewards = train_one_batch(D_model, batch, generator, opt, perturb_std, bert_tokenizer, bert_model_name)
            avg_rewards = train_one_batch(D_model, batch, generator, opt, perturb_std, bert_tokenizer, bert_model_name)
            # report_train_reward_statistics.update(batch_reward_stat)
            # total_train_reward_statistics.update(batch_reward_stat)

            if opt.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.max_grad_norm)
            avg_rewards.backward()
            PG_optimizer.step()

            if batch_i % 4000 == 0:
                print("Saving the file ...............----------->>>>>")
                print("The avg reward is", -avg_rewards.item())
                state_dfs = model.state_dict()
                torch.save(state_dfs, "RL_Checkpoints/Attention_Generator_" + str(epoch) + ".pth.tar")

    print()
    print("End of the Generator training")  # gl

######################################
