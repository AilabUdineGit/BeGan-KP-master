# from nltk.stem.porter import *
import os
import sys
import time

import numpy as np
import torch

# from nltk.stem.porter import *
import pykp
from Bert_Disc_train import build_kps_idx_list, build_src_idx_list, build_training_batch, shuffle_input_samples
# from utils import Progbar
# from pykp.metric.bleu import bleu
from pykp.masked_loss import masked_cross_entropy
from pykp.reward import sample_list_to_str_2dlist, compute_batch_reward
from utils.statistics import LossStatistics, RewardStatistics
from utils.string_helper import *
from utils.time_log import time_since


# stemmer = PorterStemmer()
def remove_duplications(kph):
    b_set = set(tuple(x) for x in kph)
    b = [list(x) for x in b_set]
    b.sort(key=lambda x: kph.index(x))
    return b


def convert_to_string_list(kph, idx2word):
    s = []
    for kps in kph:
        s.append(idx2word[kps])
    return s


def evaluate_loss(data_loader, model, opt):  # gl: only for ML model training
    model.eval()
    evaluation_loss_sum = 0.0
    total_trg_tokens = 0
    n_batch = 0
    loss_compute_time_total = 0.0
    forward_time_total = 0.0

    with torch.no_grad():
        for batch_i, batch in enumerate(data_loader):
            if not opt.one2many:  # load one2one dataset
                src, src_lens, src_mask, trg, trg_lens, trg_mask, src_oov, trg_oov, oov_lists, title, title_oov, title_lens, title_mask = batch
            else:  # load one2many dataset
                src, src_lens, src_mask, src_oov, oov_lists, src_str_list, trg_str_2dlist, trg, trg_oov, trg_lens, trg_mask, _, title, title_oov, title_lens, title_mask = batch
                num_trgs = [len(trg_str_list) for trg_str_list in
                            trg_str_2dlist]  # a list of num of targets in each batch, with len=batch_size

            max_num_oov = max([len(oov) for oov in oov_lists])  # max number of oov for each batch

            batch_size = src.size(0)
            n_batch += batch_size

            # move data to GPU if available
            src = src.to(opt.device)
            src_mask = src_mask.to(opt.device)
            trg = trg.to(opt.device)
            trg_mask = trg_mask.to(opt.device)
            src_oov = src_oov.to(opt.device)
            trg_oov = trg_oov.to(opt.device)
            if opt.title_guided:
                title = title.to(opt.device)
                title_mask = title_mask.to(opt.device)
                # title_oov = title_oov.to(opt.device)

            start_time = time.time()
            if not opt.one2many:
                decoder_dist, h_t, attention_dist, encoder_final_state, coverage, _, _, _ = model(src, src_lens, trg,
                                                                                                  src_oov, max_num_oov,
                                                                                                  src_mask, title=title,
                                                                                                  title_lens=title_lens,
                                                                                                  title_mask=title_mask)
            else:
                decoder_dist, h_t, attention_dist, encoder_final_state, coverage, _, _, _ = model(src, src_lens, trg,
                                                                                                  src_oov, max_num_oov,
                                                                                                  src_mask, num_trgs,
                                                                                                  title=title,
                                                                                                  title_lens=title_lens,
                                                                                                  title_mask=title_mask)
            forward_time = time_since(start_time)
            forward_time_total += forward_time

            start_time = time.time()
            if opt.copy_attention:  # Compute the loss using target with oov words
                loss = masked_cross_entropy(decoder_dist, trg_oov, trg_mask, trg_lens,
                                            opt.coverage_attn, coverage, attention_dist, opt.lambda_coverage,
                                            coverage_loss=False)
            else:  # Compute the loss using target without oov words
                loss = masked_cross_entropy(decoder_dist, trg, trg_mask, trg_lens,
                                            opt.coverage_attn, coverage, attention_dist, opt.lambda_coverage,
                                            coverage_loss=False)
            loss_compute_time = time_since(start_time)
            loss_compute_time_total += loss_compute_time

            evaluation_loss_sum += loss.item()
            total_trg_tokens += sum(trg_lens)

    eval_loss_stat = LossStatistics(evaluation_loss_sum, total_trg_tokens, n_batch, forward_time=forward_time_total,
                                    loss_compute_time=loss_compute_time_total)
    return eval_loss_stat


def evaluate_reward(data_loader, generator, opt):
    """Return the avg. reward in the validation dataset"""
    generator.model.eval()
    final_reward_sum = 0.0
    n_batch = 0
    sample_time_total = 0.0
    topk = opt.topk
    reward_type = opt.reward_type
    # reward_type = 7
    match_type = opt.match_type
    eos_idx = opt.word2idx[pykp.io.EOS_WORD]
    delimiter_word = opt.delimiter_word
    one2many = opt.one2many
    one2many_mode = opt.one2many_mode
    if one2many and one2many_mode > 1:
        num_predictions = opt.num_predictions
    else:
        num_predictions = 1

    with torch.no_grad():
        for batch_i, batch in enumerate(data_loader):
            # load one2many dataset
            src, src_lens, src_mask, src_oov, oov_lists, src_str_list, trg_str_2dlist, trg, trg_oov, trg_lens, trg_mask, _, title, title_oov, title_lens, title_mask = batch
            num_trgs = [len(trg_str_list) for trg_str_list in
                        trg_str_2dlist]  # a list of num of targets in each batch, with len=batch_size

            batch_size = src.size(0)
            n_batch += batch_size

            # move data to GPU if available
            src = src.to(opt.device)
            src_mask = src_mask.to(opt.device)
            src_oov = src_oov.to(opt.device)
            # trg = trg.to(opt.device)
            # trg_mask = trg_mask.to(opt.device)
            # trg_oov = trg_oov.to(opt.device)
            if opt.title_guided:
                title = title.to(opt.device)
                title_mask = title_mask.to(opt.device)
                # title_oov = title_oov.to(opt.device)

            start_time = time.time()
            # sample a sequence
            # sample_list is a list of dict, {"prediction": [], "scores": [], "attention": [], "done": True}, preidiction is a list of 0 dim tensors
            sample_list, log_selected_token_dist, output_mask, pred_idx_mask, _, _, _ = generator.sample(
                src, src_lens, src_oov, src_mask, oov_lists, opt.max_length, greedy=True, one2many=one2many,
                one2many_mode=one2many_mode, num_predictions=num_predictions, perturb_std=0, title=title,
                title_lens=title_lens, title_mask=title_mask)
            # pred_str_2dlist = sample_list_to_str_2dlist(sample_list, oov_lists, opt.idx2word, opt.vocab_size, eos_idx, delimiter_word)
            pred_str_2dlist = sample_list_to_str_2dlist(sample_list, oov_lists, opt.idx2word, opt.vocab_size, eos_idx,
                                                        delimiter_word, opt.word2idx[pykp.io.UNK_WORD], opt.replace_unk,
                                                        src_str_list)
            # print(pred_str_2dlist)
            sample_time = time_since(start_time)
            sample_time_total += sample_time

            final_reward = compute_batch_reward(pred_str_2dlist, trg_str_2dlist, batch_size, reward_type, topk,
                                                match_type, regularization_factor=0.0)  # np.array, [batch_size]

            final_reward_sum += final_reward.sum(0)

    eval_reward_stat = RewardStatistics(final_reward_sum, pg_loss=0, n_batch=n_batch, sample_time=sample_time_total)

    return eval_reward_stat


def reward_function(discriminator_output, batch_size, bert_model_name, labels=None):
    """
    Reward to use with BertForMultipleChoice
    :param: discriminator_output, pytorch tensor, the output of Discriminator
    :param: batch_size, int
    :param: bert_model_name, str, specific Bert model for Discriminator
    :param: labels, list of integer
    :return: numpy array of floats, the reward
    """
    rewards = np.zeros(batch_size)
    idx = 0
    for val_b in discriminator_output:
        if bert_model_name == 'BertForSequenceClassification':
            rewards[idx] = val_b.item()
        elif bert_model_name == 'BertForMultipleChoice':
            # rewards[idx] = 1 - (val_b[0].item() - val_b[1].item()) ** 2
            if labels:
                rewards[idx] = 1 - abs((val_b[0].item() - val_b[1].item()) / val_b[labels[idx]].item())
                # print('val_b[0]     : ', val_b[0].item())
                # print('val_b[1]     : ', val_b[1].item())
                # print('rewards[idx] : ', rewards[idx])
                # print('labels[idx]  : ', labels[idx])
                # print()
            else:
                # rewards[idx] = 1 - abs(val_b[0].item() - val_b[1].item())
                rewards[idx] = np.exp(-abs(val_b[0].item() - val_b[1].item()))
        idx += 1

    return rewards


def evaluate_valid_reward(data_loader, generator, opt, D_model, bert_tokenizer, bert_model_name):
    """Return the avg. reward in the validation dataset"""
    generator.model.eval()
    final_reward_sum = 0.0
    n_batch = 0
    sample_time_total = 0.0
    topk = opt.topk
    reward_type = opt.reward_type
    # reward_type = 7
    match_type = opt.match_type
    eos_idx = opt.word2idx[pykp.io.EOS_WORD]
    delimiter_word = opt.delimiter_word
    one2many = opt.one2many
    one2many_mode = opt.one2many_mode
    if one2many and one2many_mode > 1:
        num_predictions = opt.num_predictions
    else:
        num_predictions = 1

    D_model.eval()
    with torch.no_grad():
        for batch_i, batch in enumerate(data_loader):
            # print('batch_i : ', batch_i)  # gl: debug
            # load one2many dataset
            src, src_lens, src_mask, src_oov, oov_lists, src_str_list, trg_str_2dlist, trg, trg_oov, trg_lens, trg_mask, _, title, title_oov, title_lens, title_mask = batch
            # num_trgs = [len(trg_str_list) for trg_str_list in
            #             trg_str_2dlist]  # a list of num of targets in each batch, with len=batch_size

            batch_size = src.size(0)
            n_batch += batch_size

            # move data to GPU if available
            src = src.to(opt.device)
            src_mask = src_mask.to(opt.device)
            src_oov = src_oov.to(opt.device)
            # trg = trg.to(opt.device)
            # trg_mask = trg_mask.to(opt.device)
            # trg_oov = trg_oov.to(opt.device)
            if opt.title_guided:
                title = title.to(opt.device)
                title_mask = title_mask.to(opt.device)
                # title_oov = title_oov.to(opt.device)

            start_time = time.time()
            # sample a sequence
            # sample_list is a list of dict, {"prediction": [], "scores": [], "attention": [], "done": True}, preidiction is a list of 0 dim tensors
            sample_list, log_selected_token_dist, output_mask, pred_idx_mask, _, _, _ = generator.sample(
                src, src_lens, src_oov, src_mask, oov_lists, opt.max_length, greedy=True, one2many=one2many,
                one2many_mode=one2many_mode, num_predictions=num_predictions, perturb_std=0, title=title,
                title_lens=title_lens, title_mask=title_mask)
            pred_str_2dlist = sample_list_to_str_2dlist(sample_list, oov_lists, opt.idx2word, opt.vocab_size, eos_idx,
                                                        delimiter_word, opt.word2idx[pykp.io.UNK_WORD], opt.replace_unk,
                                                        src_str_list)
            # print(pred_str_2dlist)
            sample_time = time_since(start_time)
            sample_time_total += sample_time

            pred_idx_list = build_kps_idx_list(pred_str_2dlist, bert_tokenizer, opt.separate_present_absent)
            src_idx_list = build_src_idx_list(src_str_list, bert_tokenizer)

            pred_train_batch, pred_mask_batch, pred_segment_batch, _ = \
                build_training_batch(src_idx_list, pred_idx_list, bert_tokenizer, opt, label=0)

            if bert_model_name == 'BertForSequenceClassification':
                pred_train_batch = torch.tensor(pred_train_batch, dtype=torch.long).to(opt.device)
                pred_mask_batch = torch.tensor(pred_mask_batch, dtype=torch.long).to(opt.device)
                pred_segment_batch = torch.tensor(pred_segment_batch, dtype=torch.long).to(opt.device)

                # # final_reward = np.zeros(batch_size)
                # output = D_model(pred_train_batch,
                #                  attention_mask=pred_mask_batch,
                #                  token_type_ids=pred_segment_batch,
                #                  )
                # # idx = 0
                # # for val_p in output[0]:
                # #     final_reward[idx] = val_p.item()
                # #     idx += 1
                # final_reward = reward_function(output[0], batch_size, bert_model_name)

            elif bert_model_name == 'BertForMultipleChoice':
                trg_idx_list = build_kps_idx_list(trg_str_2dlist, bert_tokenizer, opt.separate_present_absent)

                trg_train_batch, trg_mask_batch, trg_segment_batch, _ = \
                    build_training_batch(src_idx_list, trg_idx_list, bert_tokenizer, opt, label=0)

                pred_train_batch, pred_mask_batch, pred_segment_batch, labels = \
                    shuffle_input_samples(batch_size, pred_train_batch, trg_train_batch,
                                          pred_mask_batch, trg_mask_batch,
                                          pred_segment_batch, trg_segment_batch)

                pred_train_batch = torch.tensor(pred_train_batch, dtype=torch.long).to(opt.device)
                pred_mask_batch = torch.tensor(pred_mask_batch, dtype=torch.long).to(opt.device)
                pred_segment_batch = torch.tensor(pred_segment_batch, dtype=torch.long).to(opt.device)

            output = D_model(pred_train_batch,
                             attention_mask=pred_mask_batch,
                             token_type_ids=pred_segment_batch,
                             )
            # final_reward = reward_function(output[0], batch_size, bert_model_name, labels)
            final_reward = reward_function(output[0], batch_size, bert_model_name)

            # final_reward = compute_batch_reward(pred_str_2dlist, trg_str_2dlist, batch_size, reward_type, topk,
            #                                     match_type, regularization_factor=0.0)  # np.array, [batch_size]
            final_reward_sum += final_reward.sum(0)
            # print('(validation) final_reward     : ', final_reward)
            # print('(validation) final_reward_sum : ', final_reward_sum)

    eval_reward_stat = RewardStatistics(final_reward_sum, pg_loss=0, n_batch=n_batch, sample_time=sample_time_total)

    return eval_reward_stat


def preprocess_beam_search_result(beam_search_result, idx2word, vocab_size, oov_lists, eos_idx, unk_idx, replace_unk,
                                  src_str_list):
    batch_size = beam_search_result['batch_size']
    predictions = beam_search_result['predictions']
    scores = beam_search_result['scores']
    attention = beam_search_result['attention']
    assert len(predictions) == batch_size
    pred_list = []  # a list of dict, with len = batch_size
    for pred_n_best, score_n_best, attn_n_best, oov, src_word_list in zip(predictions, scores, attention, oov_lists,
                                                                          src_str_list):
        # attn_n_best: list of tensor with size [trg_len, src_len], len=n_best
        pred_dict = {}
        sentences_n_best = []
        for pred, attn in zip(pred_n_best, attn_n_best):
            sentence = prediction_to_sentence(pred, idx2word, vocab_size, oov, eos_idx, unk_idx, replace_unk,
                                              src_word_list, attn)
            # sentence = [idx2word[int(idx.item())] if int(idx.item()) < vocab_size else oov[int(idx.item())-vocab_size] for idx in pred[:-1]]
            sentences_n_best.append(sentence)
        pred_dict[
            'sentences'] = sentences_n_best  # a list of list of word, with len [n_best, out_seq_len], does not include tbe final <EOS>
        pred_dict['scores'] = score_n_best  # a list of zero dim tensor, with len [n_best]
        pred_dict[
            'attention'] = attn_n_best  # a list of FloatTensor[output sequence length, src_len], with len = [n_best]
        pred_list.append(pred_dict)
    return pred_list


def evaluate_beam_search(generator, one2many_data_loader, opt, delimiter_word='<sep>'):
    # score_dict_all = defaultdict(list)  # {'precision@5':[],'recall@5':[],'f1_score@5':[],'num_matches@5':[],'precision@10':[],'recall@10':[],'f1score@10':[],'num_matches@10':[]}
    # file for storing the predicted keyphrases
    if opt.pred_file_prefix == "":
        pred_output_file = open(os.path.join(opt.pred_path, "predictions.txt"), "w")
    else:
        pred_output_file = open(os.path.join(opt.pred_path, "%s_predictions.txt" % opt.pred_file_prefix), "w")
    # debug
    interval = 1000

    with torch.no_grad():
        start_time = time.time()
        for batch_i, batch in enumerate(one2many_data_loader):
            if (batch_i + 1) % interval == 0:
                print("Batch %d: Time for running beam search on %d batches : %.1f" % (
                batch_i + 1, interval, time_since(start_time)))
                sys.stdout.flush()
                start_time = time.time()
            src, src_lens, src_mask, src_oov, oov_lists, src_str_list, trg_str_2dlist, _, _, _, _, original_idx_list, title, title_oov, title_lens, title_mask = batch
            """
            src: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], with oov words replaced by unk idx
            src_lens: a list containing the length of src sequences for each batch, with len=batch
            src_mask: a FloatTensor, [batch, src_seq_len]
            src_oov: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], contains the index of oov words (used by copy)
            oov_lists: a list of oov words for each src, 2dlist
            """
            src = src.to(opt.device)
            src_mask = src_mask.to(opt.device)
            src_oov = src_oov.to(opt.device)
            if opt.title_guided:
                title = title.to(opt.device)
                title_mask = title_mask.to(opt.device)
                # title_oov = title_oov.to(opt.device)

            beam_search_result = generator.beam_search(src, src_lens, src_oov, src_mask, oov_lists, opt.word2idx,
                                                       opt.max_eos_per_output_seq, title=title, title_lens=title_lens,
                                                       title_mask=title_mask)
            pred_list = preprocess_beam_search_result(beam_search_result, opt.idx2word, opt.vocab_size, oov_lists,
                                                      opt.word2idx[pykp.io.EOS_WORD], opt.word2idx[pykp.io.UNK_WORD],
                                                      opt.replace_unk, src_str_list)
            # list of {"sentences": [], "scores": [], "attention": []}

            # recover the original order in the dataset
            seq_pairs = sorted(zip(original_idx_list, src_str_list, trg_str_2dlist, pred_list, oov_lists),
                               key=lambda p: p[0])
            original_idx_list, src_str_list, trg_str_2dlist, pred_list, oov_lists = zip(*seq_pairs)

            # Process every src in the batch
            for src_str, trg_str_list, pred, oov in zip(src_str_list, trg_str_2dlist, pred_list, oov_lists):
                # src_str: a list of words; trg_str: a list of keyphrases, each keyphrase is a list of words
                # pred_seq_list: a list of sequence objects, sorted by scores
                # oov: a list of oov words
                pred_str_list = pred[
                    'sentences']  # predicted sentences from a single src, a list of list of word, with len=[beam_size, out_seq_len], does not include the final <EOS>
                # print(pred_str_list)
                pred_score_list = pred['scores']
                pred_attn_list = pred[
                    'attention']  # a list of FloatTensor[output sequence length, src_len], with len = [n_best]

                if opt.one2many:
                    all_keyphrase_list = []  # a list of word list contains all the keyphrases in the top max_n sequences decoded by beam search
                    for word_list in pred_str_list:
                        all_keyphrase_list += split_word_list_by_delimiter(word_list, delimiter_word,
                                                                           opt.separate_present_absent,
                                                                           pykp.io.PEOS_WORD)
                        # not_duplicate_mask = check_duplicate_keyphrases(all_keyphrase_list)
                    # pred_str_list = [word_list for word_list, is_keep in zip(all_keyphrase_list, not_duplicate_mask) if is_keep]
                    pred_str_list = all_keyphrase_list

                # output the predicted keyphrases to a file
                pred_print_out = ''

                # pred_str_list = remove_duplications(pred_str_list)
                # print(pred_str_list)
                for word_list_i, word_list in enumerate(pred_str_list):
                    word_list = convert_to_string_list(word_list, opt.idx2word)
                    if word_list_i < len(pred_str_list) - 1:
                        pred_print_out += '%s;' % ' '.join(word_list)
                    else:
                        pred_print_out += '%s' % ' '.join(word_list)
                pred_print_out += '\n'
                pred_output_file.write(pred_print_out)

    pred_output_file.close()
    print("done!")


if __name__ == '__main__':
    pass
