#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 22:41:50 2019

@author: r17935avinash
"""

# #############################################
# #  gl: uncomment for better cuda debug
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# #############################################

import argparse
import config
from Disc_train import main as D_train
from Bert_Disc_train import main as D_Bert_train
# from Gen_RL_Train import main as G_train
from Gen_RL_Train_new import main as G_train
import torch
import random
import numpy as np
import pykp
from torch.backends import cudnn


def process_opt(opt):
    if opt.seed > 0:
        torch.manual_seed(opt.seed)
        np.random.seed(opt.seed)
        random.seed(opt.seed)

        # torch.cuda.manual_seed(opt.seed)
        # torch.cuda.manual_seed_all(opt.seed)

        # gl: to gain reproducibility
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    if torch.cuda.is_available() and not opt.gpuid:
        opt.gpuid = 0
    if hasattr(opt, 'train_ml') and opt.train_ml:
        opt.exp += '.ml'

    if hasattr(opt, 'train_rl') and opt.train_rl:
        opt.exp += '.rl'

    if opt.one2many:
        opt.exp += '.one2many'

    if opt.one2many_mode == 1:
        opt.exp += '.cat'

    if opt.copy_attention:
        opt.exp += '.copy'

    if opt.coverage_attn:
        opt.exp += '.coverage'

    if opt.review_attn:
        opt.exp += '.review'

    if opt.orthogonal_loss:
        opt.exp += '.orthogonal'

    if opt.use_target_encoder:
        opt.exp += '.target_encode'

    if hasattr(opt, 'bidirectional') and opt.bidirectional:
        opt.exp += '.bi-directional'
    else:
        opt.exp += '.uni-directional'

    if opt.delimiter_type == 0:
        opt.delimiter_word = pykp.io.SEP_WORD
    else:
        opt.delimiter_word = pykp.io.EOS_WORD

    return opt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BERT_Discriminator.py', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    config.vocab_opts(parser)
    config.model_opts(parser)
    config.train_opts(parser)
    config.dis_opts(parser)
    config.bert_opts(parser)  # gl
    opt = parser.parse_args()   
    opt = process_opt(opt)
    if torch.cuda.is_available():
        if not opt.gpuid:
            opt.gpuid = 0
        opt.device = torch.device("cuda:%d" % opt.gpuid)
    else:
        opt.device = torch.device("cpu")
        opt.gpuid = -1
        print("CUDA is not available, fall back to CPU.")
    if opt.train_discriminator:
        if opt.use_bert_discriminator:  # gl
            D_Bert_train(opt)
        else:
            D_train(opt)

    elif opt.train_rl:
        G_train(opt)
        # G_train_original(opt)
