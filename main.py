# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
import logging

import random
import numpy as np
import torch
import pandas as pd

from run import run
from utils.utils import convert_ner_to_prompt

def main():
    parser = argparse.ArgumentParser()

    ## As the dataset or task changes

    parser.add_argument("--knowledge_graph", type=str, required=True,help="train file")
    parser.add_argument("--kg", type=str, default="kgclue")

    # parser.add_argument("--train_file", default='/home/xhsun/Desktop/graduate_saved_files/section4/nlpcc/mixture/prompt_train.csv')
    # parser.add_argument("--predict_file", default='/home/xhsun/Desktop/graduate_saved_files/section4/nlpcc/mixture/prompt_test.csv')

    parser.add_argument("--checkpoints", type=str, required=True,help="output_dir")
    parser.add_argument('--proportion', type=float, default=1,help="few-shot train data percent")
    parser.add_argument("--num_train_epochs", default=50, type=float,
                        help="Total number of training epochs to perform.")

    parser.add_argument('--load_process_tokenized', type=bool, default=False, help='Load preprocessed tokenized data. Do not use it when set proportion')
    parser.add_argument('--max_input_length', type=int, default=256)
    parser.add_argument('--max_output_length', type=int, default=30)
    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument('--prefix', type=str, default='unified-',
                        help="Prefix for saving predictions")
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
                        
    ## Maybe will change
    parser.add_argument("--bart_model_path", type=str, required=True,help="bart_model_path")
    parser.add_argument('--checkpoint_step', type=int, default=0)
    parser.add_argument("--do_lowercase", default=True)
    parser.add_argument("--append_another_bos", default=False)#中文tokenizer的case下，加上<s>会有问题

    ## No need to change
    parser.add_argument("--do_train", default=True)
    parser.add_argument("--do_predict", default=True)
    parser.add_argument("--skip_inference", default=False)
    parser.add_argument('--num_beams', type=int, default=4)

    parser.add_argument("--predict_batch_size", default=128, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--wait_step', type=int, default=10000)

    # Other parameters
    parser.add_argument("--verbose", default=False,
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        print("Output directory () already exists and is not empty.")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    # converted_train_examples=convert_ner_to_prompt(data_path=os.path.join(args.knowledge_graph,'train.json'))
    # converted_test_examples=convert_ner_to_prompt(data_path=os.path.join(args.knowledge_graph,'test_public.json'))
    # columns=['question','label']
    # converted_train_pd=pd.DataFrame(converted_train_examples,columns=columns)
    # converted_test_pd=pd.DataFrame(converted_test_examples,columns=columns)
    # args.train_file=os.path.join(args.knowledge_graph,'prompt_train.csv')
    # args.predict_file=os.path.join(args.knowledge_graph,'prompt_test.csv')
    # args.test_file=os.path.join(args.knowledge_graph,'test_public.json')
    # converted_train_pd.to_csv(args.train_file,index=None)
    # converted_test_pd.to_csv(args.predict_file,index=None)

    ##### Start writing logs
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG) 
    file_handler = logging.FileHandler(os.path.join(args.output_dir,'log.txt'),mode='w')
    file_handler.setLevel(logging.INFO) 
    file_handler.setFormatter(
            logging.Formatter(
                    fmt='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
            )
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
            logging.Formatter(
                    fmt='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
            )
    logger.addHandler(console_handler)

    logger.info(args)
    logger.info(args.output_dir)

    for key,value in args.__dict__.items():
        logger.info("{} : {}".format(str(key),str(value)))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.n_gpu = torch.cuda.device_count()

    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_predict:
        raise ValueError("At least one of `do_train` or `do_predict` must be True.")

    if args.do_train:
        if not args.train_file:
            raise ValueError("If `do_train` is True, then `train_file` must be specified.")
        if not args.predict_file:
            raise ValueError("If `do_train` is True, then `predict_file` must be specified.")

    if args.do_predict:
        if not args.predict_file:
            raise ValueError("If `do_predict` is True, then `predict_file` must be specified.")

    logger.info("Using {} gpus".format(args.n_gpu))
    try:
        run(args=args)
    except Exception as e:
        print(e)
        logger.exception(e)
if __name__=='__main__':
    #main()
    pass
