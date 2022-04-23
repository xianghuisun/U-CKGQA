from lib2to3.pgen2 import token
from operator import imod
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import torch
from transformers import BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
import transformers
from data import QAData
from bart import MyBart
from utils.utils import read_data,read_data_nlpcc,read_kg
from evaluate import get_hitsat1

import logging
logger=logging.getLogger("main.run")

def run(args):
    logger.info("transformers.__version__: {}".format(str(transformers.__version__)))
    tokenizer = BertTokenizer.from_pretrained(args.bart_model_path)

    dev_data = QAData(args=args, data_path= args.predict_file, is_training=False)

    if not args.skip_inference:
        dev_data.load_dataset(tokenizer)
        dev_data.load_dataloader()

    if args.do_train:
        train_data = QAData(args=args,data_path = args.train_file, is_training = True)
        train_data.sample_data(sample_proportion=args.proportion)
        train_data.load_dataset(tokenizer)
        train_data.load_dataloader()

        if args.checkpoint is not None and os.path.exists(args.checkpoint):
            model = MyBart.from_pretrained(args.bart_model_path,
                                           state_dict=torch.load(args.checkpoint))
            logger.info("Loading checkpoint from {}".format(args.checkpoint))
        else:
            model = MyBart.from_pretrained(args.bart_model_path)
        if args.n_gpu>1:
            model = torch.nn.DataParallel(model)
        if args.n_gpu>0:
            model.to(torch.device("cuda"))

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        t_total = len(train_data.dataloader) * args.num_train_epochs
        warmup_steps = int(t_total * args.warmup_proportion)
        scheduler =  get_linear_schedule_with_warmup(optimizer,
                                        num_warmup_steps=warmup_steps,
                                        num_training_steps=t_total)

        if args.kg_type=='kgclue':
            test_data=read_data(data_path=args.test_file)
        else:
            test_data=read_data_nlpcc(data_path=args.test_file)
        
        alias_map,sub_map=read_kg(kg_path=args.kg_path,kg=args.kg_type)

        train(args=args,
              model=model,
              train_data=train_data,
              dev_data=dev_data,
              optimizer=optimizer,
              scheduler=scheduler,
              test_data=test_data,
              tokenizer=tokenizer,
              alias_map=alias_map,
              sub_map=sub_map)

    # if args.do_predict:
    #     checkpoint = os.path.join(args.output_dir, 'best-model.pt') if args.checkpoint is None else args.checkpoint
    #     model = MyBart.from_pretrained(args.bart_model_path,
    #                                    state_dict=torch.load(checkpoint))
    #     logger.info("Loading checkpoint from {}".format(checkpoint))
    #     if args.n_gpu>0:
    #         model.to(torch.device("cuda"))
    #     model.eval()
    #     ems = inference(model, dev_data)
    #     logger.info("%s on %s data: %.2f" % (dev_data.metric, dev_data.data_type, np.mean(ems)*100))

def train(args, model, train_data, dev_data, optimizer, scheduler, test_data,tokenizer,alias_map,sub_map):
    model.train()
    global_step = 0
    train_losses = []
    best_accuracy = -1
    stop_training=False

    columns=['epoch','loss','hits@1']
    record_informations=[]
    record_epoch_losses=[]

    if args.checkpoint_step > 0:
        logger.info("Previous args.checkpoint_step : {}".format(args.checkpoint_step))
        for _ in range(args.checkpoint_step):
            global_step += 1
            scheduler.step()

    def convert_to_single_gpu(state_dict):
        def _convert(key):
            if key.startswith('module.'):
                return key[7:]
            return key
        return {_convert(key):value for key, value in state_dict.items()}

    logger.info("Starting training!")
    total_num_steps=int(args.num_train_epochs)*len(train_data.dataloader)
    show_loss_step=len(train_data.dataloader)//1
    eval_period=(total_num_steps//args.num_train_epochs)*5 #Each five epochs will evaluate
    logger.info("Each epoch has {} num_steps".format(len(train_data.dataloader)))

    for epoch in range(int(args.num_train_epochs)):
        for batch in tqdm(train_data.dataloader,total=len(train_data.dataloader),unit='batch'):
            global_step += 1
            batch = [b.to(torch.device("cuda")) for b in batch]
            loss = model(input_ids=batch[0], attention_mask=batch[1],
                         decoder_input_ids=batch[2], decoder_attention_mask=batch[3],
                         is_training=True)
            record_epoch_losses.append(loss.item())
            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu.
            if torch.isnan(loss).data:
                logger.info("Stop training because loss=%s" % (loss.data))
                stop_training=True
                break
            train_losses.append(loss.detach().cpu())
            loss.backward()

            if global_step % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()    # We have accumulated enought gradients
                scheduler.step()
                model.zero_grad()

            if global_step % show_loss_step == 0:
                mean_loss=np.mean(train_losses)
                logger.info("Epoch: {}, global_step/total_step:{}/{}, loss: {}".format(epoch,global_step,total_num_steps,mean_loss))
                train_losses = []

            if global_step % eval_period == 0:
                if args.skip_inference:
                    model_state_dict = {k:v.cpu() for (k, v) in model.state_dict().items()}
                    if args.n_gpu > 1:
                        model_state_dict = convert_to_single_gpu(model_state_dict)
                    torch.save(model_state_dict, os.path.join(args.output_dir,
                                                              "best-model-{}.pt".format(str(global_step).zfill(6))))
                else:
                    model.eval()
                    logger.info("Inference the model ...")
                    curr_em, predictions = inference(model if args.n_gpu==1 else model.module, dev_data)
                    logger.info("Step %d %s %.2f%% on epoch=%d" % (
                            global_step,
                            dev_data.metric,
                            curr_em*100,
                            epoch))
                    if best_accuracy < curr_em:
                        dev_data.save_predictions(predictions)
                        model_state_dict = {k:v.cpu() for (k, v) in model.state_dict().items()}
                        if args.n_gpu > 1:
                            model_state_dict = convert_to_single_gpu(model_state_dict)
                        torch.save(model_state_dict, os.path.join(args.output_dir, "best-model.pt"))
                        logger.info("Saving model with best %s: %.2f%% -> %.2f%% on epoch=%d, global_step=%d" % \
                                (dev_data.metric, best_accuracy*100.0, curr_em*100.0, epoch, global_step))
                        best_accuracy = curr_em
                        wait_step = 0
                        stop_training = False
                    else:
                        wait_step += 1
                        if wait_step >= args.wait_step:
                            stop_training = True
                            break
                model.train()
        if stop_training:
            break

        model.eval()
        with torch.no_grad():
            hitsat1=get_hitsat1(test_data,model,tokenizer,alias_map,sub_map)
            record_informations.append([epoch,np.mean(record_epoch_losses),hitsat1])
            logger.info("epoch: {}, loss: {}, hits@1: {}".format(epoch,mean_loss,hitsat1))
            record_epoch_losses=[]
        model.train()
    
    record_pd=pd.DataFrame(data=record_informations,columns=columns)
    record_pd.to_csv(args.record_information_path,index=None)
    
def inference(model, dev_data):
    predictions = []
    bos_token_id = dev_data.tokenizer.bos_token_id
    if dev_data.args.verbose:
        dev_data.dataloader = tqdm(dev_data.dataloader)
    for i, batch in tqdm(enumerate(dev_data.dataloader),total=len(dev_data.dataloader),unit='batch'):
        batch = [b.to(torch.device("cuda")) for b in batch]
        outputs = model.generate(input_ids=batch[0],
                                 attention_mask=batch[1],
                                 num_beams=dev_data.args.num_beams,
                                 min_length=1,
                                 max_length=dev_data.args.max_output_length,
                                 early_stopping=True,)
        for input_, output in zip(batch[0], outputs):
            pred = dev_data.decode(output)
            predictions.append(pred)
    return np.mean(dev_data.evaluate(predictions)),predictions







