import argparse
import os,sys,json,re,zhconv
from tqdm import tqdm
from collections import defaultdict
import torch
import numpy as np
from transformers import BertTokenizer
import logging,random

from bart import MyBart
from utils.utils import read_data,read_data_nlpcc,read_kg
from tools import get_topic_entity, evaluate_ner, evaluate_kgqa, evaluate_relation
from tools import inference_ner, inference_relation

log_filename = "evaluate-log.txt"

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                datefmt='%m/%d/%Y %H:%M:%S',
                level=logging.INFO,
                handlers=[logging.FileHandler(log_filename,mode='w'),
                            logging.StreamHandler()]
                    )

logger = logging.getLogger('evaluate')


device=torch.device('cuda')


def predict_ner(data,model,tokenizer):
    for example in tqdm(data,total=len(data),unit='question'):
        text=example['question'].lower().replace('“','').replace('”','').replace('"','').replace('"','').replace('\n','').replace('\t','')
        text='{}'.format(text)+'。上个句子中哪些属于实体名词？_。'#'你知道泰山极顶的作家是谁啊？。上个句子中哪些属于实体名词？_。'
        # model_inputs=tokenizer([text],return_tensors='pt')
        # input_ids=model_inputs['input_ids'].to(device)
        # attention_mask=model_inputs['attention_mask'].to(device)
        # outputs = model.generate(input_ids=input_ids,
        #                         attention_mask=attention_mask,
        #                         num_beams=4,
        #                         min_length=1,
        #                         max_length=30,
        #                         early_stopping=True,)#tensor([[ 102,  101, 3805, 2255, 3353, 7553,  102]], device='cuda:0')
        # pred=tokenizer.decode(outputs[0],skip_special_tokens=True,clean_up_tokenization_spaces=True).strip()#'泰 山 极 顶'
        pred=inference_ner(text,model=model,tokenizer=tokenizer)
        example.update({"ner_pred":pred})


def predict_relation(ner_data,model,tokenizer,alias_map,sub_map):
    for example in tqdm(ner_data,total=len(ner_data),unit='question'):
        question=example['question']
        entity=example['ner_pred']

        alias_map[entity].add(entity)
        pred_results=[]
        model_inputs=[]
        for alias_ent in alias_map[entity]:
            #找到这个主题实体对应的所有别名实体
            relations=[]
            for r,t in sub_map[alias_ent]:
                relations.append(r)
            #获取这个别名实体的所有关系
            
            candidate_relations=relations
            candidate_relations.append('不匹配')
            random.shuffle(candidate_relations)#确保“不匹配”这个选项不总是在最后一个位置
            
            relation_string=[]
            for i,rel in enumerate(candidate_relations):
                relation_string.append('（{}）'.format(i)+rel)
                
            relation_string='，'.join(relation_string)
            text='{}'.format(question)+'。'+'实体名词是：{}'.format(alias_ent)+"。这个句子的意图与下列哪一个关系最相似？_。"+' '+relation_string
            model_inputs.append(text)
        
        if model_inputs==[]:
            print(example,alias_map[entity])
            raise Exception("check")

        outputs=inference_relation(model_inputs,model=model,tokenizer=tokenizer)
        assert outputs.size(0)==len(model_inputs)
        for input_,output in zip(model_inputs,outputs):
            pred=tokenizer.decode(output,skip_special_tokens=True,clean_up_tokenization_spaces=True).strip()
            pred_results.append([input_,pred])
            
        example.update({"rel_pred":pred_results})

def get_hitsat1(data,model,tokenizer,alias_map,sub_map):
    predict_ner(data=data,model=model,tokenizer=tokenizer)
    get_topic_entity(data=data)

    tmp_ner_path='outputs/tmp_ner.json'
    with open(tmp_ner_path,'w') as f:
        for example in data:
            f.write(json.dumps(example,ensure_ascii=False)+'\n')
    
    evaluate_ner(tmp_ner_path,alias_map,sub_map,use_rule12=True)

    ner_data=[]
    with open(tmp_ner_path) as f:
        lines=f.readlines()
        for line in lines:
            ner_data.append(json.loads(line.strip()))

    predict_relation(ner_data=ner_data,model=model,tokenizer=tokenizer,alias_map=alias_map,sub_map=sub_map)
    hitsat1=evaluate_kgqa(data=ner_data,sub_map=sub_map)
    return hitsat1

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    ## As the dataset or task changes
    parser.add_argument("--test_file", type=str, default='/home/xhsun/Desktop/gitRepositories/graduation-project/code/data/kgclue/test_public.json')
    parser.add_argument("--kg_path", type=str, default='/home/xhsun/NLP/KGQA/KG/kgCLUE/Knowledge.txt')
    parser.add_argument("--kg_type", type=str, default='kgclue')
    parser.add_argument("--checkpoint", type=str,default='/home/xhsun/Desktop/graduate_models/Section4/kgclue/best-model.pt')

    # parser.add_argument("--test_file", type=str, default='/home/xhsun/Desktop/gitRepositories/graduation-project/code/data/nlpcc2018/test.txt')
    # parser.add_argument("--kg_path", type=str, default='/home/xhsun/NLP/KGQA/KG/nlpcc2018/knowledge/nlpcc-iccpol-2016.kbqa.kb')
    # parser.add_argument("--kg_type", type=str, default='nlpcc')
    # parser.add_argument("--checkpoint", type=str,default='/home/xhsun/Desktop/graduate_models/Section4/nlpcc/mixture/best-model.pt')

    parser.add_argument("--bart_model_path", type=str,default='/home/xhsun/NLP/huggingfaceModels/Chinese/chinese-bart-base')
    parser.add_argument('--num_beams', type=int, default=4)
    parser.add_argument('--max_output_length', type=int, default=30)

    # args = parser.parse_args()

    # if args.kg_type=='kgclue':
    #     data=read_data(data_path=args.test_file)
    # else:
    #     data=read_data_nlpcc(data_path=args.test_file)
    
    # alias_map,sub_map=read_kg(kg_path=args.kg_path,kg=args.kg_type)

    # model = MyBart.from_pretrained(args.bart_model_path,state_dict=torch.load(args.checkpoint))
    # model.to(device)
    # tokenizer=BertTokenizer.from_pretrained(args.bart_model_path)

    # predict_ner(data=data,model=model,tokenizer=tokenizer)
    # get_topic_entity(data=data)

    # tmp_ner_path='outputs/tmp_ner.json'
    # with open(tmp_ner_path,'w') as f:
    #     for example in data:
    #         f.write(json.dumps(example,ensure_ascii=False)+'\n')
    
    # evaluate_ner(tmp_ner_path,alias_map,sub_map,use_rule12=True)

    # ner_data=[]
    # with open(tmp_ner_path) as f:
    #     lines=f.readlines()
    #     for line in lines:
    #         ner_data.append(json.loads(line.strip()))

    # predict_relation(ner_data=ner_data,model=model,tokenizer=tokenizer,alias_map=alias_map,sub_map=sub_map)
    # evaluate_kgqa(data=ner_data,sub_map=sub_map)

    # tmp_kgqa_path='outputs/tmp_kgqa.json'

    # with open(tmp_kgqa_path,'w') as f:
    #     for example in ner_data:
    #         new_example={'id':example['id'],
    #                     "question":example['question'],
    #                     'answer':example['answer'],
    #                     'predict_ner':example['ner_pred'],
    #                     'predict_answer':list(example['relation_pred'])}
    #         f.write(json.dumps(new_example,ensure_ascii=False)+'\n')

    # evaluate_relation(tmp_kgqa_path)