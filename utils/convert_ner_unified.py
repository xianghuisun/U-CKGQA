import imp
import pandas as pd
import json
import argparse
import random
from utils import read_data, read_data_nlpcc
random.seed(42)

def convert_to_prompt(data_path,type_):
    if type_.lower()=='kgclue':
        examples=read_data(data_path)
    else:
        assert type_.lower()=='nlpcc'
        examples=read_data_nlpcc(data_path)
    

    data=[]
    bad_example=0
    for example in examples:
        text=example['question'].lower().replace('“','').replace('”','').replace('"','').replace('"','').replace('\n','').replace('\t','')
        text='{}'.format(text)+'。上个句子中哪些属于实体名词？_。'
        answer=example['answer']
        entity=answer.split('|||')[0].strip().lower()
        if type_.lower()=='kgclue':
            if '（' in entity and '）' in entity:
                entity=entity.split('（')[0]

        if entity not in text:
            bad_example+=1
        else:
            data.append([text,entity])

    print('Original examples: {}, {} examples cannot found entity in question, left {} examples to do unified ner'.format(len(examples),bad_example,len(data)))
    return data

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    ## As the dataset or task changes
    parser.add_argument("--original_train_data", type=str, default='/home/xhsun/Desktop/gitRepositories/graduation-project/code/data/kgclue/train_dev.json')
    parser.add_argument("--original_test_data", type=str, default='/home/xhsun/Desktop/gitRepositories/graduation-project/code/data/kgclue/test_public.json')
    parser.add_argument("--original_dev_data", type=str, default='/home/xhsun/Desktop/gitRepositories/graduation-project/code/data/kgclue/dev.json')
    parser.add_argument("--unified_train_data", type=str, default='/home/xhsun/Desktop/graduate_saved_files/section4/kgclue/ner/prompt_train.csv')
    parser.add_argument("--unified_dev_data", type=str, default='/home/xhsun/Desktop/graduate_saved_files/section4/kgclue/ner/prompt_dev.csv')
    parser.add_argument("--unified_test_data", type=str, default='/home/xhsun/Desktop/graduate_saved_files/section4/kgclue/ner/prompt_test.csv')
    parser.add_argument("--type_", type=str, default='kgclue')
    
    # parser.add_argument("--original_train_data", type=str, default='/home/xhsun/Desktop/gitRepositories/graduation-project/code/data/nlpcc2018/nlpcc-iccpol-2016.kbqa.training-data')
    # parser.add_argument("--original_test_data", type=str, default='/home/xhsun/Desktop/gitRepositories/graduation-project/code/data/nlpcc2018/nlpcc-iccpol-2016.kbqa.testing-data')
    # parser.add_argument("--original_dev_data", type=str, default='None')
    # parser.add_argument("--unified_train_data", type=str, default='/home/xhsun/Desktop/graduate_saved_files/section4/nlpcc/ner/prompt_train.csv')
    # parser.add_argument("--unified_dev_data", type=str, default='/home/xhsun/Desktop/graduate_saved_files/section4/nlpcc/ner/prompt_dev.csv')
    # parser.add_argument("--unified_test_data", type=str, default='/home/xhsun/Desktop/graduate_saved_files/section4/nlpcc/ner/prompt_test.csv')
    # parser.add_argument("--type_", type=str, default='nlpcc')
    args=parser.parse_args()
    # for key,value in args.__dict__.items():
    #     print(key,value)

    converted_train_examples=convert_to_prompt(data_path=args.original_train_data,type_=args.type_)
    converted_test_examples=convert_to_prompt(data_path=args.original_test_data,type_=args.type_)
    if args.original_dev_data=='None':
        random.shuffle(converted_train_examples)
        converted_dev_examples=converted_train_examples[-2000:]
        converted_train_examples=converted_train_examples[:-2000]
    else:
        converted_dev_examples=convert_to_prompt(data_path=args.original_dev_data,type_=args.type_)

    columns=['question','label']
    converted_train_pd=pd.DataFrame(converted_train_examples,columns=columns)
    converted_test_pd=pd.DataFrame(converted_test_examples,columns=columns)
    converted_dev_pd=pd.DataFrame(converted_dev_examples,columns=columns)

    converted_train_pd.to_csv(args.unified_train_data,index=None)
    converted_test_pd.to_csv(args.unified_test_data,index=None)
    converted_dev_pd.to_csv(args.unified_dev_data,index=None)