import pandas as pd
import json
from collections import defaultdict
from tqdm import tqdm
import random
import argparse
from utils import read_data_nlpcc, read_data, read_kg
random.seed(42)

def convert_to_prompt(args,demo_data,alias_map,sub_map):

    unified_examples=set()
    string_not_match=[]
    positive_nums=0
    negative_nums=0
    print('The number of qa pairs: {}'.format(len(demo_data)))

    for example in tqdm(demo_data,total=len(demo_data),unit='example'):
        text=example['question'].lower()#.replace('“','').replace('”','').replace('"','').replace('"','').replace('\n','').replace('\t','')
        triple=example['answer']
        h,r,t=triple.split('|||')
        h=h.strip()
        r_true=r.strip()
        t=t.strip()
        
        entity=h.lower()
        if args.kg_type.lower()=='kgclue' and ('（' in h and '）' in h):
            entity=entity.split('（')[0]
        if args.kg_type.lower()=='nlpcc' and ('(' in h and ')' in h):
            entity=entity.split('(')[0]
        try:
            assert entity in text
        except:
            string_not_match.append(example)
        #对于每一个问题，找到问题中的主题实体，然后找到该主题实体的所有别名实体，每一个别名实体与其相连的所有关系与问题构成一个example
        #如果这个别名实体有一个关系与问答对中的关系匹配，那么这个别名实体构成的样本就是正样本
        for ent in alias_map[entity]:
            #找到这个主题实体对应的所有别名实体
            relations=[]
            for r,t in sub_map[ent]:
                relations.append(r)
            #获取这个别名实体的所有关系
            if relations==[]:
                continue
            candidate_relations=relations
            candidate_relations.append('不匹配')
            random.shuffle(candidate_relations)#确保“不匹配”这个选项不总是在最后一个位置
            
            relation_string=[]
            for i,rel in enumerate(candidate_relations):
                relation_string.append('（{}）'.format(i)+rel)

            relation_string='，'.join(relation_string)
            question='{}'.format(text)+'。'+'实体名词是：{}'.format(ent)+"。这个句子的意图与下列哪一个关系最相似？_。"+' '+relation_string
            #问题文本+链接的别名实体+别名实体对应的所有关系构成输入，输出则是从不匹配和正确关系中选择
            if r_true in relation_string:
                #别名实体的所有关系中存在着与真实意图对应的关系，那么这个样本就是正样本
                answer=r_true
                positive_nums+=1
    #             try:
    #                 assert (ent,answer) in sp_map#容易出现的情况是：<姚明（男篮主席），身高>存在，但是<姚明（作家），身高>不存在
    #                 #然而此时的triple恰好是：<姚明（男篮主席），身高，2.26米>而ent是姚明（作家），但是姚明（作家）这个实体恰好有身高这个关系
    #                 #那么上述的断言则会出现错误，但其实这应该认为是正例才对
    #                 positive_nums+=1
    #                 #确保数据构造的正确性，即只要能找到别名实体链接到了知识库中，那么给定关系后就一定能够找到答案
    #             except:
    #                 answer_not_found.append((ent,answer,example))
    #                 answer='不匹配'
    #                 negative_nums+=1
    #                 #raise Exception("bad case")
            else:
                answer='不匹配'
                negative_nums+=1
                
            unified_examples.add((question,answer))
                
    print('Can not found entity in question text due to string dismatch',len(string_not_match))
    print('Positive and negative examples {},{}'.format(positive_nums,negative_nums))

    unified_examples=list(unified_examples)
    return unified_examples

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    ## As the dataset or task changes
    parser.add_argument("--original_train_data", type=str, default='/home/xhsun/Desktop/gitRepositories/graduation-project/code/data/kgclue/train_dev.json')
    parser.add_argument("--original_test_data", type=str, default='/home/xhsun/Desktop/gitRepositories/graduation-project/code/data/kgclue/test_public.json')
    parser.add_argument("--original_dev_data", type=str, default='/home/xhsun/Desktop/gitRepositories/graduation-project/code/data/kgclue/dev.json')
    parser.add_argument("--unified_train_data", type=str, default='/home/xhsun/Desktop/graduate_saved_files/section4/kgclue/relation/prompt_train.csv')
    parser.add_argument("--unified_test_data", type=str, default='/home/xhsun/Desktop/graduate_saved_files/section4/kgclue/relation/prompt_test.csv')
    parser.add_argument("--unified_dev_data", type=str, default='/home/xhsun/Desktop/graduate_saved_files/section4/kgclue/relation/prompt_dev.csv')
    parser.add_argument("--kg_path", type=str, default='/home/xhsun/NLP/KGQA/KG/kgCLUE/Knowledge.txt')
    parser.add_argument("--kg_type", type=str, default='kgclue')
    
    
    # parser.add_argument("--original_train_data", type=str, default='/home/xhsun/Desktop/gitRepositories/graduation-project/code/data/nlpcc2018/train.txt')
    # parser.add_argument("--original_test_data", type=str, default='/home/xhsun/Desktop/gitRepositories/graduation-project/code/data/nlpcc2018/test.txt')
    # parser.add_argument("--original_dev_data", type=str, default='None')
    # parser.add_argument("--unified_train_data", type=str, default='/home/xhsun/Desktop/graduate_saved_files/section4/nlpcc/relation/prompt_train.csv')
    # parser.add_argument("--unified_dev_data", type=str, default='/home/xhsun/Desktop/graduate_saved_files/section4/nlpcc/relation/prompt_dev.csv')
    # parser.add_argument("--unified_test_data", type=str, default='/home/xhsun/Desktop/graduate_saved_files/section4/nlpcc/relation/prompt_test.csv')
    # parser.add_argument("--kg_path", type=str, default='/home/xhsun/NLP/KGQA/KG/nlpcc2018/knowledge/nlpcc-iccpol-2016.kbqa.kb')
    # parser.add_argument("--kg_type", type=str, default='nlpcc')
    args = parser.parse_args()

    if args.kg_type=='kgclue':
        train_data=read_data(args.original_train_data)
        test_data=read_data(args.original_test_data)
        dev_data=read_data(args.original_dev_data)
    else:
        assert args.kg_type=='nlpcc'
        train_data=read_data_nlpcc(args.original_train_data)
        test_data=read_data_nlpcc(args.original_test_data)
        random.shuffle(train_data)
        dev_data=train_data[-2000:]
        train_data=train_data[:-2000]      

    alias_map,sub_map=read_kg(kg_path=args.kg_path,kg=args.kg_type)
    train_unified_examples=convert_to_prompt(args=args,demo_data=train_data,alias_map=alias_map,sub_map=sub_map)
    test_unified_examples=convert_to_prompt(args=args,demo_data=test_data,alias_map=alias_map,sub_map=sub_map)
    dev_unified_examples=convert_to_prompt(args=args,demo_data=dev_data,alias_map=alias_map,sub_map=sub_map)

    print("The number of training examples: {}".format(len(train_unified_examples)))
    print("The number of test examples: {}".format(len(test_unified_examples)))

    columns=['question','label']
    train_pd=pd.DataFrame(train_unified_examples,columns=columns)
    test_pd=pd.DataFrame(test_unified_examples,columns=columns)
    dev_pd=pd.DataFrame(dev_unified_examples,columns=columns)

    train_pd.to_csv(args.unified_train_data,index=None)
    test_pd.to_csv(args.unified_test_data,index=None)
    dev_pd.to_csv(args.unified_dev_data,index=None)

