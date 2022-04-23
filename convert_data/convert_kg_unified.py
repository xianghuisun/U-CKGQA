import pandas as pd
import json
from collections import defaultdict
from tqdm import tqdm
import random
import argparse

def read_data(data_path):
    with open(data_path) as f:
        lines=f.readlines()
    data=[]
    for line in lines:
        example=json.loads(line)
        data.append({"text":example['question'],'answer':example['answer']})
    print("The number of original examples: {}".format(len(data)))
    return data

def convert_to_prompt(args,demo_data):
    with open(args.kg_path) as f:
        lines=f.readlines()

    print('The number of triples: {}'.format(len(lines)))

    sub_map = defaultdict(set)#每一个头实体作为key，对应的所有一跳路径内的(关系，尾实体)作为value
    so_map=defaultdict(set)#每一对(头实体，尾实体)作为key，对应的所有一跳的关系作为value
    sp_map=defaultdict(set)#每一个（头实体，关系）作为key，对应的仅能有一个答案

    alias_map=defaultdict(set)
    ent_to_relations=defaultdict(set)
    bad_line=0
    for i in tqdm(range(len(lines))):
        line=lines[i]
        l = line.strip().split('\t')
        s = l[0].strip()
        p = l[1].strip()
        o = l[2].strip()
        if s=='' or p=='' or o=='':
            bad_line+=1
            continue
        sub_map[s].add((p, o))
        so_map[(s,o)].add(p)
        sp_map[(s,p)].add(o)
        
        ent_to_relations[s].add(p)
        
        entity_mention=s
        if '（' in s and '）' in s:
            entity_mention=s.split('（')[0]
            alias_map[entity_mention].add(s)
        if p in ['别名','中文名','英文名','昵称','中文名称','英文名称','别称','全称','原名']:
            alias_map[entity_mention].add(o)

    unified_examples=set()
    string_not_match=[]
    positive_nums=0
    negative_nums=0
    print('The number of qa pairs: {}'.format(len(demo_data)))

    for example in tqdm(demo_data,total=len(demo_data),unit='example'):
        text=example['text'].lower()#.replace('“','').replace('”','').replace('"','').replace('"','').replace('\n','').replace('\t','')
        triple=example['answer']
        h,r,t=triple.split('|||')
        h=h.strip()
        r=r.strip()
        t=t.strip()
        
        entity=h.lower()
        if '（' in h and '）' in h:
            entity=entity.split('（')[0]
        try:
            assert entity in text
        except:
            string_not_match.append(example)
        #对于每一个问题，找到问题中的主题实体，然后找到该主题实体的所有别名实体，每一个别名实体与其相连的所有关系与问题构成一个example
        #如果这个别名实体有一个关系与问答对中的关系匹配，那么这个别名实体构成的样本就是正样本
        for ent in alias_map[entity]:
            #找到这个主题实体对应的所有别名实体
            relations=ent_to_relations[ent]
            #获取这个别名实体的所有关系
            if relations==set():
                continue
            candidate_relations=list(relations)
            candidate_relations.append('不匹配')
            random.shuffle(candidate_relations)#确保“不匹配”这个选项不总是在最后一个位置
            
            relation_string=[]
            for i,rel in enumerate(candidate_relations):
                relation_string.append('（{}）'.format(i)+rel)

            relation_string='，'.join(relation_string)
            question='{}'.format(text)+'。'+'实体名词是：{}'.format(ent)+"。这个句子的意图与下列哪一个关系最相似？_。"+' '+relation_string
            #问题文本+链接的别名实体+别名实体对应的所有关系构成输入，输出则是从不匹配和正确关系中选择
            if r in relation_string:
                #别名实体的所有关系中存在着与真实意图对应的关系，那么这个样本就是正样本
                answer=r
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
    parser.add_argument("--original_train_data", type=str, default='/data/aisearch/nlp/data/xhsun/KG/QA_data/kgclue/official_data/train.json')
    parser.add_argument("--original_test_data", type=str, default='/data/aisearch/nlp/data/xhsun/KG/QA_data/kgclue/official_data/test_public.json')
    parser.add_argument("--unified_train_data", type=str, default='/data/aisearch/nlp/data/xhsun/seq2seqUnify/data/KGQA/QA/prompt_train.csv')
    parser.add_argument("--unified_test_data", type=str, default='/data/aisearch/nlp/data/xhsun/seq2seqUnify/data/KGQA/QA/prompt_test.csv')
    parser.add_argument("--kg_path", type=str, default='/data/aisearch/nlp/data/xhsun/KG/kgClue/Knowledge.txt')

    args = parser.parse_args()
    train_data=read_data(args.original_train_data)
    test_data=read_data(args.original_test_data)
    train_unified_examples=convert_to_prompt(args=args,demo_data=train_data)
    test_unified_examples=convert_to_prompt(args=args,demo_data=test_data)

    print("The number of training examples: {}".format(len(train_unified_examples)))
    print("The number of test examples: {}".format(len(test_unified_examples)))

    columns=['question','label']
    train_pd=pd.DataFrame(train_unified_examples,columns=columns)
    test_pd=pd.DataFrame(test_unified_examples,columns=columns)

    train_pd.to_csv(args.unified_train_data,index=None)
    test_pd.to_csv(args.unified_test_data,index=None)

