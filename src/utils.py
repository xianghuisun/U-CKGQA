import pandas as pd
import json
from collections import defaultdict
from tqdm import tqdm
import random
from copy import deepcopy
import sys,os,torch
from rule_tools import check_has_en, rule1, rule1_for_find_ner, rule2_for_find_ner,rule3_for_find_ner, rule4_for_find_ner
import zhconv
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from bart import MyBart
from transformers import BertTokenizer

#有哪些可以构造数据的idea呢？这个idea必须有助于实体识别和关系选择
'''
sub_map['金庸']={('逝世日期', '2018年10月30日'), ('中文名', '查良镛'), ('别名', '金庸（笔名）'), ('毕业院校', '上海东吴大学法学院、剑桥大学'), ('国籍', '中国'), ('外文名', 'Louis Cha'), ('职业', '作家、政论家、社会活动家'), ('民族', '汉族'), ('主要成就', '大英帝国官佐勋章勋衔法国文化部法国艺术及文学司令勋衔法国政府荣誉军团骑士英国牛津大学董事会成员香港大紫荆勋章香港文学创作终身成就奖收起'), ('出生日期', '1924年3月10日（农历二月初六）'), ('代表作品', '射雕英雄传、神雕侠侣、倚天屠龙记、天龙八部、笑傲江湖、鹿鼎记、雪山飞狐、书剑恩仇录等'), ('主要成就', '大英帝国官佐勋章勋衔法国文化部法国艺术及文学司令勋衔法国政府荣誉军团骑士英国牛津大学董事会成员香港大紫荆勋章展开大英帝国官佐勋章勋衔法国文化部法国艺术及文学司令勋衔法国政府荣誉军团骑士英国牛津大学董事会成员香港大紫荆勋章香港文学创作终身成就奖收起'), ('出生地', '浙江省嘉兴市海宁市'), ('血型', 'O型')}
(1) 根据sub_map构造：金庸和查良镛的关系是：(1)逝世日期 (2)中文名 (3)别名 (4)毕业院校 ...
                    金庸和上海东吴大学法学院、剑桥大学的关系是：(1)逝世日期 (2)中文名 (3)别名 (4)毕业院校 ...
sp_map[('10号州际公路', '国家')]={'美国'}、sp_map[('007在里约热内卢', '对白语言')]={'英语'}、sp_map[('007在里约热内卢', '导演')]={'Michel Parbot'}
(2) 根据sp_map构造：10号州际公路的国家是
'''

def read_kg(kg_path='/data/aisearch/nlp/data/xhsun/KG/kgClue/Knowledge.txt'):
    '''
    sub_map的key是每一个实体，value是这个实体所有的(关系，尾实体)的集合，也就是说sub_map[ent]是实体ent的一级子图
    sp_map的key是每一个唯一的（实体，关系），value是答案
    '''
    with open(kg_path) as f:
        lines=f.readlines()

    print('The number of triples: {}'.format(len(lines)))

    sub_map = defaultdict(set)#每一个头实体作为key，对应的所有一跳路径内的(关系，尾实体)作为value
    sp_map=defaultdict(set)#每一个（头实体，关系）作为key，对应的仅能有一个答案

    alias_map=defaultdict(set)
    ent_to_relations=defaultdict(set)
    bad_line=0
    for i in tqdm(range(len(lines)),total=len(lines),unit='triple'):
        line=lines[i]
        l = line.strip().split('\t')
        s = l[0].strip()
        p = l[1].strip()
        o = l[2].strip()
        if s=='' or p=='' or o=='':
            bad_line+=1
            continue
        sub_map[s].add((p, o))
        sp_map[(s,p)].add(o)

        ent_to_relations[s].add(p)

        entity_mention=s
        if '（' in s and '）' in s:
            entity_mention=s.split('（')[0]
            alias_map[entity_mention].add(s)
        if p in ['别名','中文名','英文名','昵称','中文名称','英文名称','别称','全称','原名']:
            alias_map[entity_mention].add(o)
    
    return sub_map,alias_map,ent_to_relations



def read_test_data(test_file='/data/aisearch/nlp/data/xhsun/KG/QA_data/kgclue/official_data/test.json'):
    with open(test_file) as f:
        lines=f.readlines()
    test_examples=[]
    for line in lines:
        test_examples.append(json.loads(line.strip()))
    
    print("The number of test examples : {}".format(len(test_examples)))
    return test_examples

def load_model(bart_model_path='/data/aisearch/nlp/data/xhsun/huggingfaceModels/chinese/chinese-bart-base',
                checkpoint='/data/aisearch/nlp/data/xhsun/seq2seqUnify/saved_models/kgqa/mixture/best-model.pt'):
    tokenizer = BertTokenizer.from_pretrained(bart_model_path)
    model=MyBart.from_pretrained(bart_model_path,state_dict=torch.load(checkpoint,map_location='cpu'))
    return model,tokenizer

def evaluate_ner(test_examples,model,tokenizer,write_results_path='../outputs/ner_results.json',prompt='。上个句子中哪些属于实体名词？_。'):
    device='cuda' if torch.cuda.is_available() else 'cpu'
    print('Using device : {}'.format(device))
    model=model.to(device)
    model.eval()

    for i in tqdm(range(len(test_examples)),total=len(test_examples),unit='sentence'):
        example=test_examples[i]
        demo_q=example['question']
        demo_q='{}'.format(demo_q)+prompt
        test_examples[i].update({'input_q_ner':demo_q})

        with torch.no_grad():
            batch=tokenizer.batch_encode_plus([demo_q])
            outputs = model.generate(input_ids=torch.LongTensor(batch['input_ids']).to(device=device),
                                    attention_mask=torch.LongTensor(batch['attention_mask']).to(device=device),
                                    num_beams=4,
                                    min_length=1,
                                    max_length=30,
                                early_stopping=True,)
        ner_results=tokenizer.decode(outputs[0],skip_special_tokens=True,clean_up_tokenization_spaces=True)#.strip()
        test_examples[i].update({"pred_ner":ner_results})
    
    with open(write_results_path,'w') as f:
        for example in test_examples:
            f.write(json.dumps(example,ensure_ascii=False)+'\n')
    
    print('Evaluating ner has completed!!!')


def get_topic_entity(ner_results,sub_map,alias_map,write_results_path):
    '''
    有如下问题需要考虑：
    （1）实体名词全部是英文：
            question中是大写字母，但是由于tokenizer是uncased，所以输出的答案是小写字母
            由于是英文，所以不能进行''.join(xx.split())操作
    （2）实体名词全部是中文：
            可以直接进行''.join(xx.split())操作，需要注意繁体字的问题
    （3）实体名词中既有中文又有英文
    '''
    if '' in sub_map:
        del sub_map['']
    if '' in alias_map:
        del alias_map['']
    assert '' not in sub_map and '' not in alias_map

    wrong_ner_results=[]
    for i in range(len(ner_results)):
        example=ner_results[i]
        question=example['question']
        pred_ner=example['pred_ner']
        topic_entity=pred_ner
        if check_has_en(pred_ner):
            #有英文字母，可能全是英文字母也可能中英文混合
    #         if only_en_char(topic_entity):
    #             #全是英文字母
            if topic_entity not in question:
                #有英文字母但是这个实体不在问题中，那么可能存在大小写问题，应该以question中的字母为准
                temp_question=question.lower()
                temp_entity=topic_entity.lower()
                if temp_entity in temp_question:
                    start_idx=temp_question.find(temp_entity)
                    end_idx=start_idx+len(temp_entity)
                    topic_entity=question[start_idx:end_idx]

        #默认现在是仅有中文的情况
        if topic_entity not in question:
            topic_entity=''.join(topic_entity.split())
            
            if check_has_en(topic_entity) and topic_entity not in question:
                #此时说明既有中文又有英文，但是这个实体不在问题中，那么可能存在大小写问题，应该以question中的字母为准
                temp_question=question.lower()
                temp_entity=topic_entity.lower()
                if temp_entity in temp_question:
                    start_idx=temp_question.find(temp_entity)
                    end_idx=start_idx+len(temp_entity)
                    topic_entity=question[start_idx:end_idx]
            
            if topic_entity not in question and zhconv.convert(topic_entity,'zh-hans') in question:
                topic_entity=zhconv.convert(topic_entity,'zh-hans')
                
        if topic_entity not in question:
            #还是没有找到
            topic_entity=rule1(topic_entity=pred_ner,question=question)

        if topic_entity not in question:
            wrong_ner_results.append(example)

        ner_results[i].update({"topic_entity":topic_entity})

    not_found_in_question=0
    for example in ner_results:
        if example['topic_entity'] not in example['question']:
            not_found_in_question+=1
    assert not_found_in_question==len(wrong_ner_results)

    print("wrong_ner_results : {}".format(len(wrong_ner_results)))
    for example in wrong_ner_results:
        print(example)
        print('-'*100)

    ##################现在的阶段是，有4个识别的主题实体在question中根本找不到，2996个实体在question中可以找得到
    ##################但是有一些识别的实体不在KG中
    for i in range(len(ner_results)):
        example=ner_results[i]
        topic_entity=example['topic_entity']
        question=example['question']
        if topic_entity not in question:
            print("{} not in {}".format(topic_entity,question))
            continue
        if  topic_entity not in sub_map and topic_entity not in alias_map:
            topic_entity=rule1_for_find_ner(topic_entity=topic_entity,question=question,sub_map=sub_map,alias_map=alias_map)
        
        if  topic_entity not in sub_map and topic_entity not in alias_map:
            topic_entity=rule2_for_find_ner(topic_entity=topic_entity,question=question,sub_map=sub_map,alias_map=alias_map)
        
        if  topic_entity not in sub_map and topic_entity not in alias_map:
            topic_entity=rule3_for_find_ner(topic_entity=topic_entity,question=question,sub_map=sub_map,alias_map=alias_map)
        
        if  topic_entity not in sub_map and topic_entity not in alias_map:
            topic_entity=rule4_for_find_ner(topic_entity=topic_entity,question=question,sub_map=sub_map,alias_map=alias_map)
        
        ner_results[i].update({"topic_entity":topic_entity})
    
    not_found_in_kg=0
    for example in ner_results:
        if example['topic_entity'] not in sub_map and example['topic_entity'] not in alias_map:
            not_found_in_kg+=1
            print("example : not found entity in KG", example)

    print('not_found_in_kg : {}'.format(not_found_in_kg))
    
    with open(write_results_path,'w') as f:
        for example in ner_results:
            idx=example['id']
            topic_entity=example['topic_entity']
            question=example['question']
            f.write(json.dumps({"id":idx,'question':question,'topic_entity':topic_entity},ensure_ascii=False)+'\n')
    
    print('Entity has been written in {}'.format(write_results_path))

def get_inputs_for_step2(ner_results,alias_map,ent_to_relations):
    '''
    根据实体识别阶段识别的实体，找出这个实体的所有别名实体
    对于每一个别名实体，找出这个实体所有的一级关系，与该别名实体和问题构成一个example
    '''
    inputs_for_kg=[]

    for i in range(len(ner_results)):
        example=ner_results[i]
        idx=example['id']
        topic_entity=example['topic_entity']
        question=example['question']
        candidate_questions=[]
        alias_entities=list(alias_map[topic_entity])
        if topic_entity not in alias_entities:
            alias_entities.append(topic_entity)

        for ent in alias_entities:
            relations=ent_to_relations[ent]
            if relations==set():
                continue
            candidate_relations=list(relations)
            candidate_relations.append('不匹配')
            random.shuffle(candidate_relations)#确保“不匹配”这个选项不总是在最后一个位置

            relation_string=[]
            for i,rel in enumerate(candidate_relations):
                relation_string.append('（{}）'.format(i)+rel)
            relation_string='，'.join(relation_string)
            kg_q='{}'.format(question)+'。'+'实体名词是：{}'.format(ent)+"。这个句子的意图与下列哪一个关系最相似？_。"+' '+relation_string
            candidate_questions.append(kg_q)
            
        #test_examples[i].update({"candidate_questions":deepcopy(candidate_questions)})
        inputs_for_kg.append({"id":idx,'question':question,'topic_entity':topic_entity,'candidate_questions':candidate_questions})
    
    return inputs_for_kg

def evaluate_qa(test_examples,model,tokenizer,write_results_path,limit_q_nums=64):
    results=[]
    device='cuda' if torch.cuda.is_available() else 'cpu'
    print('Using device : {}'.format(device))
    model=model.to(device)
    model.eval()

    for i in tqdm(range(len(test_examples)),total=len(test_examples),unit='sentence'):
        example=test_examples[i]
        idx=example['id']
        topic_entity=example['topic_entity']
        question=example['question']
        candidate_questions=example['candidate_questions']
        assert type(candidate_questions)==list

        if candidate_questions==[]:
            #没有找到可以输入的候选句子
            candidate_combinations=[]
        else:
            candidate_questions=candidate_questions[:limit_q_nums]
            with torch.no_grad():
                batch=tokenizer.batch_encode_plus(candidate_questions,pad_to_max_length=True,max_length=384)
                outputs = model.generate(input_ids=torch.LongTensor(batch['input_ids']).to(device=device),
                                        attention_mask=torch.LongTensor(batch['attention_mask']).to(device=device),
                                        num_beams=4,
                                        min_length=1,
                                        max_length=30,
                                        early_stopping=True,)

                assert len(outputs)==len(candidate_questions)
                candidate_combinations=[]
                for output,input_q in zip(outputs,candidate_questions):
                    pred=tokenizer.decode(output,skip_special_tokens=True,clean_up_tokenization_spaces=True)#.strip()
                    candidate_combinations.append((input_q,pred))

        results.append({"id":idx,
                        'question':question,
                        'topic_entity':topic_entity,
                        'candidate_questions':candidate_questions,
                        'candidate_combinations':candidate_combinations})
    
    with open(write_results_path,'w') as f:
        for example in results:
            f.write(json.dumps(example,ensure_ascii=False)+'\n')
    
    print('Evaluating step2 has completed!!!')

'''
isbn,CAS登录号,EINECS登录号,人均GDP,IATA代码,ICAO代码,GDP总计,CAS号,IMDB评分,GDP,游戏ID,CV
'''

def get_submissions(qa_results,alias_map,sub_map):
    submit_examples=[]
    for i in range(len(qa_results)):
        example=qa_results[i]
        question=example['question']
        candidate_combinations=example['candidate_combinations']
        topic_entity=example['topic_entity']
        alias_entities=list(alias_map[topic_entity])
        if topic_entity not in alias_entities:
            alias_entities.append(topic_entity)
            
        predict_answers=[]
        for each_possible_answer in candidate_combinations:
            input_q_with_prompt,predict_r=each_possible_answer
            ent_in_kg=re.findall('实体名词是：(.*)。这个句子的意图与下列哪一个关系最相似？_。',input_q_with_prompt)[0]
            predict_r=''.join(predict_r.split())
            
            if predict_r.lower() in input_q_with_prompt.lower():
                start_idx=input_q_with_prompt.lower().find(predict_r.lower())
                end_idx=start_idx+len(predict_r)
                predict_r=input_q_with_prompt[start_idx:end_idx]
                
            if predict_r=='不匹配':
                continue
            if predict_r in input_q_with_prompt:
                #此时input_q_with_prompt中的实体名词和predict_r就构成了答案
                for p,o in sub_map[ent_in_kg]:
                    if p==predict_r:
                        break
                predict_answers.append((ent_in_kg,predict_r,o))
    #         if check_has_en(predict_r):
    #             #关系中有英文字母存在
        #print(question,predict_answers)
        submit_examples.append({'id':example['id'],'question':question,'topic_entity':topic_entity,'predict_answers':predict_answers})

    submissions=[]
    multiple_answers=[]
    not_found_answers=[]
    for example in submit_examples:
        predict_answers=example['predict_answers']
        topic_entity=example['topic_entity']
        
        if predict_answers!=[]:
            if len(predict_answers)>1:
                multiple_answers.append(example)
            predict_answer=predict_answers[0]
            s,p,o=predict_answer
            answer=' ||| '.join([s,p,o])
        else:
            not_found_answers.append(example)
            answer=' ||| '.join([topic_entity,topic_entity,topic_entity])
        submissions.append({"id":example['id'],'question':example['question'],'answer':answer})

    return submissions