import os,sys,json,re,zhconv
from matplotlib.pyplot import hist
import torch

import logging
logger=logging.getLogger('evaluate.tools')

def inference_ner(text,model,tokenizer,device=torch.device('cuda')):
    model_inputs=tokenizer([text],return_tensors='pt')
    input_ids=model_inputs['input_ids'].to(device)
    attention_mask=model_inputs['attention_mask'].to(device)
    outputs = model.generate(input_ids=input_ids,
                            attention_mask=attention_mask,
                            num_beams=4,
                            min_length=1,
                            max_length=30,
                            early_stopping=True,)#tensor([[ 102,  101, 3805, 2255, 3353, 7553,  102]], device='cuda:0')
    pred=tokenizer.decode(outputs[0],skip_special_tokens=True,clean_up_tokenization_spaces=True).strip()#'泰 山 极 顶'
    return pred


def inference_relation(text,model,tokenizer,max_length=384,device=torch.device('cuda')):
    model_inputs=tokenizer(text,return_tensors='pt',padding=True,max_length=max_length)
    input_ids=model_inputs['input_ids'].to(device)
    attention_mask=model_inputs['attention_mask'].to(device)
    outputs = model.generate(input_ids=input_ids,
                            attention_mask=attention_mask,
                            num_beams=4,
                            min_length=1,
                            max_length=30,
                            early_stopping=True,)
    return outputs


def select_span(tmp_q,question,ner_pred):
    start_id=tmp_q.find(ner_pred)
    end_id=start_id+len(ner_pred)
    new_pred=question[start_id:end_id]
    return new_pred

def rule_of_departure_to_destination(question,ner_pred):
    '''
    question='东营—深圳公路途径哪些城市'
    ner_pred=''东营深圳公路''
    '''
    ner_pred=''.join(ner_pred.split())
    try:
        
        if '—' in question:
            tmp_q=question.replace('—','')
            if ner_pred in tmp_q:
                start_token=ner_pred[:2]
                end_token=ner_pred[-2:]
                assert start_token in question and end_token in question
                start_pos=question.find(start_token)
                end_pos=question.find(end_token)+1
                entity=question[start_pos:end_pos+1]
                return entity
            else:
                return ''
        else:
            return ''
    
    except Exception as e:
        print(e)
        return ''

def rule_of_chinese_quotation_mark(question,ner_pred):
    '''
    question='请问“一带一路”铁路国际人才教育联盟是由哪个单位发起的？'
    ner_pred='一带一路铁路国际人才教育联盟'
    '''
    ner_pred=''.join(ner_pred.split())
    try:
        if '“' in question:
            assert '”' in question
            token_string=re.findall('“(.*)”',question)[0]
            tmp_q=question.replace('“','').replace('”','')
            if ner_pred in tmp_q:
                assert token_string in ner_pred
                new_pred=ner_pred.replace(token_string,'“{}”'.format(token_string))
                assert new_pred in question
                return new_pred
            else:
                return ''
        else:
            return ''
        
    except Exception as e:
        print(e)
        return ''
            
def rule_of_zhconv(question,ner_pred):
    '''
    question='中国共产党中央委员会政法委员会属于什么机构你知道吗？'
    ner_pred='中国共产党中央委员会政法委员會'
    '''
    ner_pred=''.join(ner_pred.split())
    if zhconv.convert(ner_pred,'zh-hans') in question:
        return zhconv.convert(ner_pred,'zh-hans')
    else:
        return ''
    
def final_rule(question,ner_pred):
    ner_pred=''.join(ner_pred.split())
    question_lower=question.lower()
    start_token=ner_pred[:2]
    end_token=ner_pred[-2:]
    
    if start_token in question_lower and end_token in question_lower:
        start_pos=question_lower.find(start_token)
        end_pos=question_lower.find(end_token)+1
        entity=question[start_pos:end_pos+1]
        return entity
    else:
        return ''

def rule1_for_find_ner(topic_entity,question,sub_map,alias_map):
    '''
    以topic_entity为起始点，往右边找
    '''
    assert topic_entity in question
    start_idx=question.find(topic_entity)
    end_idx=start_idx+len(topic_entity)
    for i in range(end_idx,len(question)):
        if question[start_idx:i] in sub_map or question[start_idx:i] in alias_map:
            topic_entity=question[start_idx:i]
            break
    
    return topic_entity

def rule2_for_find_ner(topic_entity,question,sub_map,alias_map):
    '''
    以topic_entity为终止点，往左边找
    '''
    assert topic_entity in question
    start_idx=question.find(topic_entity)
    end_idx=start_idx+len(topic_entity)
    for i in range(0,start_idx+1):
        if question[i:end_idx] in sub_map or question[i:end_idx] in alias_map:
            topic_entity=question[i:end_idx]
            break
    
    return topic_entity

def rule3_for_find_ner(topic_entity,question,sub_map,alias_map):
    '''
    以topic_entity中心，从左到右删除字符
    '''
    assert topic_entity in question
    for i in range(len(topic_entity)):
        if (topic_entity[i:] in sub_map or topic_entity[i:] in alias_map):
            topic_entity=topic_entity[i:]
            break
    return topic_entity

def rule4_for_find_ner(topic_entity,question,sub_map,alias_map):
    '''
    以topic_entity中心，从右到左删除字符
    '''
    assert topic_entity in question
    for i in range(len(topic_entity),1):
        if topic_entity[:i] in sub_map or topic_entity[:i] in alias_map:
            topic_entity=topic_entity[:i]
            break
    return topic_entity


def get_topic_entity(data):
    wrong_results=[]

    for example in data:
        question=example['question']
        question_lower=question.lower()
        ner_pred=example['ner_pred']
        if ner_pred in question:
            continue
        
        elif rule_of_chinese_quotation_mark(question,ner_pred=ner_pred)!='':
            new_pred=rule_of_chinese_quotation_mark(question,ner_pred=ner_pred)
            example.update({"ner_pred":new_pred})

        elif ner_pred in question_lower:
            new_pred=select_span(tmp_q=question_lower,question=question,ner_pred=ner_pred)
            example.update({"ner_pred":new_pred})

        elif ''.join(ner_pred.split()) in question:
            new_pred=select_span(tmp_q=question,question=question,ner_pred=''.join(ner_pred.split()))
            example.update({"ner_pred":new_pred})
            
        elif ''.join(ner_pred.split()) in question_lower:
            new_pred=select_span(tmp_q=question_lower,question=question,ner_pred=''.join(ner_pred.split()))
            example.update({"ner_pred":new_pred})
            
        elif rule_of_departure_to_destination(question,ner_pred=ner_pred)!='':
            new_pred=rule_of_departure_to_destination(question,ner_pred=ner_pred)
            example.update({"ner_pred":new_pred})
        
        elif rule_of_zhconv(question,ner_pred=ner_pred)!='':
            new_pred=rule_of_zhconv(question,ner_pred=ner_pred)
            example.update({"ner_pred":new_pred})
            
        elif final_rule(question,ner_pred=ner_pred)!='':
            new_pred=final_rule(question,ner_pred=ner_pred)
            example.update({"ner_pred":new_pred})
            
        else:
            wrong_results.append(example)
    
    print("The following examples is wrong ner results, predicted ner span can not found in question: ")
    for e in wrong_results:
        print(e)
        print('-'*100)
    print("Those examples has {}".format(len(wrong_results)))
    

def evaluate_ner(tmp_ner_path,alias_map,sub_map,use_rule12=True):
    if '' in alias_map:
        del alias_map['']
    if '' in sub_map:
        del sub_map['']
    ner_data=[]
    TP=0
    FP=0
    FN=0
    FP_examples=[]
    FN_examples=[]
    with open(tmp_ner_path) as f:
        lines=f.readlines()
        for line in lines:
            ner_data.append(json.loads(line.strip()))

    for example in ner_data:
        answer=example['answer']
        question=example['question']
        
        topic_entity=answer.split('|||')[0].strip()
        if ('（' in topic_entity and '）' in topic_entity):
            topic_entity=topic_entity.split('（')[0]

        try:
            assert topic_entity in question
        except:
            print(example,topic_entity)
            raise Exception("check")
        ner_pred=example['ner_pred']
        if ner_pred==topic_entity:
            TP+=1
        else:
            if ner_pred not in question:
                FN+=1#is not a true entity
                FN_examples.append(example)
                continue

            if use_rule12:
                # The following rules used for finding entity in KG, 
                # The rule in get_topic entity are used for finding span in question, not related to KG
                ner_pred_rule1=rule1_for_find_ner(topic_entity=ner_pred,question=question,sub_map=sub_map,alias_map=alias_map)
                ner_pred_rule2=rule2_for_find_ner(topic_entity=ner_pred,question=question,sub_map=sub_map,alias_map=alias_map)
                ner_pred_rule3=rule3_for_find_ner(topic_entity=ner_pred,question=question,sub_map=sub_map,alias_map=alias_map)
                ner_pred_rule4=rule4_for_find_ner(topic_entity=ner_pred,question=question,sub_map=sub_map,alias_map=alias_map)
            else:
                ner_pred_rule1=ner_pred
                ner_pred_rule2=ner_pred
                ner_pred_rule3=ner_pred
                ner_pred_rule4=ner_pred
            
            if ner_pred_rule1 == topic_entity or ner_pred_rule2 == topic_entity or ner_pred_rule3 == topic_entity or ner_pred_rule4 == topic_entity:
                TP+=1
                if ner_pred_rule1 == topic_entity:
                    ner_pred=ner_pred_rule1
                if ner_pred_rule2 == topic_entity:
                    ner_pred=ner_pred_rule2
                if ner_pred_rule3 == topic_entity:
                    ner_pred=ner_pred_rule3
                if ner_pred_rule4 == topic_entity:
                    ner_pred=ner_pred_rule4
            else:
                if ner_pred in alias_map or ner_pred in sub_map:
                    FP+=1#is a true entity in KG but not current example
                    FP_examples.append(example)
                else:
                    FN+=1#is not a true entity. Original
                    FN_examples.append(example)

        example.update({"ner_pred":ner_pred})

    with open(tmp_ner_path,'w') as f:
        for example in ner_data:
            f.write(json.dumps(example,ensure_ascii=False)+'\n')

    logger.info("="*100)
    logger.info("TP: {}, FP: {}, FN: {}".format(TP,FP,FN))
    recall=TP/(TP+FN)
    precision=TP/(TP+FP)
    f1=2*recall*precision/(recall+precision)
    logger.info("recall: {}, precision: {}, f1: {}".format(recall,precision,f1))
    logger.info("="*100)

    logger.info("Following examples are False Negative examples............")
    for e in FN_examples:
        logger.info(json.dumps(e,ensure_ascii=False))
    logger.info("Following examples are False Positive examples............")
    for e in FP_examples:
        logger.info(json.dumps(e,ensure_ascii=False))


def evaluate_kgqa(data,sub_map):
    for example in data:
        question=example['question']
        ner_pred=example['ner_pred']
        all_alias_rel_preds=example['rel_pred']
        
        all_predict_rels=set()
        for each_alias_pred in all_alias_rel_preds:
            input_q_with_prompt,predict_r=each_alias_pred
            ent_in_kg=re.findall('实体名词是：(.*)。这个句子的意图与下列哪一个关系最相似？_。',input_q_with_prompt)[0]
            
            if ''.join(predict_r.split())=='不匹配':
                continue
            
            if predict_r.lower() not in input_q_with_prompt.lower():
                predict_r=''.join(predict_r.split())
                
            if predict_r.lower() in input_q_with_prompt.lower():
                start_idx=input_q_with_prompt.lower().find(predict_r.lower())
                end_idx=start_idx+len(predict_r)
                predict_r=input_q_with_prompt[start_idx:end_idx]
                for p,o in sub_map[ent_in_kg]:
                    if p==predict_r:
                        all_predict_rels.add(' ||| '.join([ent_in_kg,p,o]))
                        break            
                
            else:
                #Can not found predicted relation in question
                logger.info("Cannot found {} in {}".format(predict_r,input_q_with_prompt))
                
        example.update({"relation_pred":all_predict_rels})

    correct=0
    wrong_examples=[]
    for example in data:
        answer=example['answer']
        gold_h,gold_r,gold_t=answer.split('|||')
        gold_h=gold_h.strip()
        gold_r=gold_r.strip()
        relation_pred=example['relation_pred']

        find_answer=False
        for pred_answer in relation_pred:
            pred_h,pred_r,pred_t=pred_answer.split('|||')
            pred_h=pred_h.strip()
            pred_r=pred_r.strip()
            if pred_h==gold_h and pred_r==gold_r:
                find_answer=True
                break
        if find_answer:
            correct+=1
        else:
            wrong_examples.append(example)

    hitsat1=correct/len(data)
    logger.info("="*100)
    logger.info("Hits@1 : {}".format(hitsat1))
    logger.info("="*100)

    return hitsat1


def evaluate_relation(tmp_kgqa_path):
    kgqa_results=[]
    with open(tmp_kgqa_path) as f:
        lines=f.readlines()
    for line in lines:
        kgqa_results.append(json.loads(line.strip()))

    logger.info("Relation evaluate results!!!")
    TP=0
    FP=0
    FN=0
    FN_examples=[]
    FP_examples=[]
    for example in kgqa_results:
        answer=example['answer']
        predict_answers=example['predict_answer']
        gold_h,gold_r,gold_t=answer.split('|||')
        gold_h=gold_h.strip()
        gold_r=gold_r.strip()
        
        if predict_answers==[]:
            FN+=1
            FN_examples.append(example)
            continue
            
        for pred_answer in predict_answers:
            pred_h,pred_r,pred_t=pred_answer.split('|||')
            pred_h=pred_h.strip()
            pred_r=pred_r.strip()
            
            if gold_r==pred_r:
                TP+=1
            else:
                FP+=1
                FP_examples.append(example)


    logger.info("="*100)
    logger.info("TP: {}, FP: {}, FN: {}".format(TP,FP,FN))
    recall=TP/(TP+FN)
    precision=TP/(TP+FP)
    f1=2*recall*precision/(recall+precision)
    logger.info("recall: {}, precision: {}, f1: {}".format(recall,precision,f1))
    logger.info("="*100)

    logger.info("Following examples are False Negative examples(predict is [])............")
    for e in FN_examples:
        logger.info(json.dumps(e,ensure_ascii=False))
    logger.info("Following examples are False Positive examples(multiple predictions)............")
    for e in FP_examples:
        logger.info(json.dumps(e,ensure_ascii=False))

    new_FP=0
    question2answers={}
    for e  in FP_examples:
        question2answers[e['question']]={'answer':e['answer'],'predicts':e['predict_answer']}

    new_FP=0
    for e,answers in question2answers.items():
        predicts=answers['predicts']
        answer=answers['answer']
        gold_h,gold_r,gold_t=answer.split('|||')
        gold_h=gold_h.strip()
        gold_r=gold_r.strip()
        found=False
        for pred_answer in predicts:
            pred_h,pred_r,pred_t=pred_answer.split('|||')
            pred_h=pred_h.strip()
            pred_r=pred_r.strip()
            if gold_r==pred_r:
                found=True
                break
        if found==False:
            new_FP+=1

    logger.info("="*100)
    logger.info("TP: {}, new_FP: {}, FN: {}".format(TP,new_FP,FN))
    recall=TP/(TP+FN)
    precision=TP/(TP+new_FP)
    f1=2*recall*precision/(recall+precision)
    logger.info("recall: {}, precision: {}, f1: {}".format(recall,precision,f1))
    logger.info("="*100)