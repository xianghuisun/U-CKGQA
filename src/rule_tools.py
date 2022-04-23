from ast import alias
import re,string
from tracemalloc import start

def check_has_en(s):
    my_re = re.compile(r'[A-Za-z]',re.S)
    res = re.findall(my_re,s)
    if len(res):
        return True
    else:
        return False

def only_en_char(s):
    for char in s:
        if char not in list(string.ascii_letters)+[' ']:
            return False
    return True

def rule1(topic_entity,question):
    token_list=topic_entity.split()
    start_token=token_list[0]
    end_token=token_list[-1]
    
    temp_question=question.lower()
    start_token=start_token.lower()
    end_token=end_token.lower()
    start_idx=temp_question.find(start_token)
    end_idx=temp_question.find(end_token)
    
    if start_idx!=-1 and end_idx!=-1:
        topic_entity=question[start_idx:end_idx+1]
    return topic_entity


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
        if topic_entity[i:] in sub_map or topic_entity[i:] in alias_map:
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


