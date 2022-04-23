from utils import read_data,read_kg
import pandas as pd
import json
from collections import defaultdict
from tqdm import tqdm
import random
import argparse

tag2attrs={"组织机构":['成立日期','创始人','总部地点','员工数','成立时间','公司类型','公司名称','创办时间','创建时间','创立','所属地区','机构特点','建立时间','机构地址'],
"人名":['国籍','毕业院校','职业','出生日期','专业方向','身高','民族','血型','体重','逝世日期','性别'],
"音乐":['填词','音乐风格','所属专辑','歌曲原唱','歌曲时长','编曲','谱曲','歌手','专辑'],
"节日":['节日类型','节日意义','节日起源','节日饮食','节日活动'],
"地名":['修建时间','所在地点','著名景点','地理位置','面积','人口数量','重建时间','营业时间','人口数','总人口'],
"书籍":['装帧','出版社','丛书','作者','出版时间','书名'],
"娱乐，影视节目":['制片人','播出时间','首播时间','片长','对白语言','上映时间','主演','票房','编剧','制片地区','首演时间','集数','播出时长'],
"游戏":['游戏类型','游戏画面','游戏引擎','游戏平台','游戏类别','所属游戏','游戏大小'],
"化学药品":['用法用量'],
"疾病":['症状','传染性','相关疾病']
}

def convert_to_prompt(examples,sub_map):
    ner_examples=[]
    for example in examples:
        triple=example['answer']
        head,rel,tail=triple.split('|||')
        head=head.strip()
        try:
            assert head in sub_map
            relations=[r for r,t in sub_map[head]]
            example.update({"relations":relations})
            ner_examples.append(example)
        except:
            print(head,example)
            #raise Exception("check")
            continue
        
    ner_examples2=[]
    for example in ner_examples:
        triple=example['answer']
        #head,rel,tail=triple.split('|||')
        relations=example['relations']
        tag='O'
        for rel in relations:
            found_each_rel=False
            for ner_tag,tag_attrs in tag2attrs.items():
                if rel in tag_attrs:
                    found_each_rel=True
                    break
            if found_each_rel:
                tag=ner_tag
                break
        example.update({'tag':tag})
        ner_examples2.append(example)
    
    outputs=[]
    filtered_count=0
    for example in ner_examples2:
        question=example['question']
        triple=example['answer']
        tag=example['tag']
        head=triple.split('|||')[0].strip()
        if '（' in head and '）' in head:
            head=head.split('（')[0]
        if head not in question:
            filtered_count+=1
            continue
        assert head in question
        for tag_ner in tag2attrs.keys():
            input_=question+'。这个句子中有哪些属于{}？'.format(tag_ner)
            output_=head if tag_ner==tag else "空实体"
            outputs.append([input_,output_])
        if tag=='O':
            outputs.append([question+'。这个句子中有哪些属于{}？'.format('普通实体'),head])
        else:
            outputs.append([question+'。这个句子中有哪些属于{}？'.format('普通实体'),'空实体'])
    print(filtered_count)
    return outputs

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--original_train_data", type=str, default='../../data/kgclue/train.json')
    parser.add_argument("--original_test_data", type=str, default='../../data/kgclue/test_public.json')
    parser.add_argument("--unified_train_data", type=str, default='/home/xhsun/Desktop/graduate_saved_files/section4/kgclue/prompt_train.csv')
    parser.add_argument("--unified_test_data", type=str, default='/home/xhsun/Desktop/graduate_saved_files/section4/kgclue/prompt_test.csv')
    parser.add_argument("--kg_path", type=str, default='/home/xhsun/NLP/KGQA/KG/kgCLUE/Knowledge.txt')

    args=parser.parse_args()

    alias_map,sub_map=read_kg(args.kg_path,kg='kgclue')
    test_examples=read_data(args.original_test_data)
    #dev_examples=read_data('../../data/kgclue/dev.json')
    train_examples=read_data(args.original_train_data)

    train_unified_examples=convert_to_prompt(examples=train_examples,sub_map=sub_map)
    #dev_unified_examples=convert_to_prompt(examples=dev_examples,sub_map=sub_map)
    test_unified_examples=convert_to_prompt(examples=test_examples,sub_map=sub_map)

    print("The number of training examples: {}".format(len(train_unified_examples)))
    print("The number of test examples: {}".format(len(test_unified_examples)))
    #print("The number of dev examples: {}".format(len(dev_unified_examples)))

    columns=['question','label']
    train_pd=pd.DataFrame(train_unified_examples,columns=columns)
    test_pd=pd.DataFrame(test_unified_examples,columns=columns)
    #dev_pd=pd.DataFrame(dev_unified_examples,columns=columns)

    train_pd.to_csv(args.unified_train_data,index=None)
    test_pd.to_csv(args.unified_test_data,index=None)
    #dev_pd.to_csv(args.unified_dev_data,index=None)