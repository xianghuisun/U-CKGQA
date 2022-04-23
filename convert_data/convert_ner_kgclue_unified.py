import pandas as pd
import json
import argparse

def convert_to_prompt(data_path):
    with open(data_path, "r", encoding='utf-8') as f:
        lines=f.readlines()
    data=[]
    bad_example=0
    for idx,line in enumerate(lines):
        example=json.loads(line.strip())
        text=example['question'].lower().replace('“','').replace('”','').replace('"','').replace('"','').replace('\n','').replace('\t','')
        text='{}'.format(text)+'。上个句子中哪些属于实体名词？_。'
        answer=example['answer']
        entity=answer.split('|||')[0].strip().lower()
        if '（' in entity and '）' in entity:
            entity=entity.split('（')[0]
        if entity not in text:
            bad_example+=1
        else:
            data.append([text,entity])
    print(bad_example,len(data),len(lines))
    return data

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    ## As the dataset or task changes
    parser.add_argument("--original_train_data", type=str, default='/data/aisearch/nlp/data/xhsun/KG/QA_data/kgclue/official_data/train.json')
    parser.add_argument("--original_test_data", type=str, default='/data/aisearch/nlp/data/xhsun/KG/QA_data/kgclue/official_data/test_public.json')
    parser.add_argument("--unified_train_data", type=str, default='/data/aisearch/nlp/data/xhsun/seq2seqUnify/data/KGQA/NER/prompt_train.csv')
    parser.add_argument("--unified_test_data", type=str, default='/data/aisearch/nlp/data/xhsun/seq2seqUnify/data/KGQA/NER/prompt_test.csv')

    args=parser.parse_args()
    # for key,value in args.__dict__.items():
    #     print(key,value)

    converted_train_examples=convert_to_prompt(data_path=args.original_train_data)
    converted_test_examples=convert_to_prompt(data_path=args.original_test_data)

    columns=['question','label']
    converted_train_pd=pd.DataFrame(converted_train_examples,columns=columns)
    converted_test_pd=pd.DataFrame(converted_test_examples,columns=columns)

    converted_train_pd.to_csv(args.unified_train_data,index=None)
    converted_test_pd.to_csv(args.unified_test_data,index=None)