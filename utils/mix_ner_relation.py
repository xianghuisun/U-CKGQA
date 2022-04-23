import pandas as pd
import json
import random
import argparse
random.seed(42)

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    ## As the dataset or task changes
    parser.add_argument("--ner_train_data", type=str, default='/home/xhsun/Desktop/graduate_saved_files/section4/kgclue/ner/prompt_train.csv')
    parser.add_argument("--ner_dev_data", type=str, default='/home/xhsun/Desktop/graduate_saved_files/section4/kgclue/ner/prompt_dev.csv')
    parser.add_argument("--ner_test_data", type=str, default='/home/xhsun/Desktop/graduate_saved_files/section4/kgclue/ner/prompt_test.csv')

    parser.add_argument("--relation_train_data", type=str, default='/home/xhsun/Desktop/graduate_saved_files/section4/kgclue/relation/prompt_train.csv')
    parser.add_argument("--relation_dev_data", type=str, default='/home/xhsun/Desktop/graduate_saved_files/section4/kgclue/relation/prompt_dev.csv')
    parser.add_argument("--relation_test_data", type=str, default='/home/xhsun/Desktop/graduate_saved_files/section4/kgclue/relation/prompt_test.csv')

    parser.add_argument("--mixture_train_data", type=str, default='/home/xhsun/Desktop/graduate_saved_files/section4/kgclue/mixture/prompt_train.csv')
    parser.add_argument("--mixture_dev_data", type=str, default='/home/xhsun/Desktop/graduate_saved_files/section4/kgclue/mixture/prompt_dev.csv')
    parser.add_argument("--mixture_test_data", type=str, default='/home/xhsun/Desktop/graduate_saved_files/section4/kgclue/mixture/prompt_test.csv')


    # parser.add_argument("--ner_train_data", type=str, default='/home/xhsun/Desktop/graduate_saved_files/section4/nlpcc/ner/prompt_train.csv')
    # parser.add_argument("--ner_dev_data", type=str, default='/home/xhsun/Desktop/graduate_saved_files/section4/nlpcc/ner/prompt_dev.csv')
    # parser.add_argument("--ner_test_data", type=str, default='/home/xhsun/Desktop/graduate_saved_files/section4/nlpcc/ner/prompt_test.csv')

    # parser.add_argument("--relation_train_data", type=str, default='/home/xhsun/Desktop/graduate_saved_files/section4/nlpcc/relation/prompt_train.csv')
    # parser.add_argument("--relation_dev_data", type=str, default='/home/xhsun/Desktop/graduate_saved_files/section4/nlpcc/relation/prompt_dev.csv')
    # parser.add_argument("--relation_test_data", type=str, default='/home/xhsun/Desktop/graduate_saved_files/section4/nlpcc/relation/prompt_test.csv')

    # parser.add_argument("--mixture_train_data", type=str, default='/home/xhsun/Desktop/graduate_saved_files/section4/nlpcc/mixture/prompt_train.csv')
    # parser.add_argument("--mixture_dev_data", type=str, default='/home/xhsun/Desktop/graduate_saved_files/section4/nlpcc/mixture/prompt_dev.csv')
    # parser.add_argument("--mixture_test_data", type=str, default='/home/xhsun/Desktop/graduate_saved_files/section4/nlpcc/mixture/prompt_test.csv')

    args = parser.parse_args()


    relation_train=pd.read_csv(args.relation_train_data)
    relation_dev=pd.read_csv(args.relation_dev_data)
    relation_test=pd.read_csv(args.relation_test_data)

    ner_train=pd.read_csv(args.ner_train_data)
    ner_dev=pd.read_csv(args.ner_dev_data)
    ner_test=pd.read_csv(args.ner_test_data)


    kgqa_train=relation_train.values.tolist()+ner_train.values.tolist()
    kgqa_test=relation_test.values.tolist()+ner_test.values.tolist()
    kgqa_dev=relation_dev.values.tolist()+ner_dev.values.tolist()

    random.shuffle(kgqa_train)
    print(len(kgqa_train),len(kgqa_test),len(kgqa_dev))

    print(kgqa_train[:5])

    columns=['question','label']
    kgqa_train=pd.DataFrame(kgqa_train,columns=columns)
    kgqa_train.to_csv(args.mixture_train_data,index=None)

    kgqa_test=pd.DataFrame(kgqa_test,columns=columns)
    kgqa_test.to_csv(args.mixture_test_data,index=None)

    kgqa_dev=pd.DataFrame(kgqa_dev,columns=columns)
    kgqa_dev.to_csv(args.mixture_dev_data,index=None)