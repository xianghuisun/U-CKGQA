import pandas as pd
import argparse

def convert_to_prompt(sentences,labels):
    converted_examples=[]
    for s,l in zip(sentences,labels):
        s=s.replace('“','').replace('”','').replace('"','').replace('"','').replace('\n','').replace('\t','')
        sen='“{}”'.format(s)+' '+"这句评论的态度是什么？_。"
        label="积极" if str(l)=='1' else "消极"
        converted_examples.append([sen,label])
    return converted_examples


if __name__=="__main__":
    parser = argparse.ArgumentParser()

    ## As the dataset or task changes
    parser.add_argument("--original_train_data", type=str, default='/data/aisearch/nlp/data/xhsun/seq2seqUnify/data/CLS/original_data/weibo_sentiment_train.csv')
    parser.add_argument("--original_test_data", type=str, default='/data/aisearch/nlp/data/xhsun/seq2seqUnify/data/CLS/original_data/weibo_sentiment_test.csv')
    parser.add_argument("--unified_train_data", type=str, default='/data/aisearch/nlp/data/xhsun/seq2seqUnify/data/CLS/unified_data/prompt_train.csv')
    parser.add_argument("--unified_test_data", type=str, default='/data/aisearch/nlp/data/xhsun/seq2seqUnify/data/CLS/unified_data/prompt_test.csv')

    args = parser.parse_args()
    train_pd = pd.read_csv(args.original_train_data)
    test_pd = pd.read_csv(args.original_test_data)

    train_sentences,train_labels = train_pd['review'].values.tolist(), train_pd['label'].values.tolist()
    dev_sentences,dev_labels = test_pd['review'].values.tolist(), test_pd['label'].values.tolist()

    converted_train_examples=convert_to_prompt(sentences=train_sentences,labels=train_labels)
    converted_test_examples=convert_to_prompt(sentences=dev_sentences,labels=dev_labels)

    columns=['question','label']
    converted_train_pd=pd.DataFrame(converted_train_examples,columns=columns)
    converted_test_pd=pd.DataFrame(converted_test_examples,columns=columns)

    converted_test_pd.to_csv(args.unified_train_data,index=None)
    converted_train_pd.to_csv(args.unified_test_data,index=None)