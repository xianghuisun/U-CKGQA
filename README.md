# U-CKGQA
Unified-modeling for Chinese Knowledge Graph Question Answering


```bash
pip install -r requirements.txt
```

## 下载知识图谱和预训练语言模型
```bash
mkdir bart-base-chinese
mkdir knowledge_graph
```

### 下载预训练语言模型
[模型地址](https://huggingface.co/fnlp/bart-base-chinese/tree/main)

将下载的文件存放在chinese-roberta-wwm-ext文件夹下，确保chinese-roberta-wwm-ext文件夹含有如下4个文件：
- config.json
- pytorch_model.bin
- tokenizer.json
- vocab.txt

### 下载知识图谱和问答数据
[问答数据地址](https://github.com/CLUEbenchmark/KgCLUE/tree/main/datasets)
[知识图谱地址](https://github.com/CLUEbenchmark/KgCLUE#%E6%95%B0%E6%8D%AE%E9%9B%86%E4%BB%8B%E7%BB%8D)
将下载的文件存放在knowledge_graph文件夹下，确保knowledge_graph文件夹含有如下文件
- Knowledge.txt
- train.json
- dev.json
- test_public.json

## 训练U-CKGQA

### Step1. 将要训练的数据转成text-to-text格式

```bash
cd utils
python convert_ner_unified.py
python convert_kg_unified.py
python mix_ner_relation.py
```
convert_ner_unified.py脚本的作用是根据原始数据构造text2text形式的NER数据。

convert_kg_unified.py脚本的作用是根据原始数据构造text2text形式关系预测数据。
mix_ner_relation.py脚本的作用是将上面两个任务的数据集混合。

以上三个文件中的路径需要指定本地的文件，比如knowledge_graph
### Step2. 训练模型
```bash
python main.py
```
main.py是启动训练的脚本，其中有如下参数需要修改：

- train_file代表训练数据的路径
- predict_file代表测试数据的路径
- output_dir是训练过程中模型、预测结果保存的文件夹
- proportion是训练数据的比例，如果是0.1，代表仅用原始数据的10%进行训练，如果是1，那么代表利用全量的训练数据
- checkpoint是预学习阶段训练保存的模型
- bart_model_path是下载的BART模型路径，即bart-base-chinese

**训练日志就是当前文件夹下的log.txt**

## 评估模型
```bash
python evaluate.py
```
evaluate.py是评估脚本

- test_file是将要评估的数据
- kg_path是知识图谱的路径
- kg_type是指当前是哪一个知识图谱。[kgclue,nlpcc]
- checkpoint是模型保存的检查点

**评估日志就是当前文件夹下的evaluate-log.txt**
