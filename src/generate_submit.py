from utils import read_test_data, load_model, evaluate_ner, get_topic_entity, read_kg, get_inputs_for_step2, evaluate_qa, get_submissions
import json

write_ner_results_path='../outputs/ner_results.json'
write_qa_results_path='../outputs/qa_results.json'

test_examples=read_test_data()
model,tokenizer=load_model()
evaluate_ner(test_examples,model=model,tokenizer=tokenizer,write_results_path=write_ner_results_path)
ner_results=read_test_data(write_ner_results_path)

sub_map,alias_map,ent_to_relations=read_kg()

get_topic_entity(ner_results,sub_map,alias_map,write_ner_results_path)

inputs_for_step2=get_inputs_for_step2(ner_results,alias_map,ent_to_relations)

evaluate_qa(test_examples=inputs_for_step2,model=model,tokenizer=tokenizer,write_results_path=write_qa_results_path)

'''
每一个example的topic_entity并不是KG中的实体，而是可以在question找得到的string，需要alias_map才能链接到实体上
'''
qa_results=read_test_data(write_qa_results_path)

submissions=get_submissions(qa_results=qa_results,alias_map=alias_map,sub_map=sub_map)

with open('../outputs/submissions.json','w') as f:
    for example in submissions:
        f.write(json.dumps(example,ensure_ascii=False)+'\n')
