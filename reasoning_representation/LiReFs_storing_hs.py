#!/usr/bin/env python3
"""
Python script converted from LiReFs_storing_hs.ipynb
"""

# Cell 1

import torch

# Check if CUDA is available
if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    print(f"Find {gpu_count} GPU can be used.")

    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        print(f"GPU {i + 1}: {gpu_name}")
else:
    print("No GPU can be used.")


# Cell 2

import copy
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
# from bert_score import score
import statistics
from ast import literal_eval
import functools
import json
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

random.seed(8888)
torch.manual_seed(8888)
random.seed(8888)
np.random.seed(8888)

if torch.cuda.is_available():
    torch.cuda.manual_seed(8888)
    torch.cuda.manual_seed_all(8888)


from tqdm import tqdm

torch.set_grad_enabled(False)
tqdm.pandas()


# Cell 3

torch.cuda.set_device(0)

from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from os.path import join

model_dir = "/home/wuroderi/projects/def-zhijing/wuroderi/models"
model_name = 'Llama-3.1-8B'  #gemma-2-9b-it  gemma-2-9b
output_dir = '/home/wuroderi/projects/def-zhijing/wuroderi/Linear_Reasoning_Features/outputs'
dataset_dir = '/home/wuroderi/projects/def-zhijing/wuroderi/Linear_Reasoning_Features/dataset'

#'Meta-Llama-3-8B-Instruct' #'Llama-2-7b-chat-hf' #'llama-7b' 
# "OLMo-7B-Instruct" olmo-2-1124-7B-Instruct 'OLMo-2-1124-7B'  "OLMo-7B-0724-Instruct-hf" The situation with OLMo1 is very abnormal, not sure why
# Qwen-7B, Qwen1.5-7B, Qwen-7B-Chat, Qwen1.5-7B-Chat, Qwen2-7B-Instruct, Qwen2.5-7B-Instruct, Qwen2.5-Coder-7B, Qwen2.5 has been optimized for Chinese, the situation is quite different
# Mistral-7B-Instruct-v0.1, Mistral-7B-Instruct-v0.2, Mistral-7B-Instruct-v0.3, Mistral-7B-v0.3
# Yi-1.5-6B-Chat, Yi-6B-Chat
# gemma-2-9b-it gemma-7b-it, gemma-2-9b
# gpt-j-6b, mpt-7b-chat, opt-6.7b, pythia-6.9b, zephyr-7b-beta, falcon-7b-instruct
# deepseek


model = AutoModelForCausalLM.from_pretrained(
    join(model_dir, model_name),
    dtype=torch.float32,
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(join(model_dir, model_name), trust_remote_code=True)

if 'llama' in model.config.model_type.lower() or 'mistral' in model.config.model_type.lower() or 'yi' in model.config.model_type.lower() or 'gptj' in model.config.model_type.lower():
    tokenizer.pad_token_id = tokenizer.eos_token_id
elif 'qwen' in model.config.model_type.lower():
    tokenizer.pad_token = '<|endoftext|>'
    # in gemma, pad_token_id = 0 is default
    # in olmo, pad_token_id = 1 is default
    
tokenizer.padding_side = "left"

model.to('cuda')


# Cell 4

print('model: ',model)
print('model.config: ',model.config)
print('model.config.model_type.lower(): ',model.config.model_type.lower())  # Often provides a string identifier
# print('model.config.num_hidden_layers: ',model.config.num_hidden_layers)


#import pdb; pdb.set_trace()

if 'gptj' in model.config.model_type.lower():
    model_layers_num = int(model.config.n_layer)  
    mlp_vector_num = 16384
    mlp_dim_num = int(model.config.n_embd)
    layer_name = 'transformer.h'
    mlp_name = 'mlp'
    mlp_last_layer_name = 'fc_out'

elif 'qwen' in model.config.model_type.lower() and 'qwen2' not in model.config.model_type.lower(): #qwen1, qwen2.5
    layer_name = 'transformer.h'
    mlp_name = 'mlp'
    mlp_last_layer_name = 'w2'
    mlp_dim_num = int(model.config.hidden_size)
    model_layers_num = int(model.config.num_hidden_layers)
    mlp_vector_num = int(model.config.intermediate_size / 2)
    
else:
    model_layers_num = int(model.config.num_hidden_layers)  # on olmo1, olmo2, qwen2, qwen2.5, llama, ...
    mlp_vector_num = int(model.config.intermediate_size)
    mlp_dim_num = int(model.config.hidden_size)
    layer_name = 'model.layers' 
    mlp_name = 'mlp'
    mlp_last_layer_name = 'down_proj'
    attn_name = 'self_attn'
    
    
    


# Cell 5

import datasets
import json
import re
import random
from datasets import load_from_disk
from tqdm import tqdm

n_new_tokens = 100
NUll_num = 0

def form_options(options: list):
    option_str = 'Options are:\n'
    opts = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    for opt, o in zip(options, opts):
        option_str += f'({o}): {opt}' + '\n'
    return option_str


def get_prediction(output):
    pattern = r"answer is \(?([ABCDEFGHIJ])\)?"
    match = re.search(pattern, output)
    if match:
        #print('prediction success: ',match.group(1))
        return match.group(1)
    else:
        #print("extraction failed, do a random guess")
        global NUll_num  
        NUll_num += 1
        return random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'])


def generate_outputs(questions):
    
    inputs = tokenizer(questions, return_tensors="pt", padding="longest", return_token_type_ids=False).to('cuda')
    input_length = inputs.input_ids.size(1)
    output = model(**inputs, output_hidden_states = True)
#     Question_input = [[{"role": "user", "content": prompt}] for prompt in questions]
#     texts = tokenizer.apply_chat_template(Question_input ,tokenize=False)

#     inputs = tokenizer(texts, padding="longest", return_tensors="pt")
#     inputs = {key: val.cuda() for key, val in inputs.items()}
#     output = model(**inputs, output_hidden_states = True)
    
    return output

def generate_questions(questions):
    
    inputs = tokenizer(questions, return_tensors="pt", padding="longest", return_token_type_ids=False).to('cuda')
    input_length = inputs.input_ids.size(1)
    gen_tokens = model.generate(**inputs, max_new_tokens=n_new_tokens, do_sample=False)

    gen_text = tokenizer.batch_decode(gen_tokens[:, input_length:], skip_special_tokens=True)
    
    return gen_text


dataset = load_from_disk('/home/wuroderi/projects/def-zhijing/wuroderi/Linear_Reasoning_Features/dataset/mmlu-pro')

categories = ['computer science', 'math', 'chemistry', 'engineering', 'law', 'biology',
              'health', 'physics', 'business', 'philosophy', 'economics', 'other',
              'psychology', 'history']

per_category_accuracy = {c: [0, 0] for c in categories}
success, fail = 0, 0
answers = []

print('----------------- Start Answering -------------------')
queries_batch = []  # Can test whether batch or single approach has higher accuracy and is more suitable # Found that they are basically the same, padding does not affect accuracy
entry_batch = []

batch_size = 4

random.seed(8888)

#import pdb; pdb.set_trace()

test_data = list(dataset['test'])

# example: running on 600 samples
sampled_data = random.sample(test_data, 600)

layers_to_cache = list(range(model_layers_num))
print('layers_to_cache: ',layers_to_cache)
#####hs_cache_cot = {}
#####hs_cache_no_cot = {}

#####print('----------------- Running no cot Inference -------------------')
#####for ix, entry in tqdm(enumerate(sampled_data)):
        
    #####query = 'Q: ' + entry['question'] + "\nA: "
    
    #####queries_batch.append(query)
    
    #####if len(queries_batch) == batch_size or ix == len(dataset['test']) - 1:
        #####output = generate_outputs(queries_batch)
        
        #####for layer in layers_to_cache:
            #####if layer not in hs_cache_no_cot:
                #####hs_cache_no_cot[layer] = output["hidden_states"][layer][: ,-1 , :].cpu() #bs * tok * dims
            #####else:
                #####hs_cache_no_cot[layer] = torch.cat((hs_cache_no_cot[layer], output["hidden_states"][layer][: ,-1 , :].cpu()), dim=0)

        
        #####queries_batch = []
    #####torch.cuda.empty_cache()

#torch.save(hs_cache_no_cot, os.path.join(output_dir, f'{model_name}-base_hs_cache_no_cot.pt'))
    


# Markdown Cell 6
# # **PCA**

# Cell 7

###with open('/home/wuroderi/projects/def-zhijing/wuroderi/Linear_Reasoning_Features/dataset/mmlu-pro-600samples.json', 'r', encoding='utf-8') as f:
    ###sampled_data = json.load(f)

###reason_indices = [ix for ix, sample in enumerate(sampled_data) if sample['memory_reason_score'] > 0.5]
###memory_indices = [ix for ix, sample in enumerate(sampled_data) if sample['memory_reason_score'] <= 0.5]

with open('/home/wuroderi/projects/def-zhijing/wuroderi/Linear_Reasoning_Features/dataset/mmlu-pro-3000samples.json', 'r', encoding='utf-8') as f:
    sampled_data = json.load(f)

reason_indices = [ix for ix, sample in enumerate(sampled_data) if sample['memory_reason_score'] > 0.5]
memory_indices = [ix for ix, sample in enumerate(sampled_data) if sample['memory_reason_score'] <= 0.5]

hs_cache_no_cot = {}

print("mmlu dataset...")
for ix, entry in tqdm(enumerate(sampled_data)):
        
    query = 'Q: ' + entry['question'] + "\nA: "
    
    queries_batch.append(query)
    
    if len(queries_batch) == batch_size or ix == len(sampled_data) - 1:
        output = generate_outputs(queries_batch)
        
        for layer in layers_to_cache:
            if layer not in hs_cache_no_cot:
                hs_cache_no_cot[layer] = output["hidden_states"][layer][: ,-1 , :].cpu() #bs * tok * dims
            else:
                hs_cache_no_cot[layer] = torch.cat((hs_cache_no_cot[layer], output["hidden_states"][layer][: ,-1 , :].cpu()), dim=0)

        
        queries_batch = []
    torch.cuda.empty_cache()

torch.save(hs_cache_no_cot, os.path.join(output_dir, f'{model_name}-base_hs_cache_no_cot.pt'))
print("saved")


# Cell 8

# loading and running gsm8k or other dataset


gsm8k_ds_main = load_from_disk('/home/wuroderi/projects/def-zhijing/wuroderi/Linear_Reasoning_Features/dataset/gsm8k/main') 
gsm8k_ds_main_test = list(gsm8k_ds_main['test'])

mbpp_ds_full = load_from_disk('/home/wuroderi/projects/def-zhijing/wuroderi/Linear_Reasoning_Features/dataset/mbpp/full')
mbpp_ds_full_val = list(mbpp_ds_full['validation'])
mbpp_ds_full = load_from_disk('/home/wuroderi/projects/def-zhijing/wuroderi/Linear_Reasoning_Features/dataset/mbpp/full')
mbpp_ds_full_test = list(mbpp_ds_full['test'])

# example on MGSM, feel free to add other categories in MGSM
mgsm_zh = pd.read_csv('/home/wuroderi/projects/def-zhijing/wuroderi/Linear_Reasoning_Features/dataset/mgsm/mgsm_zh.tsv', sep='\t')
mgsm_zh_test = mgsm_zh.values.tolist()
mgsm_de = pd.read_csv('/home/wuroderi/projects/def-zhijing/wuroderi/Linear_Reasoning_Features/dataset/mgsm/mgsm_de.tsv', sep='\t')
mgsm_de_test = mgsm_de.values.tolist()
mgsm_bn = pd.read_csv('/home/wuroderi/projects/def-zhijing/wuroderi/Linear_Reasoning_Features/dataset/mgsm/mgsm_bn.tsv', sep='\t')
mgsm_bn_test = mgsm_bn.values.tolist()
mgsm_ja = pd.read_csv('/home/wuroderi/projects/def-zhijing/wuroderi/Linear_Reasoning_Features/dataset/mgsm/mgsm_ja.tsv', sep='\t')
mgsm_ja_test = mgsm_ja.values.tolist()
mgsm_te = pd.read_csv('/home/wuroderi/projects/def-zhijing/wuroderi/Linear_Reasoning_Features/dataset/mgsm/mgsm_te.tsv', sep='\t')
mgsm_te_test = mgsm_te.values.tolist()


# example on C-Eval, feel free to add other categories in C-Eval
ceval_chi = pd.read_csv('/home/wuroderi/projects/def-zhijing/wuroderi/Linear_Reasoning_Features/dataset/ceval-exam/test/chinese_language_and_literature_test.csv')['question']
ceval_chi_test = ceval_chi.tolist()
ceval_his = pd.read_csv('/home/wuroderi/projects/def-zhijing/wuroderi/Linear_Reasoning_Features/dataset/ceval-exam/test/high_school_history_test.csv')['question']
ceval_his_test = ceval_his.tolist()
ceval_pol = pd.read_csv('/home/wuroderi/projects/def-zhijing/wuroderi/Linear_Reasoning_Features/dataset/ceval-exam/test/high_school_politics_test.csv')['question']
ceval_pol_test = ceval_pol.tolist()
ceval_mar = pd.read_csv('/home/wuroderi/projects/def-zhijing/wuroderi/Linear_Reasoning_Features/dataset/ceval-exam/test/marxism_test.csv')['question']
ceval_mar_test = ceval_mar.tolist()
ceval_bus = pd.read_csv('/home/wuroderi/projects/def-zhijing/wuroderi/Linear_Reasoning_Features/dataset/ceval-exam/test/business_administration_test.csv')['question']
ceval_bus_test = ceval_bus.tolist()


popqa_test = load_from_disk('/home/wuroderi/projects/def-zhijing/wuroderi/Linear_Reasoning_Features/dataset/PopQA/test') 
popqa_test = list(popqa_test)



other_running_set_name_list = ['ceval_liberal', 'gsm8k', 'mgsm', 'popqa'] # mbpp,, hoppingtoolate， , 'mbpp', 'popqa',
# other_running_set_name_list = ['mbpp']
other_dataset = None

hs_cache_no_cot_other_all = {}

for other_running_set_name in other_running_set_name_list:
    
    if other_running_set_name == 'mbpp':
        other_dataset = mbpp_ds_full_test
    elif other_running_set_name == 'gsm8k':
        other_dataset = gsm8k_ds_main_test
    elif other_running_set_name == 'mgsm': #multilingual gsm8k
        other_dataset = mgsm_zh_test + mgsm_de_test + mgsm_bn_test + mgsm_ja_test + mgsm_te_test
    elif other_running_set_name == 'ceval_liberal':
        other_dataset = ceval_chi_test + ceval_his_test + ceval_pol_test + ceval_mar_test # + ceval_bus_test
    elif other_running_set_name == 'popqa': #multilingual gsm8k
        other_dataset = popqa_test


    print(f'#####Running on {other_running_set_name} test set')
    print(f'The size is {len(other_dataset)}')

    layers_to_cache_other = list(range(model_layers_num))
    print('layers_to_cache_other: ',layers_to_cache_other)
    hs_cache_no_cot_other = {}
    queries_batch = []
    batch_size = 4

    for ix, entry in tqdm(enumerate(other_dataset)):

        if other_running_set_name == 'gsm8k':
            query = 'Q: ' + entry['question'] + "\nA: "
        elif other_running_set_name == 'mbpp':
            query = 'Q: ' + entry['text'] + "\nA: "
        elif other_running_set_name == 'mgsm':
            query = 'Q: ' + entry[0] + "\nA: "
        elif other_running_set_name == 'ceval_liberal':
            query = 'Q: ' + entry + "\nA: "
        elif other_running_set_name == 'popqa':
            query = 'Q: ' + entry['question'] + "\nA: "

        queries_batch.append(query)

        if len(queries_batch) == batch_size or ix == len(other_dataset) - 1:
            output = generate_outputs(queries_batch)

            for layer in layers_to_cache_other:
                if layer not in hs_cache_no_cot_other:
                    hs_cache_no_cot_other[layer] = output["hidden_states"][layer][: ,-1 , :].cpu() #bs * tok * dims
                else:
                    hs_cache_no_cot_other[layer] = torch.cat((hs_cache_no_cot_other[layer], output["hidden_states"][layer][: ,-1 , :].cpu()), dim=0)

            queries_batch = []
        torch.cuda.empty_cache()

    hs_cache_no_cot_other_all[other_running_set_name] = hs_cache_no_cot_other



hs_cache_no_cot_other_all['mmlu-pro-m'] = sampled_data[memory_indices]
hs_cache_no_cot_other_all['mmlu-pro-r'] = sampled_data[reason_indices]

# Cell 9

# os.makedirs(save_path, exist_ok=True)  
save_path = '/home/wuroderi/projects/def-zhijing/wuroderi/Linear_Reasoning_Features/outputs'

torch.save(hs_cache_no_cot_other_all, os.path.join(save_path, f'{model_name}-base_hs_cache_no_cot_all.pt'))

