#!/usr/bin/env python3
"""
Python script converted from /project/6098391/wuroderi/Linear_Reasoning_Features/reasoning_representation/Figures_Interp_Reason&Memory.ipynb
"""

# Cell 1

import torch
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import json
import torch
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import LogisticRegression

# Set random seed
torch.manual_seed(8888)

save_path = '/mnt/workspace/workgroup/yhhong/reasoning_representations_outputs'


model_hs_cache_dict = {}

model_dict = {
    'Meta-Llama-3-8B': {'full_name': 'LLaMA3-8B-base', 'layer': 21},
    'gemma-2-9b': {'full_name': 'Gemma2-9B-base', 'layer': 16},
    'Mistral-7B-v0.3': {'full_name': 'Mistral-7B-v0.3-base', 'layer': 13},
    'OLMo-2-1124-7B': {'full_name': 'OLMo2-7B-base', 'layer': 17},
}

for model_name in model_dict.keys():
    
    loaded_dict = torch.load(os.path.join(save_path, f'{model_name}-base_hs_cache_no_cot_all.pt'))
    # print(loaded_dict)
    model_hs_cache_dict[model_name] = loaded_dict

    


# Cell 2

model_hs_cache_dict['Mistral-7B-v0.3'].keys()


# Cell 3

with open('/mnt/workspace/Interp_Reasoning/dataset/mmlu-pro-600samples.json', 'r', encoding='utf-8') as f:
    sampled_data = json.load(f)

reason_indices = [ix for ix, sample in enumerate(sampled_data) if sample['memory_reason_score'] > 0.5]
memory_indices = [ix for ix, sample in enumerate(sampled_data) if sample['memory_reason_score'] <= 0.5]

other_running_set_name_list = ['popqa', 'gsm_symbolic'] #['gsm8k', 'mgsm', 'popqa', 'ceval_liberal']


# Cell 4


# Create 1x4 figure layout, increase overall figure width
fig, axes = plt.subplots(1, 4, figsize=(24, 5))  # Increased figsize width

colors_shapes_dict = {
    'mmlu-pro_reason': ['purple','x'], ##2921A2 ##1C1771 #000080
    'mmlu-pro_memory': ['#FE4867', '*'], ##F85070 #FE4867 #B0573C
    'ceval_liberal': ['#FFCA9C', '*', 'C-Eval-H'],
    'gsm8k': ['#B3B3D9', 'x', 'GSM8K'],
    'mgsm': ['#A4E5EC', 'x', 'MGSM'],
    'mbpp': ['blue', 'x', 'MBPP'],
    'popqa': ['orange', '*', 'PopQA'],
    'gsm_symbolic': ['#000080', 'x', 'GSM-Symbolic'],
}

model_dict = {
    'Meta-Llama-3-8B': {'full_name': 'LLaMA3-8B-base', 'layer': 21},
    'gemma-2-9b': {'full_name': 'Gemma2-9B-base', 'layer': 16},
    'Mistral-7B-v0.3': {'full_name': 'Mistral-7B-v0.3-base', 'layer': 13},
    'OLMo-2-1124-7B': {'full_name': 'OLMo2-7B-base', 'layer': 17},
}

# Process each model's data and plot
for idx, model_name in enumerate(model_dict.keys()):
    
    hs_cache_no_cot_other_all = model_hs_cache_dict[model_name]
    
    mmlu_pro_hs = hs_cache_no_cot_other_all['mmlu-pro']
    
    mmlu_pro_hs_layer = mmlu_pro_hs[model_dict[model_name]['layer']]  
    mmlu_pro_hs_layer_flattened = mmlu_pro_hs_layer.squeeze(1)  

    pca_no_cot = PCA(n_components=2)
    hs_no_cot_pca = pca_no_cot.fit_transform(mmlu_pro_hs_layer_flattened.cpu().numpy()) # Each model layer has its own space
    
    
    # Plot on corresponding subplot
    ax = axes[idx]

    ax.scatter(hs_no_cot_pca[reason_indices, 0], hs_no_cot_pca[reason_indices, 1], color=colors_shapes_dict['mmlu-pro_reason'][0], marker=colors_shapes_dict['mmlu-pro_reason'][1], label='MMLU-Pro-R (Reasoning) HS', alpha=0.6)
    ax.scatter(hs_no_cot_pca[memory_indices, 0], hs_no_cot_pca[memory_indices, 1], color=colors_shapes_dict['mmlu-pro_memory'][0], marker=colors_shapes_dict['mmlu-pro_memory'][1], label='MMLU-Pro-M (Memory) HS', alpha=0.6)

    reason_part = hs_no_cot_pca[reason_indices]
    memory_part = hs_no_cot_pca[memory_indices]
    
    print('len(reason_part): ',len(reason_part))
    print('len(memory_part): ',len(memory_part))
    
    for ix, name in enumerate(other_running_set_name_list):
        
        print(hs_cache_no_cot_other_all.keys())
        other_hs_no_cot = hs_cache_no_cot_other_all[name][model_dict[model_name]['layer']]
        
        num_samples = 400
        random_indices = torch.randint(0, other_hs_no_cot.shape[0], (num_samples,))
        other_hs_no_cot = other_hs_no_cot[random_indices]


        print('len(other_hs_no_cot): ',len(other_hs_no_cot))
        other_hs_no_cot_pca = pca_no_cot.transform(other_hs_no_cot.cpu().numpy())
        ax.scatter(other_hs_no_cot_pca[:, 0], other_hs_no_cot_pca[:, 1], color=colors_shapes_dict[name][0], marker=colors_shapes_dict[name][1], label=f'{colors_shapes_dict[name][2]} HS', alpha=0.6)
        
        if name in ['gsm8k', 'mgsm']:
            reason_part = np.vstack((reason_part, other_hs_no_cot_pca))
        else:
            memory_part = np.vstack((memory_part, other_hs_no_cot_pca))
     
    
    
    # logic regression
    # Merge data and create labels
    X = np.vstack((reason_part, memory_part))
    y = np.array([0]*len(reason_part) + [1]*len(memory_part))
    # Train logistic regression model
    lr = LogisticRegression()
    lr.fit(X, y)
    # Get decision boundary parameters
    coef = lr.coef_[0]
    intercept = lr.intercept_
    # Generate x range for boundary line
    x_min, x_max = ax.get_xlim()
    x_values = np.linspace(x_min, x_max, 100)
    y_values = (-(intercept + coef[0] * x_values)) / coef[1]

    # Plot black dotted decision boundary
    ax.plot(x_values, y_values, 
     linestyle=':', 
     color='gray', 
     linewidth=3,  # Recommended to use values between 1.5-2.5
     zorder=3)

    memory_mean = np.mean(memory_part, axis=0)
    reason_mean = np.mean(reason_part, axis=0)

    # Draw arrow (parameters can be adjusted as needed)
    ax.annotate("", 
         xy=reason_mean,  
         xytext=memory_mean,  
         arrowprops=dict(arrowstyle="->",
                         # color="gray",
                         color = "#0432FF",
                         lw=3.5,          # Bold line width (original 1.5)
                         linestyle="-",
                         alpha=0.8,
                         mutation_scale=20,  # Increase arrow head size
                         shrinkA=30,       # Start point shrinkage
                         shrinkB=30),      # End point shrinkage
         )

    # Add labels and title
    ax.set_title(f"{model_dict[model_name]['full_name']}")
    ax.legend(fontsize=8)
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(ax.get_ylim())
    

# Adjust spacing between subplots
plt.subplots_adjust(wspace=0.12)  # Increased horizontal spacing between subplots

# Save as PDF

plt.savefig('4model_memory_reason_pca_comparison.pdf', format='pdf', bbox_inches='tight', dpi=300)

# Display chart
plt.show()


# Markdown Cell 5
# # PCA for Coding

# Cell 6


# Create 1x4 figure layout, increase overall figure width
fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # Increased figsize width

colors_shapes_dict = {
    'mmlu-pro_reason': ['purple','x'], ##2921A2 ##1C1771 #000080
    'mmlu-pro_memory': ['#FE4867', '*'], ##F85070 #FE4867 #B0573C
    'ceval_liberal': ['#FFCA9C', '*', 'C-Eval-H'],
    'gsm8k': ['#B3B3D9', 'x', 'GSM8K'],
    'mgsm': ['#A4E5EC', 'x', 'MGSM'],
    'mbpp': ['#00E1FA', '+', 'MBPP'],
    'human_eval': ['#ADFF49', '+', 'HumanEval'], #BCBD26 E4E5A8 ADFF49 4CFFAA
    'popqa': ['orange', '*', 'PopQA']
}

model_dict = {
    'Meta-Llama-3-8B': {'full_name': 'LLaMA3-8B-base', 'layer': 21},
    # 'gemma-2-9b': {'full_name': 'Gemma2-9B-base', 'layer': 16},
    'Mistral-7B-v0.3': {'full_name': 'Mistral-7B-v0.3-base', 'layer': 13}, #25
    # 'olmo2-7b': {'full_name': 'OLMo2-7B-base', 'layer': 17},
}

other_running_set_name_list = ['mbpp', 'human_eval']

# Process each model's data and plot
for idx, model_name in enumerate(model_dict.keys()):
    
    
    hs_cache_no_cot_other_all = model_hs_cache_dict[model_name]
    
    mmlu_pro_hs = hs_cache_no_cot_other_all['mmlu-pro']
    
    mmlu_pro_hs_layer = mmlu_pro_hs[model_dict[model_name]['layer']]  
    mmlu_pro_hs_layer_flattened = mmlu_pro_hs_layer.squeeze(1)  

    pca_no_cot = PCA(n_components=2)
    hs_no_cot_pca = pca_no_cot.fit_transform(mmlu_pro_hs_layer_flattened.cpu().numpy()) # Each model layer has its own space
    
    
    # Plot on corresponding subplot
    ax = axes[idx]

    ax.scatter(hs_no_cot_pca[reason_indices, 0], hs_no_cot_pca[reason_indices, 1], color=colors_shapes_dict['mmlu-pro_reason'][0], marker=colors_shapes_dict['mmlu-pro_reason'][1], label='MMLU-Pro-R (Reasoning) HS', alpha=0.6)
    ax.scatter(hs_no_cot_pca[memory_indices, 0], hs_no_cot_pca[memory_indices, 1], color=colors_shapes_dict['mmlu-pro_memory'][0], marker=colors_shapes_dict['mmlu-pro_memory'][1], label='MMLU-Pro-M (Memory) HS', alpha=0.6)

    reason_part = hs_no_cot_pca[reason_indices]
    memory_part = hs_no_cot_pca[memory_indices]
    
    print('len(reason_part): ',len(reason_part))
    print('len(memory_part): ',len(memory_part))
    
    for ix, name in enumerate(other_running_set_name_list):

        other_hs_no_cot = hs_cache_no_cot_other_all[name][model_dict[model_name]['layer']]
        
        num_samples = 400
        random_indices = torch.randint(0, other_hs_no_cot.shape[0], (num_samples,))
        other_hs_no_cot = other_hs_no_cot[random_indices]


        print('len(other_hs_no_cot): ',len(other_hs_no_cot))
        other_hs_no_cot_pca = pca_no_cot.transform(other_hs_no_cot.cpu().numpy())
        ax.scatter(other_hs_no_cot_pca[:, 0], other_hs_no_cot_pca[:, 1], color=colors_shapes_dict[name][0], marker=colors_shapes_dict[name][1], label=f'{colors_shapes_dict[name][2]} HS', alpha=0.6)
        
        # if name in ['gsm8k', 'mgsm']:
        #     reason_part = np.vstack((reason_part, other_hs_no_cot_pca))
        # else:
        #     memory_part = np.vstack((memory_part, other_hs_no_cot_pca))
             
    
    
    # logic regression
    # Merge data and create labels
    X = np.vstack((reason_part, memory_part))
    y = np.array([0]*len(reason_part) + [1]*len(memory_part))
    # Train logistic regression model
    lr = LogisticRegression()
    lr.fit(X, y)
    # Get decision boundary parameters
    coef = lr.coef_[0]
    intercept = lr.intercept_
    # Generate x range for boundary line
    x_min, x_max = ax.get_xlim()
    x_values = np.linspace(x_min, x_max, 100)
    y_values = (-(intercept + coef[0] * x_values)) / coef[1]

    # Plot black dotted decision boundary
    ax.plot(x_values, y_values, 
     linestyle=':', 
     color='gray', 
     linewidth=3,  # Recommended to use values between 1.5-2.5
     zorder=3)

    memory_mean = np.mean(memory_part, axis=0)
    reason_mean = np.mean(reason_part, axis=0)

    # Draw arrow (parameters can be adjusted as needed)
    ax.annotate("", 
         xy=reason_mean,  
         xytext=memory_mean,  
         arrowprops=dict(arrowstyle="->",
                         # color="gray",
                         color = "#0432FF",
                         lw=3.5,          # Bold line width (original 1.5)
                         linestyle="-",
                         alpha=0.8,
                         mutation_scale=20,  # Increase arrow head size
                         shrinkA=30,       # Start point shrinkage
                         shrinkB=30),      # End point shrinkage
         )

    # Add labels and title
    ax.set_title(f"{model_dict[model_name]['full_name']}")
    ax.legend(fontsize=8)
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(ax.get_ylim())
    

# Adjust spacing between subplots
plt.subplots_adjust(wspace=0.12)  # Increased horizontal spacing between subplots

# Save as PDF

plt.savefig('2model_memory_reason_pca_for_coding.pdf', format='pdf', bbox_inches='tight', dpi=300)

# 显示图表
plt.show()


# Cell 7

# 创建2x2的图表布局
fig, axes = plt.subplots(2, 2, figsize=(12, 10))  # 2行2列的布局

colors_shapes_dict = {
    'mmlu-pro_reason': ['purple', 'x'],
    'mmlu-pro_memory': ['#FE4867', '*'],
    'ceval_liberal': ['#FFCA9C', '*', 'C-Eval-H'],
    'gsm8k': ['#B3B3D9', 'x', 'GSM8K'],
    'mgsm': ['#A4E5EC', 'x', 'MGSM'],
    'mbpp': ['#00E1FA', '+', 'MBPP'],
    'human_eval': ['#ADFF49', '+', 'HumanEval'],
    'popqa': ['orange', '*', 'PopQA']
}

model_dict = {
    'Meta-Llama-3-8B': {'full_name': 'LLaMA3-8B-base', 'layer': 21},
    'Mistral-7B-v0.3': {'full_name': 'Mistral-7B-v0.3-base', 'layer': 13},
    'gemma-2-9b': {'full_name': 'Gemma2-9B-base', 'layer': 16},
    'olmo2-7b': {'full_name': 'OLMo2-7B-base', 'layer': 17},
}

other_running_set_name_list = ['mbpp', 'human_eval']

# 处理每个模型的数据并绘图
for idx, model_name in enumerate(model_dict.keys()):
    hs_cache_no_cot_other_all = model_hs_cache_dict[model_name]
    mmlu_pro_hs = hs_cache_no_cot_other_all['mmlu-pro']
    
    mmlu_pro_hs_layer = mmlu_pro_hs[model_dict[model_name]['layer']]  
    mmlu_pro_hs_layer_flattened = mmlu_pro_hs_layer.squeeze(1)  

    pca_no_cot = PCA(n_components=2)
    hs_no_cot_pca = pca_no_cot.fit_transform(mmlu_pro_hs_layer_flattened.cpu().numpy()) 

    # 选择相应的子图位置
    row = idx // 2  # 计算行
    col = idx % 2   # 计算列
    ax = axes[row, col]

    ax.scatter(hs_no_cot_pca[reason_indices, 0], hs_no_cot_pca[reason_indices, 1], color=colors_shapes_dict['mmlu-pro_reason'][0], marker=colors_shapes_dict['mmlu-pro_reason'][1], label='MMLU-Pro-R (Reasoning) HS', alpha=0.6)
    ax.scatter(hs_no_cot_pca[memory_indices, 0], hs_no_cot_pca[memory_indices, 1], color=colors_shapes_dict['mmlu-pro_memory'][0], marker=colors_shapes_dict['mmlu-pro_memory'][1], label='MMLU-Pro-M (Memory) HS', alpha=0.6)

    reason_part = hs_no_cot_pca[reason_indices]
    memory_part = hs_no_cot_pca[memory_indices]
    
    for ix, name in enumerate(other_running_set_name_list):
        other_hs_no_cot = hs_cache_no_cot_other_all[name][model_dict[model_name]['layer']]
        num_samples = 400
        random_indices = torch.randint(0, other_hs_no_cot.shape[0], (num_samples,))
        other_hs_no_cot = other_hs_no_cot[random_indices]

        other_hs_no_cot_pca = pca_no_cot.transform(other_hs_no_cot.cpu().numpy())
        ax.scatter(other_hs_no_cot_pca[:, 0], other_hs_no_cot_pca[:, 1], color=colors_shapes_dict[name][0], marker=colors_shapes_dict[name][1], label=f'{colors_shapes_dict[name][2]} HS', alpha=0.6)
    
    # logic regression
    X = np.vstack((reason_part, memory_part))
    y = np.array([0]*len(reason_part) + [1]*len(memory_part))
    lr = LogisticRegression()
    lr.fit(X, y)
    coef = lr.coef_[0]
    intercept = lr.intercept_
    x_min, x_max = ax.get_xlim()
    x_values = np.linspace(x_min, x_max, 100)
    y_values = (-(intercept + coef[0] * x_values)) / coef[1]

    ax.plot(x_values, y_values, linestyle=':', color='gray', linewidth=3, zorder=3)

    memory_mean = np.mean(memory_part, axis=0)
    reason_mean = np.mean(reason_part, axis=0)
    
    ax.annotate("", 
         xy=reason_mean,  
         xytext=memory_mean,  
         arrowprops=dict(arrowstyle="->",
                         color = "#0432FF",
                         lw=3.5,
                         linestyle="-",
                         alpha=0.8,
                         mutation_scale=20,
                         shrinkA=30,
                         shrinkB=30))

    ax.set_title(f"{model_dict[model_name]['full_name']}")
    ax.legend(fontsize=8)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(ax.get_ylim())

# Adjust spacing between subplots
plt.subplots_adjust(wspace=0.12, hspace=0.12)  # Increase horizontal and vertical spacing between subplots

# Save as PDF
plt.savefig('2model_memory_reason_pca_for_coding_2x2.pdf', format='pdf', bbox_inches='tight', dpi=300)

# Display chart
plt.show()


# Markdown Cell 8
# # Consine with LiReFs

# Cell 9

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F


def find_direction(reason_set=None, memory_set=None, layer = None):
    
#     if layer is not None:
#         reason_set_layer = reason_set[layer].to(torch.float64)
#         memory_set_layer = memory_set[layer].to(torch.float64)
        
#         print('reason_set_layer.shape: ', reason_set_layer.shape)
        
#         mean_reason_set_layer = reason_set_layer.mean(dim=-1)
#         mean_memory_set_layer = memory_set_layer.mean(dim=-1)

#         mean_diff = mean_reason_set_layer - mean_memory_set_layer  

    if layer is None:
        reason_set_layer = reason_set.to(torch.float64)
        memory_set_layer = memory_set.to(torch.float64)

        #print('reason_set_layer.shape: ', reason_set_layer.shape)

        mean_reason_set_layer = reason_set_layer.mean(dim=0)
        mean_memory_set_layer = memory_set_layer.mean(dim=0)

        mean_diff = mean_reason_set_layer - mean_memory_set_layer  
    
    return mean_diff
    

# 设置随机种子以确保可重复性
np.random.seed(8888)




model_dict = {
    'Meta-Llama-3-8B': {'full_name': 'LLaMA3-8B', 'layer': 21, 'type': 'base'},
    'gemma-2-9b': {'full_name': 'Gemma2-9B', 'layer': 16, 'type': 'base'},
    'Mistral-7B-v0.3': {'full_name': 'Mistral-7B-v0.3', 'layer': 13, 'type': 'base'},
    'olmo2-7b': {'full_name': 'OLMo2-7B', 'layer': 17, 'type': 'base'},
    'Meta-Llama-3-8B-Instruct': {'full_name': 'LLaMA3-8B', 'layer': 21, 'type': 'instruct'},
    'gemma-2-9b-it': {'full_name': 'Gemma2-9B', 'layer': 16, 'type': 'instruct'},
    'Mistral-7B-Instruct-v0.3': {'full_name': 'Mistral-7B-v0.3', 'layer': 13, 'type': 'instruct'},
    'olmo-2-1124-7B-Instruct': {'full_name': 'OLMo2-7B', 'layer': 17, 'type': 'instruct'},
}

model_hs_cache_dict = {}
    
cosine_data = {
    'LLaMA3-8B': {'reason': [],
                  'memory': [],
                  'reason_instruct': [],
                  'memory_instruct': []},
    'Gemma2-9B': {'reason': [],
                  'memory': [],
                  'reason_instruct': [],
                  'memory_instruct': []},
    'Mistral-7B-v0.3': {'reason': [],
                  'memory': [],
                  'reason_instruct': [],
                  'memory_instruct': []},
    'OLMo2-7B': {'reason': [],
                  'memory': [],
                  'reason_instruct': [],
                  'memory_instruct': []},
}

for model_name in model_dict.keys():
    
    loaded_dict = torch.load(os.path.join(save_path, f'{model_name}-base_hs_cache_no_cot_all.pt'))
    # print(loaded_dict)
    model_hs_cache_dict[model_name] = loaded_dict
    
    for layer in range(len(loaded_dict['mmlu-pro'])):

        reason_set = torch.cat((loaded_dict['mmlu-pro'][layer][reason_indices], loaded_dict['gsm8k'][layer], loaded_dict['mgsm'][layer]), dim=0)
        memory_set = torch.cat((loaded_dict['mmlu-pro'][layer][memory_indices], loaded_dict['ceval_liberal'][layer]), dim=0)

        direction = find_direction(reason_set=reason_set, memory_set=memory_set)
  

        if model_dict[model_name]['type'] == 'base':

            cosine_data[model_dict[model_name]['full_name']]['reason'].append(F.cosine_similarity(reason_set, direction.unsqueeze(0), dim=-1))
            cosine_data[model_dict[model_name]['full_name']]['memory'].append(F.cosine_similarity(memory_set, direction.unsqueeze(0), dim=-1))

        else:

            cosine_data[model_dict[model_name]['full_name']]['reason_instruct'].append(F.cosine_similarity(reason_set, direction.unsqueeze(0), dim=-1))
            cosine_data[model_dict[model_name]['full_name']]['memory_instruct'].append(F.cosine_similarity(memory_set, direction.unsqueeze(0), dim=-1))


# Cell 10

# Create one row with four columns of subplots
fig, axes = plt.subplots(1, 4, figsize=(28, 6))
model_list = ['LLaMA3-8B', 'Gemma2-9B', 'Mistral-7B-v0.3', 'OLMo2-7B']
# Plot data for each subplot
for idx, ax in enumerate(axes):
    
    # Calculate mean and standard deviation and plot
    def plot_line_with_ci(layers, data, color, label):
        
        data_tensor = torch.stack(data)
        mean = torch.mean(data_tensor, dim=-1)
        std = torch.std(data_tensor, dim=-1)
        
        ax.plot(layers, mean, color=color, label=label, linewidth=2)
        ax.fill_between(layers, mean-std, mean+std, color=color, alpha=0.2)
        
     
    model_name = model_list[idx]
    layers = np.arange(0, len(cosine_data[model_name]['reason']), 1)
    
    # Plot all lines
    plot_line_with_ci(layers, cosine_data[model_name]['reason'], 'blue', 'Reasoning, Base')
    plot_line_with_ci(layers, cosine_data[model_name]['reason_instruct'], 'purple', 'Reasoning, Instruct')
    plot_line_with_ci(layers, cosine_data[model_name]['memory'], 'red', 'Memory, Base')
    plot_line_with_ci(layers, cosine_data[model_name]['memory_instruct'], 'orange', 'Memory, Instruct')

    # 设置图表属性
    ax.set_xlabel('Layer', fontsize=16)
    ax.set_title(model_list[idx], fontsize=18, pad=15)  # 添加标题
    if ax == axes[0]:  # 只在第一个子图显示y轴标签
        ax.set_ylabel('Cosine similarity with\nLinear Reasoning Features', fontsize=17)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.tick_params(axis='both', labelsize=15)
    
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.8, label='Boundary' if idx == 0 else '')
    # ax.set_ylim(0, 0.55)

# 添加共享图例
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center',
           bbox_to_anchor=(0.5, 1.05), ncol=5, fontsize=16)

# 调整布局
plt.subplots_adjust(wspace=0.15, bottom=0.25)  # 为图例留出空间

# 保存图表
plt.savefig('model_memory_reason_cosine.pdf', format='pdf', bbox_inches='tight', dpi=300)

# 显示图表
plt.show()


# Markdown Cell 11
# # Cosine between LiReFs and Direction between PopQA and GSM-symbolic

# Cell 12

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F


def find_direction(reason_set=None, memory_set=None, layer = None):
    
#     if layer is not None:
#         reason_set_layer = reason_set[layer].to(torch.float64)
#         memory_set_layer = memory_set[layer].to(torch.float64)
        
#         print('reason_set_layer.shape: ', reason_set_layer.shape)
        
#         mean_reason_set_layer = reason_set_layer.mean(dim=-1)
#         mean_memory_set_layer = memory_set_layer.mean(dim=-1)

#         mean_diff = mean_reason_set_layer - mean_memory_set_layer  

    if layer is None:
        reason_set_layer = reason_set.to(torch.float64)
        memory_set_layer = memory_set.to(torch.float64)

        #print('reason_set_layer.shape: ', reason_set_layer.shape)

        mean_reason_set_layer = reason_set_layer.mean(dim=0)
        mean_memory_set_layer = memory_set_layer.mean(dim=0)

        mean_diff = mean_reason_set_layer - mean_memory_set_layer  
    
    return mean_diff
    

# 设置随机种子以确保可重复性
np.random.seed(8888)




model_dict = {
    'Meta-Llama-3-8B': {'full_name': 'LLaMA3-8B', 'layer': 21, 'type': 'base'},
    'gemma-2-9b': {'full_name': 'Gemma2-9B', 'layer': 16, 'type': 'base'},
    'Mistral-7B-v0.3': {'full_name': 'Mistral-7B-v0.3', 'layer': 13, 'type': 'base'},
    'OLMo-2-1124-7B': {'full_name': 'OLMo2-7B', 'layer': 17, 'type': 'base'},
    # 'Meta-Llama-3-8B-Instruct': {'full_name': 'LLaMA3-8B', 'layer': 21, 'type': 'instruct'},
    # 'gemma-2-9b-it': {'full_name': 'Gemma2-9B', 'layer': 16, 'type': 'instruct'},
    # 'Mistral-7B-Instruct-v0.3': {'full_name': 'Mistral-7B-v0.3', 'layer': 13, 'type': 'instruct'},
    # 'olmo-2-1124-7B-Instruct': {'full_name': 'OLMo2-7B', 'layer': 17, 'type': 'instruct'},
}


model_hs_cache_dict = {}
    
cosine_data = {
    'LLaMA3-8B': {'reason': [],
                  'memory': [],
                  'reason_instruct': [],
                  'memory_instruct': []},
    'Gemma2-9B': {'reason': [],
                  'memory': [],
                  'reason_instruct': [],
                  'memory_instruct': []},
    'Mistral-7B-v0.3': {'reason': [],
                  'memory': [],
                  'reason_instruct': [],
                  'memory_instruct': []},
    'OLMo2-7B': {'reason': [],
                  'memory': [],
                  'reason_instruct': [],
                  'memory_instruct': []},
}

for model_name in model_dict.keys():
    
    loaded_dict = torch.load(os.path.join(save_path, f'{model_name}-base_hs_cache_no_cot_all.pt'))
    # print(loaded_dict)
    model_hs_cache_dict[model_name] = loaded_dict
    
    for layer in range(len(loaded_dict['mmlu-pro'])):

        reason_set = loaded_dict['mmlu-pro'][layer][reason_indices]
        memory_set = loaded_dict['mmlu-pro'][layer][memory_indices]

        mmlu_pro_direction = find_direction(reason_set=reason_set, memory_set=memory_set)
        
        reason_set = loaded_dict['gsm_symbolic'][layer]
        memory_set = loaded_dict['popqa'][layer]

        popqa_gsmsymbolic_direction = find_direction(reason_set=reason_set, memory_set=memory_set)
  

        if model_dict[model_name]['type'] == 'base':

            cosine_data[model_dict[model_name]['full_name']]['reason'].append(F.cosine_similarity(popqa_gsmsymbolic_direction.unsqueeze(0), mmlu_pro_direction.unsqueeze(0), dim=-1))
            # cosine_data[model_dict[model_name]['full_name']]['memory'].append(F.cosine_similarity(memory_set, direction.unsqueeze(0), dim=-1))

#         else:

#             cosine_data[model_dict[model_name]['full_name']]['reason_instruct'].append(F.cosine_similarity(reason_set, direction.unsqueeze(0), dim=-1))
#             cosine_data[model_dict[model_name]['full_name']]['memory_instruct'].append(F.cosine_similarity(memory_set, direction.unsqueeze(0), dim=-1))


# Cell 13

# cosine_data

for model_name in model_dict.keys():
    
    print(model_name)
    
    for layer in range(len(loaded_dict['mmlu-pro'])):
        
        if layer % 4 == 0:
            print(cosine_data[model_dict[model_name]['full_name']]['reason'][layer])
         
    list_of_tensors = cosine_data[model_dict[model_name]['full_name']]['reason']
    stacked_tensor = torch.stack(list_of_tensors)
    average_tensor = torch.mean(stacked_tensor, dim=0)
    print('avg: ', average_tensor)


# Markdown Cell 14
# # Cosine between PCA Main Components and Mean Diff

# Cell 15

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

model_dict = {
    'Meta-Llama-3-8B': {},
    'Mistral-7B-v0.3': {}
}

for ix, model_name in enumerate(model_dict.keys()):
    
    loaded_dict = torch.load(os.path.join(save_path, f'{model_name}-base_hs_cache_no_cot_all.pt'))

    
    for layer in range(len(loaded_dict['mmlu-pro'])):
        model_dict[model_name][layer] = []

        # reason_set = torch.cat((loaded_dict['mmlu-pro'][layer][reason_indices], loaded_dict['gsm8k'][layer], loaded_dict['mgsm'][layer]), dim=0)
        # memory_set = torch.cat((loaded_dict['mmlu-pro'][layer][memory_indices], loaded_dict['ceval_liberal'][layer]), dim=0)
        reason_set = loaded_dict['mmlu-pro'][layer][reason_indices]
        memory_set = loaded_dict['mmlu-pro'][layer][memory_indices]

        direction = find_direction(reason_set=reason_set, memory_set=memory_set) # mean diff
        
        #pca
        pca_no_cot = PCA(n_components=10)
        hs_pca = pca_no_cot.fit_transform(torch.cat((reason_set, memory_set), dim=0).cpu().numpy()) # Each model layer has its own space
        
        
        principal_components = pca_no_cot.components_
        for component in principal_components:
            # print('component: ',component)
            model_dict[model_name][layer].append(F.cosine_similarity(torch.tensor(component), direction, dim=-1))


# Cell 16

# Create a figure containing two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 6))


data1 = np.array([[tensor.item() for tensor in model_dict['Meta-Llama-3-8B'][i]] for i in range(len(model_dict['Meta-Llama-3-8B']))]).T  
data2 = np.array([[tensor.item() for tensor in model_dict['Mistral-7B-v0.3'][i]] for i in range(len(model_dict['Mistral-7B-v0.3']))]).T
 
# First heatmap
sns.heatmap(data1,
            cmap='RdBu_r',
            center=0,
            vmin=-1,
            vmax=1,
            cbar_kws={'label': 'Cosine Similarity'},
            xticklabels=True,
            yticklabels=True,
            ax=ax1)

cbar1 = ax1.collections[0].colorbar
cbar1.set_label('Cosine Similarity', fontsize=15)  # Increase colorbar label font size
cbar1.ax.tick_params(labelsize=14)  # Set colorbar tick font size

# Set title and labels for first plot
ax1.set_title('Cosine Similarity between Mean Difference and Top Principle Component (LLaMA3-8B-base)', fontsize=16, pad=20)
ax1.set_xlabel('Layer')
ax1.set_ylabel('Component Index')

ax1.set_xlabel('Layer', fontsize=14)
ax1.set_ylabel('Component Index', fontsize=14)
ax1.tick_params(axis='both', labelsize=13)

# Second heatmap
sns.heatmap(data2,
            cmap='RdBu_r',
            center=0,
            vmin=-1,
            vmax=1,
            cbar_kws={'label': 'Cosine Similarity'},
            xticklabels=True,
            yticklabels=True,
            ax=ax2)

cbar2 = ax2.collections[0].colorbar
cbar2.set_label('Cosine Similarity', fontsize=15)  # Increase colorbar label font size
cbar2.ax.tick_params(labelsize=14)  # Set colorbar tick font size

# Set title and labels for second plot
ax2.set_title('Cosine Similarity between Mean Difference and Top Principle Component (Mistral-7B-v0.3-base)', fontsize=16, pad=20)

ax2.set_xlabel('Layer', fontsize=16)
ax2.set_ylabel('Component Index', fontsize=16)
ax2.tick_params(axis='both', labelsize=14)

# Adjust layout
plt.tight_layout()


plt.savefig('cosine_pca_component_mean_diff.pdf', format='pdf', bbox_inches='tight', dpi=300)

# Display figure
plt.show()


# Markdown Cell 17
# # Similarity between Reason Score and Projection

# Cell 18

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

from scipy.stats import spearmanr


def get_projection(direction, dataset_hs):
    
    # print('dataset_hs.shape: ',dataset_hs.shape)
    dataset_hs = dataset_hs.to(torch.float64)
    
    direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8) # direction needs to be a unit vector
    projection_value = dataset_hs @ direction
    
#     print('projection_value.shape: ',projection_value.shape)
    
    
    return projection_value
    

model_dict = {
    'Meta-Llama-3-8B': {},
    'Mistral-7B-v0.3': {}
}

with open('/mnt/workspace/Interp_Reasoning/dataset/mmlu-pro-600samples.json', 'r', encoding='utf-8') as f:
    sampled_data = json.load(f)

reason_indices = [ix for ix, sample in enumerate(sampled_data) if sample['memory_reason_score'] > 0.5]
memory_indices = [ix for ix, sample in enumerate(sampled_data) if sample['memory_reason_score'] <= 0.5]



for ix, model_name in enumerate(model_dict.keys()):
    
    loaded_dict = torch.load(os.path.join(save_path, f'{model_name}-base_hs_cache_no_cot_all.pt'))

    
    for layer in range(len(loaded_dict['mmlu-pro'])):

        # reason_set = torch.cat((loaded_dict['mmlu-pro'][layer][reason_indices], loaded_dict['gsm8k'][layer], loaded_dict['mgsm'][layer]), dim=0)
        # memory_set = torch.cat((loaded_dict['mmlu-pro'][layer][memory_indices], loaded_dict['ceval_liberal'][layer]), dim=0)
        reason_set = loaded_dict['mmlu-pro'][layer][reason_indices]
        memory_set = loaded_dict['mmlu-pro'][layer][memory_indices]

        direction = find_direction(reason_set=reason_set, memory_set=memory_set) # mean diff
        
        projection = get_projection(direction=direction, dataset_hs=loaded_dict['mmlu-pro'][layer])
        
        model_dict[model_name][layer] = [{"score": s['memory_reason_score'], "projection": p} for s, p in zip(sampled_data, projection)]


# Cell 19

from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


model_list = ['LLaMA3-8B-base', 'Mistral-7B-v0.3-base']

for layer in range(len(loaded_dict['mmlu-pro'])): 
    
    if layer <= 2:
        continue
        
    # Plot two subplots
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))  # 1 row, 2 columns of subplots
    
    # # Add overall title
    # fig.suptitle(f'Layer {layer} Analysis', fontsize=16, y=1.05)
    
    for ix, model_name in enumerate(model_dict.keys()):
        
        data = model_dict[model_name][layer]
        # Extract data
        scores = np.array([d['score'] for d in data])
        projections = np.array([d['projection'] for d in data])
        
        # Calculate Pearson correlation coefficient and p-value
        pearson_corr, pearson_p = pearsonr(scores, projections)
        spearman_corr, spearman_p = spearmanr(projections, scores)
        
        print(f"Pearson Correlation on {model_name} in Layer {layer}: {pearson_corr:.3f}, p-value: {pearson_p:.3e}")
        print(f"Spearman Correlation: {spearman_corr:.3f}, p-value: {spearman_p:.3e}")
        
        unique_scores = np.unique(scores)
        n_boxes = len(unique_scores)
        cmap = plt.get_cmap('coolwarm_r')
        colors = [cmap(i/(n_boxes-1)) for i in range(n_boxes)]
        
        # Create separate box plots for each score interval
        positions = np.arange(n_boxes)
        for pos, color, score in zip(positions, colors, unique_scores):
            mask = scores == score
            sns.boxplot(x=[pos] * sum(mask), y=projections[mask], 
                       color=color, width=0.3, ax=axs[ix])
        
        # Set title and labels
        axs[ix].set_title(f'{model_list[ix]}\nSpearman ρ = {spearman_corr:.3f} (p = {spearman_p:.2e})',
                         fontsize=16, pad=10)
        axs[ix].set_xlabel('Reasoning Score Categories provided by GPT-4o', fontsize=14)
                           
        if ix == 0:
            axs[ix].set_ylabel('Projection Value\non Linear Reasoning Features (LiReFs)', fontsize=15)
        
        # Set x-axis tick labels
        axs[ix].set_xticks(positions)
        axs[ix].set_xticklabels([f'{score:.2f}' for score in unique_scores], rotation=45)
        
        axs[ix].tick_params(axis='both', labelsize=11)
        
        # Add grid lines
        axs[ix].grid(True, alpha=0.3)
        
        
    # Adjust spacing between subplots
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.11)
    
    if layer == 9:
        plt.savefig('spearman_boxplot_on_score_and_projection.pdf', format='pdf', bbox_inches='tight', dpi=300)
        
    plt.show()


# Cell 20

import seaborn as sns
sns.jointplot(x=projections, y=scores, kind='hex', cmap='Blues')


# Cell 21



# Cell 22

# !pip install datasets
# from datasets import load_dataset

# ds = load_dataset("akariasai/PopQA")


# # Cell 23

# ds.save_to_disk("PopQA")


# # Cell 24

# !zip -r PopQA.zip PopQA

