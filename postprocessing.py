import json
import numpy as np
import seaborn as sns 
import pandas as pd
import matplotlib.pyplot as plt

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)
sns.set_theme()
sns.set(font_scale=1.5) 

metric_list = [['IIF', 'IID', 'IIR'], ['IGF','IGD','IGR'],  ['GIF','GID','GIR'],  ['GGF','GGD','GGR'],  ['AIF','AID','AIR'],  ['AGF','AGD','AGR']]
column_names = ["II", "IG", "GI", "GG", "AI", "AG"]
x_values = ['8', '4', '2', '1', '1/2', '1/4', '1/8', 'ST']
group = 'Age'


def load_data(model, Experiment_nr, group, metric_list = metric_list, apply_min_max = False, max_ = 0, min_ = 0):
    
    df = pd.DataFrame(columns = column_names)
    df["II"] = [[], [], []]
    for metric in range(len(metric_list)):
        for component in range(len(metric_list[metric])):
            if group == 'Age':
                metrics = json.load(open('./save_exp/ml-1m/Experiment_' + str(Experiment_nr) + '_' + model + '/' + group +'/' + metric_list[metric][component]+ '_all_' + model + '_Y.json', 'r'))
                static  = json.load(open('./save_exp/ml-1m/Experiment_' + str(Experiment_nr) + '_' + model + '/' + group +'/' + metric_list[metric][component] +'_all_' + model + '_static_Y.json', 'r'))
            elif group == 'Gender':
                metrics = json.load(open('./save_exp/ml-1m/Experiment_' + str(Experiment_nr) + '_' + model + '/' + group +'/' + metric_list[metric][component]+ '_all_' + model + '.json', 'r'))
                static  = json.load(open('./save_exp/ml-1m/Experiment_' + str(Experiment_nr) + '_' + model + '/' + group +'/' + metric_list[metric][component] +'_all_' + model + '_static.json', 'r'))
            elif group == 'lt':
                metrics = json.load(open('./save_exp/lt/Experiment_' + str(Experiment_nr) + '_' + model + '/' + metric_list[metric][component]+ '_all_' + model + '.json', 'r'))
                static  = json.load(open('./save_exp/lt/Experiment_' + str(Experiment_nr) + '_' + model + '/' + metric_list[metric][component] +'_all_' + model + '_static.json', 'r'))
            else:
                print('Unknown group')
            metrics.extend(static)
            if apply_min_max == True:
                try:
                    metrics = min_max(np.array(metrics), max_ = max_[component][metric], min_ = min_[component][metric])
                except TypeError:
                    metrics = min_max(np.array(metrics), max_ = np.max(metrics), min_ = np.min(metrics))
            else:
                metrics = np.array(metrics)
                
            df[column_names[metric]][component] = metrics
    
    df.index = ['F', 'D', 'R']
    return df





def AUC_trap(x, y, minval):
    area = 0
    
    for i in range(len(x)-1):
        if x[i+1] <= minval:
            area += 0.5*(y[i+1]+ y[i])*(x[i+1] - x[i])
    return area 

def AUC(x, y, minval):
    area = 0
    for i in range(len(x)-1):
        if x[i+1] <= minval:
            area += (x[i+1]- x[i])*y[i+1] - 0.5*(x[i+1]- x[i])*(y[i+1]-y[i])
    return area 

def find_global_min_max(models):
    global_min = np.ones((3, 6))*100
    global_max = np.zeros((3, 6))
    for model in models:
        for i in range(len(model)):
            for j in range(len(model.iloc[i])):
                if global_min[i][j] > np.min(model.iloc[i][j]):
                    global_min[i][j] = np.min(model.iloc[i][j])
                if global_max[i][j] < np.max(model.iloc[i][j]):
                    global_max[i][j] = np.max(model.iloc[i][j])
    return global_min, global_max


def min_max(arr, max_ = 0, min_ = 0):
    if max_ == 0:
        max_ = np.max(arr)
    if min == 0:
        min_ = np.min(arr)
    min_max = (arr-min_)/(max_-min_)
    return min_max



def figure_2(model, column_names = column_names):
    """ Reproduce figure 2
    """
    fig, axs = plt.subplots(3, 6, constrained_layout=True, figsize = (20, 10))
    beta_values = ['8', '4', '2', '1', '1/2', '1/4', '1/8', 'ST']
    
    x_axis = np.arange(len(beta_values))

    for metric in range(len(metric_list)):
        for component in range(len(metric_list[metric])):
            metrics = model[column_names[metric]][component]
            sns.barplot(ax=axs[component, metric], x=x_axis - 0.2, y=metrics, palette='Blues_d')
            axs[component, metric].plot(x_axis - 0.2, metrics, '--.', color = 'r')
            axs[component, metric].set_xticks(x_axis, beta_values, rotation = 60)
            axs[component, metric].set_title(metric_list[metric][component])
    fig.supxlabel(r'$\beta$')

    plt.savefig('Figure_2.png', bbox_inches='tight')
    



def figure_3(models, model_name = ['BPRMF', 'LDA', 'PureSVD', 'SLIM', 'WRMF'], column_names = column_names):
    """ Reproduce Figure 3 - Disparity-relevance tradeoff curves
    """
    fig, axs = plt.subplots(1, 6, constrained_layout=True, figsize = (20, 3.5))
    
    x_axis = np.arange(len(x_values))
    x_min_vals = np.ones(len(column_names))
    for model in models:
        for metric in range(len(column_names)):
                x_min_vals[metric] = model[column_names[metric]][1][-1]
     


    for i in range(len(models)):
        model = models[i]
        AUC_list = []
        for metric in range(len(column_names)):
            axs[metric].plot(model[column_names[metric]][1], model[column_names[metric]][2], '--*', label = model_name[i])
            axs[metric].set_xlabel(column_names[metric] + '-D') 
            axs[metric].set_ylabel(column_names[metric] + '-R')
            AUC_list.append(np.round(AUC(model[column_names[metric]][1], model[column_names[metric]][2], x_min_vals[metric]), 5))

        # Print AUC table
        print(model_name[i], ' & ' , AUC_list[0], ' & ', AUC_list[1], ' & ', AUC_list[2], ' & ', AUC_list[3], ' & ', AUC_list[4], ' & ',AUC_list[5], ' \\\\ ')
    fig.legend(labels=model_name, loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=5)
    
    plt.savefig('Figure_3.png', bbox_inches = 'tight')

# Load unnormalized data
"""
BPRMF = load_data('BPRMF', 1, group)
LDA = load_data('LDA', 2, group)
PureSVD = load_data('PureSVD', 3, group)
SLIM = load_data('SLIM', 4, group)
WRMF = load_data('WRMF', 5, group)
all_models = [BPRMF, LDA, PureSVD, SLIM, WRMF]

# Find global min and max across all models per metric per component
global_min, global_max = find_global_min_max(all_models)


# Normalize data
BPRMF = load_data('BPRMF', 1, group, apply_min_max=True, max_ = global_max, min_ = global_min)
LDA = load_data('LDA', 2, group, apply_min_max=True, max_ = global_max, min_ = global_min)
PureSVD = load_data('PureSVD', 3, group, apply_min_max=True, max_ = global_max, min_ = global_min)
SLIM = load_data('SLIM', 4, group, apply_min_max=True, max_ = global_max, min_ = global_min)
WRMF = load_data('WRMF', 5, group, apply_min_max=True, max_ = global_max, min_ = global_min)
all_models_norm = [BPRMF, LDA, PureSVD, SLIM, WRMF]
"""
'''
BPRMF_mm = load_data('BPRMF', 1, group, apply_min_max= True)
#figure_2(BPRMF_mm)

BPRMF_mm = load_data('BPRMF', 1, 'lt', apply_min_max= True)
figure_2(BPRMF_mm)
figure_3([BPRMF_mm])
'''
BPRMF_new = load_data('BPRMF', 1, 'Gender', apply_min_max= True)
figure_2(BPRMF_new)
figure_3([BPRMF_new])
#figure_3(all_models_norm)
