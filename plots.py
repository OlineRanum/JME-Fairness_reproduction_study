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


def min_max(arr):
    min_ = np.min(arr)
    max_ = np.max(arr)
    min_max = (arr-min_)/(max_-min_)
    return min_max

def AUC(x, y):
    area = 0
    for i in range(len(x)-1):
        area += (x[i+1]- x[i])*y[i+1] - 0.5*(x[i+1]- x[i])*(y[i+1]-y[i])
    return area 

def AUC_trap(x, y):
    area = 0
    for i in range(len(x)-1):
        area += 0.5*(y[i+1]+ y[i])*(x[i+1]- x[i])
    return area 


def load_data(model, metric_list = metric_list):
    
    df = pd.DataFrame(columns = column_names)
    df["II"] = [[], [], []]
    for metric in range(len(metric_list)):
        for component in range(len(metric_list[metric])):
            metrics = json.load(open('./save_exp/ml-1m/Experiment_1/' + model + '/' + metric_list[metric][component]+ '_all_' + model + '_Y.json', 'r'))
            static  = json.load(open('./save_exp/ml-1m/Experiment_1/' + model + '/' + metric_list[metric][component] +'_all_' + model + '_static_Y.json', 'r'))
            metrics.extend(static)
            metrics = min_max(np.array(metrics))
            df[column_names[metric]][component] = metrics

    df.index = ['F', 'D', 'R']
    return df


BPRMF = load_data('BPRMF')
#LDA = load_data('LDA')
#PureSVD = load_data('PureSVD')



def figure_2(model, column_names = column_names):
    fig, axs = plt.subplots(3, 6, constrained_layout=True)
    beta_values = ['8', '4', '2', '1', '1/2', '1/4', '1/8', 'ST']

    x_axis = np.arange(len(beta_values))

    for metric in range(len(metric_list)):
        for component in range(len(metric_list[metric])):
            metrics = model[column_names[metric]][component]
            sns.barplot(ax=axs[component, metric], x=x_axis - 0.2, y=metrics, palette='Blues_d')
            axs[component, metric].plot(x_axis - 0.2, metrics, '--.', color = 'r')
            axs[component, metric].set_xticks(x_axis, beta_values, rotation = 60)
            axs[component, metric].set_title(metric_list[metric][component])

    #plt.suptitle('Figure 2')
    #plt.savefig('Figure2.png', bbox_inches='tight')
    plt.show()

figure_2(BPRMF)


def figure_3(models, model_name = ['BPRMF'], column_names = column_names):
    fig, axs = plt.subplots(1, 6, constrained_layout=True)

    x_axis = np.arange(len(x_values))

    for i in range(len(models)):
        model = models[i]
        for metric in range(len(column_names)):
            axs[metric].plot(model[column_names[metric]][1], model[column_names[metric]][2], '--*', label = model_name[i])
            axs[metric].set_xlabel(column_names[metric] + '-D')
            axs[metric].set_ylabel(column_names[metric] + '-R')
            print('AUC ', column_names[metric],' = ', AUC(model[column_names[metric]][1], model[column_names[metric]][2]))
            print('AUC trap', column_names[metric],' = ', AUC_trap(model[column_names[metric]][1], model[column_names[metric]][2]))
    # Put a legend below current axis
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5)
    plt.show()

#figure_3([BPRMF])
