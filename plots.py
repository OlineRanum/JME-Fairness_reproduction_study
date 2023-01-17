import json
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt


def min_max(arr):
    min_ = np.min(arr)
    max_ = np.max(arr)
    min_max = (arr-min_)/(max_-min_)
    return min_max


def figure_2():
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)

    fig, axs = plt.subplots(3, 6, constrained_layout=True)
    metric_list = [['IIF', 'IID', 'IIR'], ['IGF','IGD','IGR'],  ['GIF','GID','GIR'],  ['GGF','GGD','GGR'],  ['AIF','AID','AIR'],  ['AGF','AGD','AGR']]
    x_values = ['8', '4', '2', '1', '1/2', '1/4', '1/8', 'ST']

    x_axis = np.arange(len(x_values))

    for metric in range(len(metric_list)):
        for component in range(len(metric_list[metric])):
            metrics = json.load(open('./save_exp/ml-1m/BPRMF/' +metric_list[metric][component]+ '_all_BPRMF_Y.json', 'r'))
            static = json.load(open('./save_exp/ml-1m/BPRMF/' + metric_list[metric][component] +'_all_BPRMF_static_Y.json', 'r'))
            metrics.extend(static)
            metrics = min_max(np.array(metrics))

            sns.barplot(ax=axs[component, metric], x=x_axis - 0.2, y=metrics, palette='Blues_d')
            axs[component, metric].plot(x_axis - 0.2, metrics, '--.', color = 'r')
            axs[component, metric].set_xticks(x_axis, x_values, rotation = 60)
            axs[component, metric].set_title(metric_list[metric][component])

    plt.suptitle('Figure 2')
    #plt.savefig('Figure2.png', bbox_inches='tight')
    plt.show()

figure_2()