import json
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt



def figure_2():
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)

    fig, axs = plt.subplots(3, 6, constrained_layout=True)
    metric_list = [['IIF', 'IID', 'IIR'], ['IGF','IGD','IGR'],  ['GIF','GID','GIR'],  ['GGF','GGD','GGR'],  ['AIF','AID','AIR'],  ['AGF','AGD','AGR']]
    x_values = ['8', '4', '2', '1', '1/2', '1/4', '1/8', 'ST']

    x_axis = np.arange(len(x_values))

    for metric in range(len(metric_list)):
        for component in range(len(metric_list[metric])):
            metrics = json.load(open('./save_exp/ml-1m/' +metric_list[metric][component]+ '_all_BPRMF_Y.json', 'r'))
            static = json.load(open('./save_exp/ml-1m/' + metric_list[metric][component] +'_all_BPRMF_static_Y.json', 'r'))
            metrics.extend(static)
    
            axs[component, metric].bar(x_axis - 0.2, metrics)
            axs[component, metric].set_xticks(x_axis, x_values, rotation = 60)
            axs[component, metric].set_title(metric_list[metric][component])
    plt.show()

figure_2()