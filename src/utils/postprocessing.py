import json
import numpy as np
import seaborn as sns 
import pandas as pd
import matplotlib.pyplot as plt

from argparse import ArgumentParser

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)
sns.set_theme()
sns.set(font_scale=1.5) 


# args.group = 'Occupation'

def parser_args():
    parser = ArgumentParser(description="JMEF Postprocessing")

    parser.add_argument(
        '--figures',
        type=int,
        nargs='+',
        default=[2,3,4],
        help="Figures to construct"
    )

    parser.add_argument(
        '--fig2_models', 
        type=str, 
        nargs='+', 
        default=['BPRMF'],
        help="Model(s) to construct figure 2"
    )

    parser.add_argument(
        '--fig3_models',
        type=str,
        nargs='+',
        default=['BPRMF', 'LDA', 'PureSVD', 'SLIM', 'WRMF'],
        help="Model(s) to construct figure 3"
    )

    parser.add_argument(
        '--fig4_models', 
        type=str,
        nargs='+', 
        default=['all'], 
        help="Model(s) to construct figure 4"
    )
    
    parser.add_argument(
        '--group',
        type=str,
        default='Age',
        help="User-side group"
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='../outputs/figures/',
        help='Path to output directory'
    )

    return parser.parse_args()

    

def load_data(model, Experiment_nr, group, metric_list, apply_min_max = False, max_ = 0, min_ = 0, fig4 = False):
    """ Load experiments for plotting
    """
    df = pd.DataFrame(columns = column_names)
    df["II"] = [[], [], []]
    for metric in range(len(metric_list)):
        for component in range(len(metric_list[metric])):
            if group == 'Age':
                metrics = json.load(open('../outputs/ml-1m/Experiment_' + str(Experiment_nr) + '_' + model + '/' + group +'/' + metric_list[metric][component]+ '_all_' + model + '_Y.json', 'r'))
                static  = json.load(open('../outputs/ml-1m/Experiment_' + str(Experiment_nr) + '_' + model + '/' + group +'/' + metric_list[metric][component] +'_all_' + model + '_static_Y.json', 'r'))
            elif group == 'Occupation':
                metrics = json.load(open('../outputs/ml-1m/Experiment_' + str(Experiment_nr) + '_' + model + '/' + group +'/' + metric_list[metric][component]+ '_all_' + model + '.json', 'r'))
                static  = json.load(open('../outputs/ml-1m/Experiment_' + str(Experiment_nr) + '_' + model + '/' + group +'/' + metric_list[metric][component] +'_all_' + model + '_static.json', 'r'))
            elif group == 'Gender':
                metrics = json.load(open('../outputs/ml-1m/Experiment_' + str(Experiment_nr) + '_' + model + '/' + group +'/' + metric_list[metric][component]+ '_all_' + model + '.json', 'r'))
                static  = json.load(open('../outputs/ml-1m/Experiment_' + str(Experiment_nr) + '_' + model + '/' + group +'/' + metric_list[metric][component] +'_all_' + model + '_static.json', 'r'))
            elif group == 'lt':
                metrics = json.load(open('../outputs/lt/Experiment_' + str(Experiment_nr) + '_' + model + '/' + metric_list[metric][component]+ '_all_' + model + '.json', 'r'))
                static  = json.load(open('../outputs/lt/Experiment_' + str(Experiment_nr) + '_' + model + '/' + metric_list[metric][component] +'_all_' + model + '_static.json', 'r'))
            else:
                print('Unknown group')

            if fig4:
                df[column_names[metric]][component] = metrics
            else:
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
    """ Trapezoidal Rule for calculating Area under curve
    Input:
        x: x-data
        y: y-data
        minval: set value for which to integrate upto
    
    Output:
        Area: area under curve
    """
    AUC = 0
    
    for i in range(len(x)-1):
        if x[i+1] <= minval:
            AUC += 0.5*(y[i+1]+ y[i])*(x[i+1] - x[i])
    return AUC 

def find_global_min_max(models):
    global_min = np.ones((3, 6))*100
    global_max = np.zeros((3, 6))
    for model in models.values():
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



def figure_2(data, name, column_names):
    """ Reproduce figure 2
    """
    fig, axs = plt.subplots(3, 6, constrained_layout=True, figsize = (20, 10))
    beta_values = ['8', '4', '2', '1', '1/2', '1/4', '1/8', 'ST']
    
    x_axis = np.arange(len(beta_values))

    for metric in range(len(metric_list)):
        for component in range(len(metric_list[metric])):
            metrics = data[column_names[metric]][component]
            sns.barplot(ax=axs[component, metric], x=x_axis - 0.2, y=metrics, palette='Blues_d')
            axs[component, metric].plot(x_axis - 0.2, metrics, '--.', color = 'r')
            axs[component, metric].set_xticks(x_axis, beta_values, rotation = 60)
            axs[component, metric].set_title(metric_list[metric][component])
    fig.supxlabel(r'$\beta$')

    plt.savefig('{}Figure_2_{}_{}.png'.format(args.output_dir, name, args.group), bbox_inches='tight')
    print("Figure 2 saved to {}".format(args.output_dir))
    plt.clf()



def figure_3(models, model_name, column_names):
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
            AUC_list.append(np.round(AUC_trap(model[column_names[metric]][1], model[column_names[metric]][2], x_min_vals[metric]), 5))

        # Print AUC table
        print('AUC Table:')
        print(model_name[i], ' & ' , AUC_list[0], ' & ', AUC_list[1], ' & ', AUC_list[2], ' & ', AUC_list[3], ' & ', AUC_list[4], ' & ',AUC_list[5], ' \\\\ ')
    fig.legend(labels=model_name, loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=5)
    
    plt.savefig('{}Figure_3_{}.png'.format(args.output_dir, args.group), bbox_inches = 'tight')
    print("Figure 3 saved to {}".format(args.output_dir))
    plt.clf()


def figure_4(models, group, column_names):
    """ Reproduce Figure 4 - Kendall correlation heatmap
    """
    df = pd.DataFrame()
    for model in models:
        df = df.append(model)

    for column in column_names:
        df = df.explode(column_names)

    main_metric = df.loc['F'].astype(float)
    disparity = df.loc['D'].astype(float)
    relevance = df.loc['R'].astype(float)

    main_corr = main_metric.corr(method='kendall')
    d_corr = disparity.corr(method='kendall')
    r_corr = relevance.corr(method='kendall')

    j = 0
    for i in [main_corr, d_corr, r_corr]:
        sns.set(font_scale=1.3)
        plot = sns.heatmap(i, cbar=False, annot=True, cmap="YlGnBu", fmt='.3g')
        plt.savefig('{}Figure_4_{}_{}'.format(args.output_dir, group, j))
        
        plt.clf()

        j += 1
    print("Figure 4 saved to {}".format(args.output_dir))

if __name__ == '__main__':
    args = parser_args()

    metric_list = [['IIF','IID','IIR'], 
                   ['IGF','IGD','IGR'],  
                   ['GIF','GID','GIR'],  
                   ['GGF','GGD','GGR'],  
                   ['AIF','AID','AIR'],  
                   ['AGF','AGD','AGR']]

    column_names = ["II", "IG", "GI", "GG", "AI", "AG"]
    x_values = ['8', '4', '2', '1', '1/2', '1/4', '1/8', 'ST']

    # Experiment number of corresponding models
    exp_nr = {
        'BPRMF': 1, 'LDA': 2, 'PureSVD': 3,'SLIM': 4,
        'WRMF': 5, 'CHI2': 6, 'HT': 7, 'KLD': 8,
        'LMWI': 9, 'LMWU': 10, 'SVD': 11, 'NNI': 12,
        'NNU': 13, 'PLSA': 14, 'Random': 15, 'RM1': 16,
        'RM2': 17, 'RSV': 18, 'RW': 19, 'UIR': 20,
        'Bert4Rec': 21
    }

    if 2 in args.figures:
        print("\nReconstruct figure 2 on group: {}".format(args.group))
        print("Models: {}".format(args.fig2_models))
        # Load unnormalized data
        all_models = {}
        for model_name in args.fig2_models:
            model_data = load_data(model_name, exp_nr[model_name], args.group, metric_list)
            all_models[model_name] = model_data

        # Find global min and max across all models per metric per component
        global_min, global_max = find_global_min_max(all_models)

        all_models_norm = {}
        for model_name in args.fig2_models:
            model_data = load_data(model_name, exp_nr[model_name], args.group, metric_list, apply_min_max=True, max_ = global_max, min_ = global_min)
            all_models_norm[model_name] = model_data
    
        for name, data in all_models_norm.items():
            figure_2(data, name, column_names)

        
    if 3 in args.figures:
        print("\nReconstruct figure 3 on group: {}".format(args.group))
        print("Models: {}".format(args.fig3_models))
        all_models = {}
        for model_name in args.fig3_models:
            model_data = load_data(model_name, exp_nr[model_name], args.group, metric_list)
            all_models[model_name] = model_data

        # Find global min and max across all models per metric per component
        global_min, global_max = find_global_min_max(all_models)

        all_models_norm = {}
        for model_name in args.fig3_models:
            model_data = load_data(model_name, exp_nr[model_name], args.group, metric_list, apply_min_max=True, max_ = global_max, min_ = global_min)
            all_models_norm[model_name] = model_data
    
        figure_3(list(all_models_norm.values()), list(all_models_norm.keys()), column_names)

    if 4 in args.figures:
        print("\nReconstruct figure 4 on group: {}".format(args.group))
        
        if 'all' in args.fig4_models:
            mod_names = ['BPRMF', 'LDA', 'PureSVD', 'SLIM', 'WRMF', 'CHI2', 'HT', 'KLD', 'LMWI', 'LMWU', 'SVD', 'NNI', 'NNU', 'PLSA', 'Random', 'RM1', 'RM2', 'RSV', 'RW', 'UIR'] 
            print("Models: {}".format(args.fig4_models))
        else:
            mod_names = args.fig4_models

        model_data = []
        for model_name in mod_names:
            model_load = load_data(model_name, exp_nr[model_name], args.group, metric_list, fig4=True)
            model_data.append(model_load)

        figure_4(model_data, args.group, column_names)
