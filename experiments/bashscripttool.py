# Write experiment GPU time to file
models = ['BPRMF', 'LDA', 'PureSVD', 'SLIM', 'WRMF', 'CHI2', 'HT', 'KLD', 'LMWI', 'LMWU', 'SVD', 'NNI', 'NNU', 'PLSA', 'POP', 'Random', 'RM1', 'RM2', 'RSV', 'RW', 'UIR']


with open("run_experiments.sh", "w") as f:
    for i in range(len(models)):
        f.write('mkdir outputs/ml-1m/Experiment_' + str(i+1) +'_' + models[i] +'\n')
        f.write('python3 run_metric.py --data ml-1m --model '+ models[i]+' --conduct sh --age N\n')
        f.write('python3 run_metric.py --data ml-1m --model '+ models[i]+' --conduct st --age N\n')
        f.write('mkdir outputs/ml-1m/Experiment_' + str(i+1) +'_' + models[i] +'/Gender\n')
        f.write('mv outputs/ml-1m/*.json outputs/ml-1m/Experiment_' + str(i+1) +'_' + models[i] +'/Gender\n')
        f.write('python3 run_metric.py --data ml-1m --model '+ models[i] +' --conduct sh --age Y\n')
        f.write('python3 run_metric.py --data ml-1m --model '+ models[i]+' --conduct st --age Y\n')
        f.write('mkdir outputs/ml-1m/Experiment_' + str(i+1) +'_' + models[i] +'/Age\n')
        f.write('mv outputs/ml-1m/*.json outputs/ml-1m/Experiment_' + str(i+1) +'_' + models[i] +'/Age\n\n')
