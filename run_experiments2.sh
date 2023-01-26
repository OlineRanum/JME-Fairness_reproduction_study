#!/bin/sh

python3 run_metric.py --data ml-1m --model BPRMF --conduct sh --age NN
python3 run_metric.py --data ml-1m --model BPRMF --conduct st --age NN
mkdir save_exp/ml-1m/Experiment_2/Occupation
mv save_exp/ml-1m/*.json save_exp/ml-1m/Occupation/Experiment_2

python3 run_metric.py --data ml-1m --model LDA --conduct sh --age NN
python3 run_metric.py --data ml-1m --model LDA --conduct st --age NN
mkdir save_exp/ml-1m/Experiment_3_LDA/Occupation
mv save_exp/ml-1m/*.json save_exp/ml-1m/Occupation/Experiment_3_LDA

python3 run_metric.py --data ml-1m --model PureSVD --conduct sh --age NN
python3 run_metric.py --data ml-1m --model PureSVD --conduct st --age NN
mkdir save_exp/ml-1m/Experiment_4_PureSVD/Occupation
mv save_exp/ml-1m/*.json save_exp/ml-1m/Occupation/Experiment_4_PureSVD

python3 run_metric.py --data ml-1m --model SLIM --conduct sh --age NN
python3 run_metric.py --data ml-1m --model SLIM --conduct st --age NN
mkdir save_exp/ml-1m/Experiment_5_SLIM/Occupation
mv save_exp/ml-1m/*.json save_exp/ml-1m/Occupation/Experiment_5_SLIM

python3 run_metric.py --data ml-1m --model WRMF --conduct sh --age NN
python3 run_metric.py --data ml-1m --model WRMF --conduct st --age NN
mkdir save_exp/ml-1m/Experiment_6_WRMF/Occupation
mv save_exp/ml-1m/*.json save_exp/ml-1m/Occupation/Experiment_6_WRMF

python3 run_metric.py --data ml-1m --model CHI2 --conduct sh --age NN
python3 run_metric.py --data ml-1m --model CHI2 --conduct st --age NN
mkdir save_exp/ml-1m/Experiment_7_CHI2/Occupation
mv save_exp/ml-1m/*.json save_exp/ml-1m/Occupation/Experiment_7_CHI2

python3 run_metric.py --data ml-1m --model HT --conduct sh --age NN
python3 run_metric.py --data ml-1m --model HT --conduct st --age NN
mkdir save_exp/ml-1m/Experiment_8_HT/Occupation
mv save_exp/ml-1m/*.json save_exp/ml-1m/Occupation/Experiment_8_HT


python3 run_metric.py --data ml-1m --model KLD --conduct sh --age NN
python3 run_metric.py --data ml-1m --model KLD --conduct st --age NN
mkdir save_exp/ml-1m/Experiment_9_KLD/Occupation
mv save_exp/ml-1m/*.json save_exp/ml-1m/Occupation/Experiment_9_KLD

python3 run_metric.py --data ml-1m --model LMWI --conduct sh --age NN
python3 run_metric.py --data ml-1m --model LMWI --conduct st --age NN
mkdir save_exp/ml-1m/Experiment_10_LMWI/Occupation
mv save_exp/ml-1m/*.json save_exp/ml-1m/Occupation/Experiment_10_LMWI

python3 run_metric.py --data ml-1m --model LMWU --conduct sh --age NN
python3 run_metric.py --data ml-1m --model LMWU --conduct st --age NN
mkdir save_exp/ml-1m/Experiment_11_LMWU/Occupation
mv save_exp/ml-1m/*.json save_exp/ml-1m/Occupation/Experiment_11_LMWU

python3 run_metric.py --data ml-1m --model SVD --conduct sh --age NN
python3 run_metric.py --data ml-1m --model SVD --conduct st --age NN
mkdir save_exp/ml-1m/Experiment_12_SVD/Occupation
mv save_exp/ml-1m/*.json save_exp/ml-1m/Occupation/Experiment_12_SVD

python3 run_metric.py --data ml-1m --model NNI --conduct sh --age NN
python3 run_metric.py --data ml-1m --model NNI --conduct st --age NN
mkdir save_exp/ml-1m/Experiment_13_NNI/Occupation
mv save_exp/ml-1m/*.json save_exp/ml-1m/Occupation/Experiment_13_NNI

python3 run_metric.py --data ml-1m --model NNU --conduct sh --age NN
python3 run_metric.py --data ml-1m --model NNU --conduct st --age NN
mkdir save_exp/ml-1m/Experiment_14_NNU/Occupation
mv save_exp/ml-1m/*.json save_exp/ml-1m/Occupation/Experiment_14_NNU

python3 run_metric.py --data ml-1m --model PLSA --conduct sh --age NN
python3 run_metric.py --data ml-1m --model PLSA --conduct st --age NN
mkdir save_exp/ml-1m/Experiment_15_PLSA/Occupation
mv save_exp/ml-1m/*.json save_exp/ml-1m/Occupation/Experiment_15_PLSA

python3 run_metric.py --data ml-1m --model POP --conduct sh --age NN
python3 run_metric.py --data ml-1m --model POP --conduct st --age NN
mkdir save_exp/ml-1m/Experiment_16_POP/Occupation
mv save_exp/ml-1m/*.json save_exp/ml-1m/Occupation/Experiment_16_POP

python3 run_metric.py --data ml-1m --model Random --conduct sh --age NN
python3 run_metric.py --data ml-1m --model Random --conduct st --age NN
mkdir save_exp/ml-1m/Experiment_17_Random/Occupation
mv save_exp/ml-1m/*.json save_exp/ml-1m/Occupation/Experiment_17_Random

python3 run_metric.py --data ml-1m --model RM1 --conduct sh --age NN
python3 run_metric.py --data ml-1m --model RM1 --conduct st --age NN
mkdir save_exp/ml-1m/Experiment_18_RM1/Occupation
mv save_exp/ml-1m/*.json save_exp/ml-1m/Occupation/Experiment_18_RM1

python3 run_metric.py --data ml-1m --model RM2 --conduct sh --age NN
python3 run_metric.py --data ml-1m --model RM2 --conduct st --age NN
mkdir save_exp/ml-1m/Experiment_19_RM2/Occupation
mv save_exp/ml-1m/*.json save_exp/ml-1m/Occupation/Experiment_19_RM2

python3 run_metric.py --data ml-1m --model RSV --conduct sh --age NN
python3 run_metric.py --data ml-1m --model RSV --conduct st --age NN
mkdir save_exp/ml-1m/Experiment_20_RSV/Occupation
mv save_exp/ml-1m/*.json save_exp/ml-1m/Occupation/Experiment_20_RSV

python3 run_metric.py --data ml-1m --model RW --conduct sh --age NN
python3 run_metric.py --data ml-1m --model RW --conduct st --age NN
mkdir save_exp/ml-1m/Experiment_21_RW/Occupation
mv save_exp/ml-1m/*.json save_exp/ml-1m/Occupation/Experiment_21_RW

python3 run_metric.py --data ml-1m --model UIR --conduct sh --age NN
python3 run_metric.py --data ml-1m --model UIR --conduct st --age NN
mkdir save_exp/ml-1m/Experiment_22_UIR/Occupation
mv save_exp/ml-1m/*.json save_exp/ml-1m/Occupation/Experiment_22_UIR



