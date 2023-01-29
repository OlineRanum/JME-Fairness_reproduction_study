

python3 run_metric.py --data ml-1m --model CHI2 --conduct sh --age N
python3 run_metric.py --data ml-1m --model CHI2 --conduct st --age N
mkdir save_exp/ml-1m/Experiment_7_CHI2/Gender
mv save_exp/ml-1m/*.json save_exp/ml-1m/Gender/Experiment_7_CHI2
python3 run_metric.py --data ml-1m --model CHI2 --conduct sh --age Y
python3 run_metric.py --data ml-1m --model CHI2 --conduct st --age Y
mkdir save_exp/ml-1m/Experiment_7_CHI2/Age/
mv save_exp/ml-1m/*.json save_exp/ml-1m/Experiment_7_CHI2/Age

python3 run_metric.py --data ml-1m --model HT --conduct sh --age N
python3 run_metric.py --data ml-1m --model HT --conduct st --age N
mkdir save_exp/ml-1m/Experiment_8_HT/Gender
mv save_exp/ml-1m/*.json save_exp/ml-1m/Gender/Experiment_8_HT
python3 run_metric.py --data ml-1m --model HT --conduct sh --age Y
python3 run_metric.py --data ml-1m --model HT --conduct st --age Y
mkdir save_exp/ml-1m/Experiment_8_HT/Age
mv save_exp/ml-1m/*.json save_exp/ml-1m/Experiment_8_HT/Age

python3 run_metric.py --data ml-1m --model KLD --conduct sh --age N
python3 run_metric.py --data ml-1m --model KLD --conduct st --age N
mkdir save_exp/ml-1m/Experiment_9_KLD/Gender
mv save_exp/ml-1m/*.json save_exp/ml-1m/Gender/Experiment_9_KLD
python3 run_metric.py --data ml-1m --model KLD --conduct sh --age Y
python3 run_metric.py --data ml-1m --model KLD --conduct st --age Y
mkdir save_exp/ml-1m/Experiment_9_KLD/Age
mv save_exp/ml-1m/*.json save_exp/ml-1m/Experiment_9_KLD/Age

python3 run_metric.py --data ml-1m --model LMWI --conduct sh --age N
python3 run_metric.py --data ml-1m --model LMWI --conduct st --age N
mkdir save_exp/ml-1m/Experiment_10_LMWI/Gender
mv save_exp/ml-1m/*.json save_exp/ml-1m/Gender/Experiment_10_LMWI
python3 run_metric.py --data ml-1m --model LMWI --conduct sh --age Y
python3 run_metric.py --data ml-1m --model LMWI --conduct st --age Y
mkdir save_exp/ml-1m/Experiment_10_LMWI/Age
mv save_exp/ml-1m/*.json save_exp/ml-1m/Experiment_10_LMWI/Age

python3 run_metric.py --data ml-1m --model LMWU --conduct sh --age N
python3 run_metric.py --data ml-1m --model LMWU --conduct st --age N
mkdir save_exp/ml-1m/Experiment_11_LMWU/Gender
mv save_exp/ml-1m/*.json save_exp/ml-1m/Gender/Experiment_11_LMWU
python3 run_metric.py --data ml-1m --model LMWU --conduct sh --age Y
python3 run_metric.py --data ml-1m --model LMWU --conduct st --age Y
mkdir save_exp/ml-1m/Experiment_11_LMWU/Age
mv save_exp/ml-1m/*.json save_exp/ml-1m/Experiment_11_LMWU/Age

python3 run_metric.py --data ml-1m --model SVD --conduct sh --age N
python3 run_metric.py --data ml-1m --model SVD --conduct st --age N
mkdir save_exp/ml-1m/Experiment_12_SVD/Gender
mv save_exp/ml-1m/*.json save_exp/ml-1m/Gender/Experiment_12_SVD
python3 run_metric.py --data ml-1m --model SVD --conduct sh --age Y
python3 run_metric.py --data ml-1m --model SVD --conduct st --age Y
mkdir save_exp/ml-1m/Experiment_12_SVD/Age
mv save_exp/ml-1m/*.json save_exp/ml-1m/Experiment_12_SVD/Age

python3 run_metric.py --data ml-1m --model NNI --conduct sh --age N
python3 run_metric.py --data ml-1m --model NNI --conduct st --age N
mkdir save_exp/ml-1m/Experiment_13_NNI/Gender
mv save_exp/ml-1m/*.json save_exp/ml-1m/Gender/Experiment_13_NNI
python3 run_metric.py --data ml-1m --model NNI --conduct sh --age Y
python3 run_metric.py --data ml-1m --model NNI --conduct st --age Y
mkdir save_exp/ml-1m/Experiment_13_NNI/Age
mv save_exp/ml-1m/*.json save_exp/ml-1m/Experiment_13_NNI/Age

python3 run_metric.py --data ml-1m --model NNU --conduct sh --age N
python3 run_metric.py --data ml-1m --model NNU --conduct st --age N
mkdir save_exp/ml-1m/Experiment_14_NNU/Gender
mv save_exp/ml-1m/*.json save_exp/ml-1m/Gender/Experiment_14_NNU
python3 run_metric.py --data ml-1m --model NNU --conduct sh --age Y
python3 run_metric.py --data ml-1m --model NNU --conduct st --age Y
mkdir save_exp/ml-1m/Experiment_14_NNU/Age
mv save_exp/ml-1m/*.json save_exp/ml-1m/Experiment_14_NNU/Age

python3 run_metric.py --data ml-1m --model PLSA --conduct sh --age N
python3 run_metric.py --data ml-1m --model PLSA --conduct st --age N
mkdir save_exp/ml-1m/Experiment_15_PLSA/Gender
mv save_exp/ml-1m/*.json save_exp/ml-1m/Gender/Experiment_15_PLSA
python3 run_metric.py --data ml-1m --model PLSA --conduct sh --age Y
python3 run_metric.py --data ml-1m --model PLSA --conduct st --age Y
mkdir save_exp/ml-1m/Experiment_15_PLSA/Age
mv save_exp/ml-1m/*.json save_exp/ml-1m/Experiment_15_PLSA/Age

python3 run_metric.py --data ml-1m --model POP --conduct sh --age N
python3 run_metric.py --data ml-1m --model POP --conduct st --age N
mkdir save_exp/ml-1m/Experiment_16_POP/Gender
mv save_exp/ml-1m/*.json save_exp/ml-1m/Gender/Experiment_16_POP
python3 run_metric.py --data ml-1m --model POP --conduct sh --age Y
python3 run_metric.py --data ml-1m --model POP --conduct st --age Y
mkdir save_exp/ml-1m/Experiment_16_POP/Age
mv save_exp/ml-1m/*.json save_exp/ml-1m/Experiment_16_POP/Age

python3 run_metric.py --data ml-1m --model Random --conduct sh --age N
python3 run_metric.py --data ml-1m --model Random --conduct st --age N
mkdir save_exp/ml-1m/Experiment_17_Random/Gender
mv save_exp/ml-1m/*.json save_exp/ml-1m/Gender/Experiment_17_Random
python3 run_metric.py --data ml-1m --model Random --conduct sh --age Y
python3 run_metric.py --data ml-1m --model Random --conduct st --age Y
mkdir save_exp/ml-1m/Experiment_17_Random/Age
mv save_exp/ml-1m/*.json save_exp/ml-1m/Experiment_17_Random/Age

python3 run_metric.py --data ml-1m --model RM1 --conduct sh --age N
python3 run_metric.py --data ml-1m --model RM1 --conduct st --age N
mkdir save_exp/ml-1m/Experiment_18_RM1/Gender
mv save_exp/ml-1m/*.json save_exp/ml-1m/Gender/Experiment_18_RM1
python3 run_metric.py --data ml-1m --model RM1 --conduct sh --age Y
python3 run_metric.py --data ml-1m --model RM1 --conduct st --age Y
mkdir save_exp/ml-1m/Experiment_18_RM1/Age
mv save_exp/ml-1m/*.json save_exp/ml-1m/Experiment_18_RM1/Age

python3 run_metric.py --data ml-1m --model RM2 --conduct sh --age N
python3 run_metric.py --data ml-1m --model RM2 --conduct st --age N
mkdir save_exp/ml-1m/Experiment_19_RM2/Gender
mv save_exp/ml-1m/*.json save_exp/ml-1m/Gender/Experiment_19_RM2
python3 run_metric.py --data ml-1m --model RM2 --conduct sh --age Y
python3 run_metric.py --data ml-1m --model RM2 --conduct st --age Y
mkdir save_exp/ml-1m/Experiment_19_RM2/Age
mv save_exp/ml-1m/*.json save_exp/ml-1m/Experiment_19_RM2/Age

python3 run_metric.py --data ml-1m --model RSV --conduct sh --age N
python3 run_metric.py --data ml-1m --model RSV --conduct st --age N
mkdir save_exp/ml-1m/Experiment_20_RSV/Gender
mv save_exp/ml-1m/*.json save_exp/ml-1m/Gender/Experiment_20_RSV
python3 run_metric.py --data ml-1m --model RSV --conduct sh --age Y
python3 run_metric.py --data ml-1m --model RSV --conduct st --age Y
mkdir save_exp/ml-1m/Experiment_20_RSV/Age
mv save_exp/ml-1m/*.json save_exp/ml-1m/Experiment_20_RSV/Age

python3 run_metric.py --data ml-1m --model RW --conduct sh --age N
python3 run_metric.py --data ml-1m --model RW --conduct st --age N
mkdir save_exp/ml-1m/Experiment_21_RW/Gender
mv save_exp/ml-1m/*.json save_exp/ml-1m/Gender/Experiment_21_RW
python3 run_metric.py --data ml-1m --model RW --conduct sh --age Y
python3 run_metric.py --data ml-1m --model RW --conduct st --age Y
mkdir save_exp/ml-1m/Experiment_21_RW/Age
mv save_exp/ml-1m/*.json save_exp/ml-1m/Experiment_21_RW/Age

python3 run_metric.py --data ml-1m --model UIR --conduct sh --age N
python3 run_metric.py --data ml-1m --model UIR --conduct st --age N
mkdir save_exp/ml-1m/Experiment_22_UIR/Gender
mv save_exp/ml-1m/*.json save_exp/ml-1m/Gender/Experiment_22_UIR
python3 run_metric.py --data ml-1m --model UIR --conduct sh --age Y
python3 run_metric.py --data ml-1m --model UIR --conduct st --age Y
mkdir save_exp/ml-1m/Experiment_22_UIR/Age
mv save_exp/ml-1m/*.json save_exp/ml-1m/Experiment_22_UIR/Age


