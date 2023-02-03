mkdir src/outputs/ml-1m/Experiment_1_BPRMF
python3 src/run_metric.py --data ml-1m --model BPRMF --conduct sh --age N
python3 src/run_metric.py --data ml-1m --model BPRMF --conduct st --age N
mkdir src/outputs/ml-1m/Experiment_1_BPRMF/Gender
mv src/outputs/ml-1m/*.json outputs/ml-1m/Experiment_1_BPRMF/Gender
python3 src/run_metric.py --data ml-1m --model BPRMF --conduct sh --age Y
python3 src/run_metric.py --data ml-1m --model BPRMF --conduct st --age Y
mkdir src/outputs/ml-1m/Experiment_1_BPRMF/Age
mv src/outputs/ml-1m/*.json outputs/ml-1m/Experiment_1_BPRMF/Age

mkdir src/outputs/ml-1m/Experiment_2_LDA
python3 src/run_metric.py --data ml-1m --model LDA --conduct sh --age N
python3 src/run_metric.py --data ml-1m --model LDA --conduct st --age N
mkdir src/outputs/ml-1m/Experiment_2_LDA/Gender
mv src/outputs/ml-1m/*.json outputs/ml-1m/Experiment_2_LDA/Gender
python3 src/run_metric.py --data ml-1m --model LDA --conduct sh --age Y
python3 src/run_metric.py --data ml-1m --model LDA --conduct st --age Y
mkdir src/outputs/ml-1m/Experiment_2_LDA/Age
mv src/outputs/ml-1m/*.json outputs/ml-1m/Experiment_2_LDA/Age

mkdir src/outputs/ml-1m/Experiment_3_PureSVD
python3 src/run_metric.py --data ml-1m --model PureSVD --conduct sh --age N
python3 src/run_metric.py --data ml-1m --model PureSVD --conduct st --age N
mkdir src/outputs/ml-1m/Experiment_3_PureSVD/Gender
mv src/outputs/ml-1m/*.json outputs/ml-1m/Experiment_3_PureSVD/Gender
python3 src/run_metric.py --data ml-1m --model PureSVD --conduct sh --age Y
python3 src/run_metric.py --data ml-1m --model PureSVD --conduct st --age Y
mkdir src/outputs/ml-1m/Experiment_3_PureSVD/Age
mv src/outputs/ml-1m/*.json outputs/ml-1m/Experiment_3_PureSVD/Age

mkdir src/outputs/ml-1m/Experiment_4_SLIM
python3 src/run_metric.py --data ml-1m --model SLIM --conduct sh --age N
python3 src/run_metric.py --data ml-1m --model SLIM --conduct st --age N
mkdir src/outputs/ml-1m/Experiment_4_SLIM/Gender
mv src/outputs/ml-1m/*.json outputs/ml-1m/Experiment_4_SLIM/Gender
python3 src/run_metric.py --data ml-1m --model SLIM --conduct sh --age Y
python3 src/run_metric.py --data ml-1m --model SLIM --conduct st --age Y
mkdir src/outputs/ml-1m/Experiment_4_SLIM/Age
mv src/outputs/ml-1m/*.json outputs/ml-1m/Experiment_4_SLIM/Age

mkdir src/outputs/ml-1m/Experiment_5_WRMF
python3 src/run_metric.py --data ml-1m --model WRMF --conduct sh --age N
python3 src/run_metric.py --data ml-1m --model WRMF --conduct st --age N
mkdir src/outputs/ml-1m/Experiment_5_WRMF/Gender
mv src/outputs/ml-1m/*.json outputs/ml-1m/Experiment_5_WRMF/Gender
python3 src/run_metric.py --data ml-1m --model WRMF --conduct sh --age Y
python3 src/run_metric.py --data ml-1m --model WRMF --conduct st --age Y
mkdir src/outputs/ml-1m/Experiment_5_WRMF/Age
mv src/outputs/ml-1m/*.json outputs/ml-1m/Experiment_5_WRMF/Age

mkdir src/outputs/ml-1m/Experiment_6_CHI2
python3 src/run_metric.py --data ml-1m --model CHI2 --conduct sh --age N
python3 src/run_metric.py --data ml-1m --model CHI2 --conduct st --age N
mkdir src/outputs/ml-1m/Experiment_6_CHI2/Gender
mv src/outputs/ml-1m/*.json outputs/ml-1m/Experiment_6_CHI2/Gender
python3 src/run_metric.py --data ml-1m --model CHI2 --conduct sh --age Y
python3 src/run_metric.py --data ml-1m --model CHI2 --conduct st --age Y
mkdir src/outputs/ml-1m/Experiment_6_CHI2/Age
mv src/outputs/ml-1m/*.json outputs/ml-1m/Experiment_6_CHI2/Age

mkdir src/outputs/ml-1m/Experiment_7_HT
python3 src/run_metric.py --data ml-1m --model HT --conduct sh --age N
python3 src/run_metric.py --data ml-1m --model HT --conduct st --age N
mkdir src/outputs/ml-1m/Experiment_7_HT/Gender
mv src/outputs/ml-1m/*.json outputs/ml-1m/Experiment_7_HT/Gender
python3 src/run_metric.py --data ml-1m --model HT --conduct sh --age Y
python3 src/run_metric.py --data ml-1m --model HT --conduct st --age Y
mkdir src/outputs/ml-1m/Experiment_7_HT/Age
mv src/outputs/ml-1m/*.json outputs/ml-1m/Experiment_7_HT/Age

mkdir src/outputs/ml-1m/Experiment_8_KLD
python3 src/run_metric.py --data ml-1m --model KLD --conduct sh --age N
python3 src/run_metric.py --data ml-1m --model KLD --conduct st --age N
mkdir src/outputs/ml-1m/Experiment_8_KLD/Gender
mv src/outputs/ml-1m/*.json outputs/ml-1m/Experiment_8_KLD/Gender
python3 src/run_metric.py --data ml-1m --model KLD --conduct sh --age Y
python3 src/run_metric.py --data ml-1m --model KLD --conduct st --age Y
mkdir src/outputs/ml-1m/Experiment_8_KLD/Age
mv src/outputs/ml-1m/*.json outputs/ml-1m/Experiment_8_KLD/Age

mkdir src/outputs/ml-1m/Experiment_9_LMWI
python3 src/run_metric.py --data ml-1m --model LMWI --conduct sh --age N
python3 src/run_metric.py --data ml-1m --model LMWI --conduct st --age N
mkdir src/outputs/ml-1m/Experiment_9_LMWI/Gender
mv src/outputs/ml-1m/*.json outputs/ml-1m/Experiment_9_LMWI/Gender
python3 src/run_metric.py --data ml-1m --model LMWI --conduct sh --age Y
python3 src/run_metric.py --data ml-1m --model LMWI --conduct st --age Y
mkdir src/outputs/ml-1m/Experiment_9_LMWI/Age
mv src/outputs/ml-1m/*.json outputs/ml-1m/Experiment_9_LMWI/Age

mkdir src/outputs/ml-1m/Experiment_10_LMWU
python3 src/run_metric.py --data ml-1m --model LMWU --conduct sh --age N
python3 src/run_metric.py --data ml-1m --model LMWU --conduct st --age N
mkdir src/outputs/ml-1m/Experiment_10_LMWU/Gender
mv src/outputs/ml-1m/*.json outputs/ml-1m/Experiment_10_LMWU/Gender
python3 src/run_metric.py --data ml-1m --model LMWU --conduct sh --age Y
python3 src/run_metric.py --data ml-1m --model LMWU --conduct st --age Y
mkdir src/outputs/ml-1m/Experiment_10_LMWU/Age
mv src/outputs/ml-1m/*.json outputs/ml-1m/Experiment_10_LMWU/Age

mkdir src/outputs/ml-1m/Experiment_11_SVD
python3 src/run_metric.py --data ml-1m --model SVD --conduct sh --age N
python3 src/run_metric.py --data ml-1m --model SVD --conduct st --age N
mkdir src/outputs/ml-1m/Experiment_11_SVD/Gender
mv src/outputs/ml-1m/*.json outputs/ml-1m/Experiment_11_SVD/Gender
python3 src/run_metric.py --data ml-1m --model SVD --conduct sh --age Y
python3 src/run_metric.py --data ml-1m --model SVD --conduct st --age Y
mkdir src/outputs/ml-1m/Experiment_11_SVD/Age
mv src/outputs/ml-1m/*.json outputs/ml-1m/Experiment_11_SVD/Age

mkdir src/outputs/ml-1m/Experiment_12_NNI
python3 src/run_metric.py --data ml-1m --model NNI --conduct sh --age N
python3 src/run_metric.py --data ml-1m --model NNI --conduct st --age N
mkdir src/outputs/ml-1m/Experiment_12_NNI/Gender
mv src/outputs/ml-1m/*.json outputs/ml-1m/Experiment_12_NNI/Gender
python3 src/run_metric.py --data ml-1m --model NNI --conduct sh --age Y
python3 src/run_metric.py --data ml-1m --model NNI --conduct st --age Y
mkdir src/outputs/ml-1m/Experiment_12_NNI/Age
mv src/outputs/ml-1m/*.json outputs/ml-1m/Experiment_12_NNI/Age

mkdir src/outputs/ml-1m/Experiment_13_NNU
python3 src/run_metric.py --data ml-1m --model NNU --conduct sh --age N
python3 src/run_metric.py --data ml-1m --model NNU --conduct st --age N
mkdir src/outputs/ml-1m/Experiment_13_NNU/Gender
mv src/outputs/ml-1m/*.json outputs/ml-1m/Experiment_13_NNU/Gender
python3 src/run_metric.py --data ml-1m --model NNU --conduct sh --age Y
python3 src/run_metric.py --data ml-1m --model NNU --conduct st --age Y
mkdir src/outputs/ml-1m/Experiment_13_NNU/Age
mv src/outputs/ml-1m/*.json outputs/ml-1m/Experiment_13_NNU/Age

mkdir src/outputs/ml-1m/Experiment_14_PLSA
python3 src/run_metric.py --data ml-1m --model PLSA --conduct sh --age N
python3 src/run_metric.py --data ml-1m --model PLSA --conduct st --age N
mkdir src/outputs/ml-1m/Experiment_14_PLSA/Gender
mv src/outputs/ml-1m/*.json outputs/ml-1m/Experiment_14_PLSA/Gender
python3 src/run_metric.py --data ml-1m --model PLSA --conduct sh --age Y
python3 src/run_metric.py --data ml-1m --model PLSA --conduct st --age Y
mkdir src/outputs/ml-1m/Experiment_14_PLSA/Age
mv src/outputs/ml-1m/*.json outputs/ml-1m/Experiment_14_PLSA/Age

mkdir src/outputs/ml-1m/Experiment_15_POP
python3 src/run_metric.py --data ml-1m --model POP --conduct sh --age N
python3 src/run_metric.py --data ml-1m --model POP --conduct st --age N
mkdir src/outputs/ml-1m/Experiment_15_POP/Gender
mv src/outputs/ml-1m/*.json outputs/ml-1m/Experiment_15_POP/Gender
python3 src/run_metric.py --data ml-1m --model POP --conduct sh --age Y
python3 src/run_metric.py --data ml-1m --model POP --conduct st --age Y
mkdir src/outputs/ml-1m/Experiment_15_POP/Age
mv src/outputs/ml-1m/*.json outputs/ml-1m/Experiment_15_POP/Age

mkdir src/outputs/ml-1m/Experiment_16_Random
python3 src/run_metric.py --data ml-1m --model Random --conduct sh --age N
python3 src/run_metric.py --data ml-1m --model Random --conduct st --age N
mkdir src/outputs/ml-1m/Experiment_16_Random/Gender
mv src/outputs/ml-1m/*.json outputs/ml-1m/Experiment_16_Random/Gender
python3 src/run_metric.py --data ml-1m --model Random --conduct sh --age Y
python3 src/run_metric.py --data ml-1m --model Random --conduct st --age Y
mkdir src/outputs/ml-1m/Experiment_16_Random/Age
mv src/outputs/ml-1m/*.json outputs/ml-1m/Experiment_16_Random/Age

mkdir src/outputs/ml-1m/Experiment_17_RM1
python3 src/run_metric.py --data ml-1m --model RM1 --conduct sh --age N
python3 src/run_metric.py --data ml-1m --model RM1 --conduct st --age N
mkdir src/outputs/ml-1m/Experiment_17_RM1/Gender
mv src/outputs/ml-1m/*.json outputs/ml-1m/Experiment_17_RM1/Gender
python3 src/run_metric.py --data ml-1m --model RM1 --conduct sh --age Y
python3 src/run_metric.py --data ml-1m --model RM1 --conduct st --age Y
mkdir src/outputs/ml-1m/Experiment_17_RM1/Age
mv src/outputs/ml-1m/*.json outputs/ml-1m/Experiment_17_RM1/Age

mkdir src/outputs/ml-1m/Experiment_18_RM2
python3 src/run_metric.py --data ml-1m --model RM2 --conduct sh --age N
python3 src/run_metric.py --data ml-1m --model RM2 --conduct st --age N
mkdir src/outputs/ml-1m/Experiment_18_RM2/Gender
mv src/outputs/ml-1m/*.json outputs/ml-1m/Experiment_18_RM2/Gender
python3 src/run_metric.py --data ml-1m --model RM2 --conduct sh --age Y
python3 src/run_metric.py --data ml-1m --model RM2 --conduct st --age Y
mkdir src/outputs/ml-1m/Experiment_18_RM2/Age
mv src/outputs/ml-1m/*.json outputs/ml-1m/Experiment_18_RM2/Age

mkdir src/outputs/ml-1m/Experiment_19_RSV
python3 src/run_metric.py --data ml-1m --model RSV --conduct sh --age N
python3 src/run_metric.py --data ml-1m --model RSV --conduct st --age N
mkdir src/outputs/ml-1m/Experiment_19_RSV/Gender
mv src/outputs/ml-1m/*.json outputs/ml-1m/Experiment_19_RSV/Gender
python3 src/run_metric.py --data ml-1m --model RSV --conduct sh --age Y
python3 src/run_metric.py --data ml-1m --model RSV --conduct st --age Y
mkdir src/outputs/ml-1m/Experiment_19_RSV/Age
mv src/outputs/ml-1m/*.json outputs/ml-1m/Experiment_19_RSV/Age

mkdir src/outputs/ml-1m/Experiment_20_RW
python3 src/run_metric.py --data ml-1m --model RW --conduct sh --age N
python3 src/run_metric.py --data ml-1m --model RW --conduct st --age N
mkdir src/outputs/ml-1m/Experiment_20_RW/Gender
mv src/outputs/ml-1m/*.json outputs/ml-1m/Experiment_20_RW/Gender
python3 src/run_metric.py --data ml-1m --model RW --conduct sh --age Y
python3 src/run_metric.py --data ml-1m --model RW --conduct st --age Y
mkdir src/outputs/ml-1m/Experiment_20_RW/Age
mv src/outputs/ml-1m/*.json outputs/ml-1m/Experiment_20_RW/Age

mkdir src/outputs/ml-1m/Experiment_21_UIR
python3 src/run_metric.py --data ml-1m --model UIR --conduct sh --age N
python3 src/run_metric.py --data ml-1m --model UIR --conduct st --age N
mkdir src/outputs/ml-1m/Experiment_21_UIR/Gender
mv src/outputs/ml-1m/*.json outputs/ml-1m/Experiment_21_UIR/Gender
python3 src/run_metric.py --data ml-1m --model UIR --conduct sh --age Y
python3 src/run_metric.py --data ml-1m --model UIR --conduct st --age Y
mkdir src/outputs/ml-1m/Experiment_21_UIR/Age
mv src/outputs/ml-1m/*.json outputs/ml-1m/Experiment_21_UIR/Age

