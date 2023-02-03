# JME-Fairness

This repository reproduces the paper [Joint Multisided Exposure Fairness for Recommendation](https://arxiv.org/abs/2205.00048) (SIGIR 2022) as part of the [ML Reproducibility Challenge 2022](https://paperswithcode.com/rc2022), and an extention of the JME-fairness framework to the neural Bert4Rec model.

![Title](Figures/Figure_5.png)
*The JME-Fairness metrics calculated across a neural Bert4Rec model, for the MovieLens1M (orange) and the LibraryThing (turquoise) dataset.*

## Installation

``` Installing and configuring repo
git clone https://github.com/OlineRanum/FACT.git
cd FACT
bash env/setup.sh
```

## How to use the code
This section provides examples of how to use the repository

### Per experiment
How to run a single experiment
#### To run a single experiment with default setting
```
python3 src/run_metric.py 
```
#### Produce the toy example experiment results
```
python3 src/utils/evaluation_functions/fariness_metrics.py
```

### To run all experiments associated with the reproduction study 
In the subsequent section we provide instructions on running bash-scripts that reproduce all the fairness metric calculations associated with our paper, and how to plot 

#### Reproduce all plots in our reproduction study
```
bash experiments/run_files/plot_reproduction_results.sh
```
## Rerun all metric calculations on MovieLens1M dataset
NB! Expected runtime on RTX 3070 GPU is 13h - this will estimate the fairness metrics across all 21 pretrained models to build the kendal rank correlations for the MovieLens1M dataset. 

```
bash experiments/run_files/run_metrics_movielens.sh
```

## Rerun all metric calculations on LibraryThing dataset
NB! Expected runtime on RTX 3070 GPU is 15h - this will estimate the fairness metrics across all 21 pretrained models to build the kendal rank correlations for the LibraryThing dataset. 

```
bash experiments/run_files/run_metrics_librarything.sh
```

## Train Bert4Rec
```
bash 
```


## Directory Overview
### Folder
The [`env`](./env) contains the environment to run this codebase and a bash script _setup.sh_ to download the LibraryTh
ing dataset and install/activate the environment. 

The [`experiments`](./experiments) contains the run files with all the bash scripts to reproduce all experiments associated with the reproduction study. 

The [`Figures`](./Figures) contains the coverpicture.

The [`src`](./src) contains the source code of the project. 

## Datasets 

The [`data`](./data) contains the datasets we used from [here](https://grouplens.org/datasets/movielens/).

The [`saved_model`](./saved_model) contains the pre-trained model from [here](https://github.com/dvalcarce/evalMetrics).



## Acknowledgements



## Citation of the original paper
```
@inproceedings{wu2022joint,
  title={Joint Multisided Exposure Fairness for Recommendation},
  author={Wu, Haolun and Mitra, Bhaskar and Ma, Chen and Diaz, Fernando and Liu, Xue},
  booktitle={SIGIR},
  publisher = {{ACM}},
  year={2022}
}
```

