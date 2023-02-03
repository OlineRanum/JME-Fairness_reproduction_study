# JME-Fairness

This repository reproduces the paper [Joint Multisided Exposure Fairness for Recommendation](https://arxiv.org/abs/2205.00048) (SIGIR 2022) as part of the [ML Reproducibility Challenge 2022](https://paperswithcode.com/rc2022).

![Title](Figures/Figure_3_Age.png)
*caption*

## Installation

``` Installing and configuring repo
git clone https://github.com/OlineRanum/FACT.git
cd FACT
bash src/setup.sh
```

## Usage Example

### To run a single experiment with default setting

### To run all experiments associated with the reproduction study 

## Code Overview
### Folder
The [`data`](./data) contains the datasets we used from [here](https://grouplens.org/datasets/movielens/).

The [`saved_model`](./saved_model) contains the pre-trained model from [here](https://github.com/dvalcarce/evalMetrics).

### File
The [`read_data.py`](./read_data.py) contains the data reading and preprocessing.
The [`Disparity_Metrics.py`](./Disparity_Metrics.py) contains the implementation of the proposed JME-Fairness metrics and outputs results for the toy example of job recommendation systems experiment.
The [`run_metric.py`](./run_metric.py) outputs the output values for different JME-Fairness metrics.
The [`postprocessing.py`](./postprocessing.py) produces all the results for the JME-fairness metrics analysis experiments.




## Produce the toy example experiment results
```
python Disparity_Metrics.py
```
## Reproduce all plots in our reproduction study
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
bash experiments/run_files/run_metrics_movielens.sh
```

## Obtain the values for JME-fairness metrics
```
python run_metric.py
```

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

