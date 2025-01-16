# DP Tabular Data Synthesis Benchmark
This is a benchmark for dp tabular data synthesis. The necessary code for the paper is all included in this repository. 

## Introduction
This benchmark is based on the following algorithms.
|Algorithms | Link |
|-----------|------|
|AIM        |[AIM: An Adaptive and Iterative Mechanism for Differentially Private Synthetic Data](https://arxiv.org/pdf/2201.12677)|
|DP-MERF    |[DP-MERF: Differentially Private Mean Embeddings with Random Features for Practical Privacy-Preserving Data Generation](https://proceedings.mlr.press/v130/harder21a/harder21a.pdf)|
|GEM        |[Iterative Methods for Private Synthetic Data: Unifying Framework and New Methods](https://proceedings.neurips.cc/paper/2021/file/0678c572b0d5597d2d4a6b5bd135754c-Paper.pdf)|
|Private-GSD|[Generating Private Synthetic Data with Genetic Algorithms](https://proceedings.mlr.press/v202/liu23ag/liu23ag.pdf)|
|PrivMRF    |[Data Synthesis via Differentially Private Markov Random Fields](https://www.vldb.org/pvldb/vol14/p2190-cai.pdf)| 
|PrivSyn    |[PrivSyn: Differentially Private Data Synthesis](https://www.usenix.org/system/files/sec21fall-zhang-zhikun.pdf)|
|RAP++      |[Private Synthetic Data for Multitask Learning and Marginal Queries](https://proceedings.neurips.cc/paper_files/paper/2022/file/7428310c0f97f1c6bb2ef1be99c1ec2a-Paper-Conference.pdf)|
|TabDDPM    |[TabDDPM: Modelling Tabular Data with Diffusion Models](https://proceedings.mlr.press/v202/kotelnikov23a/kotelnikov23a.pdf)|


## Quick Start 
### Hyper-parameter Introduction
The code for running experiments is in main.py. The detailed description of the hyper-parameters are give as follows.
* `method`: which synthesis method you will run.
* `dataset`: name of dataset.
* `device`: the device used for running algorithms. 
* `epsilon`: DP parameter, which must be delivered when running code. 
* `--delta`: DP parameter, which is set to $1e-5$ by default.
* `--num_preprocess`: preprocessing method for numerical attributes, which is set to uniform binning by default. 
* `--rare_threshold`: threshold of preprocessing method for categorical attributes, which is set to $0.2\%$ by default.
* `--sample_device`: device used for sample data, by default is set to the same as running device.
* `--test`: hyper-parameter used for testing and debug. 


### Preparation
The necessary packages for the environment are listed in file `requirement.txt`. Firstly, make sure the datasets are put in the correct fold (in the following examples, the fold is `data/bank`, and the necessary dataset has already been provided). In this repository, the evaluation model is already tuned so users do not need any operation. Otherwise, you should tune the evaluation model (using the following code) before any further operation.
```
python evaluator/tune_eval_model.py bank mlp cv cuda:0
```


### Overall Evaluation
After you activate your enviroment, try the following code to make an overall evaluation. In our paper, we by default set `num_preprocess` to be "uniform_kbins" except for DP-MERF and TabDDPM, and set `rare_threshold` to 0.002 for overall evaluation. 
```
python main.py aim bank cuda:0 1.0 --num_preprocess uniform_kbins --rare_threshold 0.002
```


### Preprocessing Investigation
If you want to try other preprocessing methods or preprocessing hyper-parameter settings, you can modify the value of preprocessing hyper-parameters like this 
```
python main.py aim bank cuda:0 1.0 --num_preprocess privtree --rare_threshold 0.01
```


### Module Comparison 
In the experiment section of the paper, we compare different modules by comparing the performances of different reconstructed algorithms.
These reconstructed algorithms are allocated new names, which can be delivered to `method`. 
For example, if you want to try PrivSyn selector with generative network synthesizer, you can try
```
python main.py gem_syn bank cuda:0 1.0 --num_preprocess uniform_kbins --rare_threshold 0.002
```




## Results Collection
The code for evaluation is in file `evaluator/eval_seeds.py`. By default, we generate data 5 times and conduct evaluation each time we generate the data. The results are the average of all evaluations. All the results are collected in JSON format and saved in the fold `exp/{name of dataset}/{name of method}`, which can be used for further analysis. 


## Acknowledge 
Part of the code is from [AIM](https://github.com/ryan112358/private-pgm), [DP-MERF](https://github.com/ParkLabML/DP-MERF), [GEM](https://github.com/terranceliu/iterative-dp?tab=readme-ov-file), [Private-GSD](https://github.com/giusevtr/private_gsd), [PrivMRF](https://github.com/caicre/PrivMRF), [PrivSyn](https://github.com/agl-c/deid2_dpsyn), [RAP++](https://github.com/amazon-science/relaxed-adaptive-projection), [TabDDPM](https://github.com/yandex-research/tab-ddpm). We sincerely thank them for their contribution to the community.