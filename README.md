# DP Tabular Data Synthesis Benchmark
This is a benchmark for dp tabular data synthesis. The necessary code for the paper is all included in this repository.

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


### How to run 
We give some simple example. Firstly, make sure the datasets are put in the correct fold (in these examples, the fold is `data/bank`). In this example, the evaluation model is already tuned so users do not need any operation. Otherwise, you should tune the evaluation model (using the following code) before any further operation.
```
python evaluator/tune_eval_model.py bank mlp cv cuda:0
```

After you activate your enviroment, try the following code to make an overall evaluation.
```
python main.py aim bank cuda:0 1.0 --num_preprocess uniform_kbins --rare_threshold 0.002
```

If you want to try other preprocessing methods or hyper-parameter settings, you can try like this 
```
python main.py aim bank cuda:0 1.0 --num_preprocess privtree --rare_threshold 0.01
```

Finally, the deconstructed algorithms are allocated new names, which can be delivered to `method`. For example, if you want to try PrivSyn selector with generative network synthesizer, you can try like following
```
python main.py gem_syn bank cuda:0 1.0 --num_preprocess uniform_kbins --rare_threshold 0.002
```
