# dpsyn_clean

## Steps

1. Check out dpsyn_clean repo. 

2. You can download the raw datasets from [here](https://drive.google.com/drive/folders/1MHRJxLhnJWZln8XBCon9UrN_EwVj14BE). Don't forget to put the dataset csv directly under folder temp_data/raw_data.

3. Install the required packages if you haven't done so, including  `numpy, pandas, scipy, networkx, scikit-learn, matplotlib`.

4. If you're using VSCode, you can change `.vscode/launch.json` to select which code to run under `program`. It also has the `args` defined.

5. You first need to preprocess data. Run `lib_preprocess/preprocess_network.py`, set `dataset_name` of `args` to the raw dataset file (without the `.csv` extension, like `ton`). The `num_synthesize_records` can be set to the similar number of the raw dataset. A preprocessed pickle file will be generated, under `temp_data/processed_data`, together with a mapping for binning. It will also generate a trivially decoded csv (binning and unbinning) in `temp_data/synthesized_records`. But please remember to put a json file to record the field of each column in path `temp_data/fields`.  

6. Then, you need to run `main.py` to generate synthesized data with privsyn, which will be under `temp_data/synthesized_records`.

7. For downstream tasks, you can run code from `lib_downstream` (e.g., `lib_downstream/flow_eval.py`) and specify the same `dataset_name`. It prints out the evaluation results on raw dataset and synthesized dataset.

## This version fixed the privacy overuse problem in Privsyn code
The orignial version of privsyn code overused privacy budget by calculating one way marginals multiple times(in the filter stage and calculating indif stage). According to paper(https://www.usenix.org/conference/usenixsecurity21/presentation/zhang-zhikun), the filter stage shouldn't calculate one way marginals, instead, it should take advantage of the one way marginals calculated before.
