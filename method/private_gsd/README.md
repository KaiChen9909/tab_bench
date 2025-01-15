
# Description 
This GitHub repository houses an implementation of the Private-GSD mechanism
as outlined in the research paper, "Generating Private Synthetic Data with
Genetic Algorithms," presented at the 40th International Conference
on Machine Learning in 2023.

The Private-GSD mechanism is a specialized synthetic data generation tool,
designed to preserve different classes of statistical queries derived
from a given dataset while adhering to the principles of differential privacy.

# Example 


Visit this [Colab link](https://colab.research.google.com/drive/1t49XFG51pmcIsRqAhF_veHbrbfrVZBuy?usp=sharing) to start using Private-GSD.


# Setup

Set up conda environment
````
conda create -n gsd python=3.9

conda activate gsd 
pip install --upgrade pip
````

Install via setuptools
````
cd ~/
git clone https://github.com/giusevtr/private_gsd.git
cd ~/private_gsd 
pip install -e .
````

Install [JAX](https://github.com/google/jax#installation) separately. For example,
````
pip install --upgrade  "jax[cuda11_cudnn82]==0.4.6" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
````
Please make sure that the command you execute matches your system (i.e., tpu vs. gpu, right CUDA/cuDNN versions, etc.)

Download and preprocess datasets using [dp-data](https://github.com/terranceliu/dp-data).
````
cd ~/private_gsd 
git clone https://github.com/terranceliu/dp-data
cd ~/private_gsd/dp-data
pip install -e .
./preprocess_all.sh
cd ~/private_gsd
````
