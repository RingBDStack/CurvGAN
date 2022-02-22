# [WWW 2022] Curvature Graph Generative Adversarial Networks (CurvGAN)

This repository is the implementation of CurvGAN in PyTorch. 

## Environment 
```
python==3.7.1
pytorch==1.8.0
networkx==2.5.1
scikit-learn==0.24.2
pandas==1.2.4
GraphRicciCurvature==0.5.3
numpy==1.20.3
```
and their dependencies.

## Usage 
### 1. Setup
* Clone this repo
* Create a virtual environment using conda or virtualenv.
  ```
  conda env create -f environment.yml
  virtualenv -p [PATH to python3.6 binary] curvgan
  ```
* Enter the virtual environment and run `pip install -r requirements.txt`.

### 2. Usage
* Create a config file in `config/`
* Run `python curvgan.py [--config config_filename]` to train our model, with setting custom parameters.
  * An example, for link prediction (LP) task on Cora dataset: `python curvgan.py --config cora-lp-best.yaml`


## Thanks
Some of the code was forked from the following repositories: 
* [HazyResearch/HGCN](https://github.com/HazyResearch/hgcn);
* [emilemathieu/pvae](https://github.com/emilemathieu/pvae);  

We deeply thanks for their contributions to the open-source community.

