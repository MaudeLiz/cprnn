# CPRNN

This repository is the official implementation of [A Tensorized Perspective on Second-order RNNs](link to arxiv). 

## Installation
This project uses Python 3.8. 
It is recommended to run this code in a virtual environment. I used `venv` for this project.
To setup the virtual environment and download all the necessary packages, follow the steps below
First, load the Python module you want to use:
```
module load python/3.8
```
Or use `python3.8` instead of `python` in the following command that creates a virtual environment in your home directory:
```
python -m venv $HOME/<env>
```
Where `<env>` is the name of your environment. Finally, activate the environment:
```
source $HOME/<env>/bin/activate
```
Now to install the packages simply run
```
pip install -r requirements.txt
```

## Build data

To generate Penn Tree Bank dataset:

```cmd
python features/build_features.py -d ptb
```
 

## Train

Enter training configs in `configs.yaml` then run the command below. Results will be saved in `runs` folder.

```train
python train.py
```

## Visualize

To visualize the training process run:
```train
tensorboard --logdir=runs
```

## Reproducing results
To run all experiments in the figure below run:
```commandline
bash job.sh
```

Then visualize using:
```commandline
python visualization/bpc_vs_params.py
```

<!-- ![img.png](img.png) -->



