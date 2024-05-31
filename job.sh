#!/bin/bash
#SBATCH --cpus-per-task=2                                # Ask for 2 CPUs
#SBATCH --gres=gpu:1                                     # Ask for 1 GPU
#SBATCH --mem=10G                                        # Ask for 10 GB of RAM
#SBATCH --time=1:00:00                                   # The job will run for 1 hours


## Step 1,2,3,4 for non-interactive jobs running on an external compute node##

# 1. Load the required modules
# module load python/3.8

# 2. Load your environment
# source /pathtoyourvenv/bin/activate

# 3. Make visible
# export CUDA_VISIBLE_DEVICES=0,1

# 3. Copy your dataset on the compute node
# cp -r data $SLURM_TMPDIR/data

# 4. Launch your job, tell it to save the model in $SLURM_TMPDIR and look for the dataset into $SLURM_TMPDIR

#####################################################################################
#### Experiment example : BPC vs params : CPRNN ####
#####################################################################################

python train.py model.name=cprnn model.hidden_size=32 model.rank=4 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.01 train.patience=2 train.seq_len=50
python train.py model.name=cprnn model.hidden_size=75 model.rank=34 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.01 train.patience=2 train.seq_len=50
python train.py model.name=cprnn model.hidden_size=128 model.rank=192 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.01 train.patience=2 train.seq_len=50
python train.py model.name=cprnn model.hidden_size=400 model.rank=210 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.01 train.patience=2 train.seq_len=50
python train.py model.name=cprnn model.hidden_size=512 model.rank=1183 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.01 train.patience=2 train.seq_len=50
python train.py model.name=cprnn model.hidden_size=1024 model.rank=2550 train.batch_size=128 train.lr=0.001 train.factor=0.5 train.threshold=0.02 train.patience=2 train.seq_len=50

