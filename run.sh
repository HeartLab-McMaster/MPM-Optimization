#!/bin/bash 
#SBATCH --account=def-onaizah
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --mem=24G
#SBATCH --gres=gpu:a100:4
#SBATCH --time=0-05:00
#SBATCH --error=e_%j.txt

numpr=$1
seed=$2
population=$3
generation=$4
tagname=$5

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

module load python/3.10
module load cuda
module load libspatialindex

pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt
pip install taichi-1.5.0+computecanada-cp310-cp310-linux_x86_64.whl

python cma_opti_yz_hight_restricted.py -s $seed --num_proc $numpr --population $population --generation $generation --tag_name $tagname
