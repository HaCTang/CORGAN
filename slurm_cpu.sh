#!/bin/bash
#SBATCH --job-name=CORGAN_cpu_test   
#SBATCH --partition=smp             
#SBATCH --ntasks=1                  
#SBATCH --cpus-per-task=32           
#SBATCH --time=24:00:00             
#SBATCH --mem=16G                  
#SBATCH --output=cpu_test.log       
#SBATCH --error=cpu_test_error.log  

module load gcc/8.2.0

source /ihome/jwang/hat170/miniconda3/etc/profile.d/conda.sh
conda activate tf1.4

python /ihome/jwang/hat170/CORGAN/CORGAN/condi_example.py