#!/bin/bash
SBATCH --job-name=ippo_overcooked
SBATCH --output=logs/ippo_%j.out
SBATCH --error=logs/ippo_%j.err
SBATCH --time=24:00:00
SBATCH --partition=gpu
SBATCH --gres=gpu:1
SBATCH --cpus-per-task=8
SBATCH --mem=32G

# Exit immediately if a command fails
set -e

# Load required modules
module load cuda/11.8.0-fasrc01
module load cudnn/8.9.2.26_cuda11-fasrc01
module load python/3.10.13-fasrc01

# Activate virtual environment
source .venv/bin/activate

# Run the script using uv
uv run python baselines/IPPO/ippo_ff_overcooked.py
