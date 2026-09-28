#!/bin/bash
#SBATCH --job-name=chat
#SBATCH --output=job.%j.out        # %j 会自动替换为作业ID，避免重名覆盖
#SBATCH --error=job.%j.err
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --partition=gpujl
#SBATCH --gres=gpu:2

# 加载必要的环境（按需修改）
# module load cuda/11.8 python/3.10
# 进入作业提交时的目录（Slurm默认是提交目录，但有时需要显式指定）
cd $SLURM_SUBMIT_DIR

echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURM_JOB_NODELIST"
echo "Start time: $(date)"

# 在这里运行你的实际工作，例如：
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
source .venv/bin/activate

# train the tokenizer with vocab size 2**15 = 32768 on ~2B characters of data
python -m scripts.tok_train
# evaluate the tokenizer (report compression ratio etc.)
python -m scripts.tok_eval

# d24 model (slightly undertrained to beat GPT-2 => decrease data:params ratio from compute optimal 10.5 (default) to 8)
torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- --depth=24 --target-param-data-ratio=8 --device-batch-size=16 --fp8
# evaluate the model: CORE metric, BPB on train/val, and draw samples
torchrun --standalone --nproc_per_node=1 -m scripts.base_eval -- --device-batch-size=16

# -----------------------------------------------------------------------------
# SFT (teach the model conversation special tokens, tool use, multiple choice)

# run SFT and eval the model
torchrun --standalone --nproc_per_node=1 -m scripts.chat_sft
torchrun --standalone --nproc_per_node=1 -m scripts.chat_eval -- -i sft

echo "End time: $(date)"