set -e
r=/home/z009565592/catccos-vllm/work
d="$r/deepseek-v4-diagnosis-20261011"
torchrun --nproc-per-node=4 --master-port=27062 "$d/moe_ablation.py" --backend deepmegamoe --out "$d/micro-deepmegamoe" >"$d/micro-deepmegamoe/run.log" 2>&1
python3 "$d/profile_pipeline.py" >"$d/profile-pipeline.log" 2>&1
