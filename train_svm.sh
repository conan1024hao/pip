source .venv/bin/activate

DATASETS=("NIPS17" "LLaVA-Instruct-150K" "Medical-Multimodal-Eval")
ATTACK_METHOD=("M-Attack" "FOA-Attack" "SSA-CWA")

for dataset in "${DATASETS[@]}"; do
  for attack_method in "${ATTACK_METHOD[@]}"; do
    python /workspace/pip/train_svm.py \
      --clean_attention_dir /workspace/pip/results/${dataset}/original/train \
      --attacked_attention_dir /workspace/pip/results/${dataset}/attacked/${attack_method}/dev \
      --svm_dir svm/${dataset}/${attack_method} \
      --question_index 0
  done
done
