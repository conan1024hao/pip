source .venv/bin/activate

DATASETS=("NIPS17" "LLaVA-Instruct-150K" "Medical-Multimodal-Eval")
ATTACK_METHOD=("M-Attack" "FOA-Attack" "SSA-CWA")
LOG_DIR="/workspace/pip/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/test_svm_$(date +%Y%m%d_%H%M%S).txt"

for train_data in "${DATASETS[@]}"; do
    for test_data in "${DATASETS[@]}"; do
        for attack_method in "${ATTACK_METHOD[@]}"; do
            echo "Testing with train data: ${train_data}, test data: ${test_data}, attack method: ${attack_method}" | tee -a "${LOG_FILE}"
            python /workspace/pip/test_svm.py \
                --clean_attention_dir /workspace/pip/results/${test_data}/original/test \
                --attacked_attention_dir /workspace/pip/results/${test_data}/attacked/${attack_method}/test \
                --calib_clean_attention_dir /workspace/pip/results/${train_data}/original/dev \
                --svm_dir /workspace/pip/svm/${train_data}/${attack_method} \
                --svm_total_num 1 | tee -a "${LOG_FILE}"
            echo "----" | tee -a "${LOG_FILE}"
        done
    done
done

echo "Saved test output to ${LOG_FILE}"