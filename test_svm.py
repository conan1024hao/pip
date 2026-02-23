import numpy as np
from sklearn.metrics import roc_auc_score
import os
import joblib
import argparse

def _collect_npy_files(root_dir):
    npy_files = []
    for root, _, files in os.walk(root_dir):
        for name in files:
            if name.endswith(".npy"):
                npy_files.append(os.path.join(root, name))
    npy_files.sort()
    return npy_files

def _load_attention_group(attention_dir):
    file_list = _collect_npy_files(attention_dir)
    if len(file_list) == 0:
        raise ValueError(f"No .npy files found in: {attention_dir}")
    data = [np.load(p) for p in file_list]
    return np.stack(data, axis=1)

def _reshape_attention(attention_map):
    question_num = np.size(attention_map, 0)
    sample_num = np.size(attention_map, 1)
    return attention_map.reshape(question_num, sample_num, -1)

def _load_svm_models(svm_dir, svm_total_num):
    return [joblib.load(os.path.join(svm_dir, f"svm_question_{i}.pkl")) for i in range(svm_total_num)]

def _aggregate_vote_scores(attention_map_reshaped, svm_models):
    vote_scores = None
    for i, svm_classifier in enumerate(svm_models):
        # Prefer continuous margins for smoother ROC/FPR-threshold metrics.
        if hasattr(svm_classifier, "decision_function"):
            pred_i = np.asarray(svm_classifier.decision_function(attention_map_reshaped[i]), dtype=np.float64)
        else:
            pred_i = svm_classifier.predict(attention_map_reshaped[i]).astype(np.float64)
        if vote_scores is None:
            vote_scores = pred_i
        else:
            vote_scores += pred_i
    return vote_scores

def _quantile_threshold_for_fpr(clean_scores, target_fpr):
    # Choose the smallest threshold that keeps empirical FPR <= target_fpr.
    q = 1.0 - target_fpr
    try:
        return float(np.quantile(clean_scores, q, method="higher"))
    except TypeError:
        return float(np.quantile(clean_scores, q, interpolation="higher"))

def _threshold_for_target_fpr(clean_scores, target_fpr):
    """
    Find threshold t for rule (score >= t) whose achieved clean FPR is the
    smallest one that is still >= target_fpr.
    If target cannot be reached exactly due discrete scores, this avoids
    collapsing to the trivial all-negative prediction.
    """
    clean_scores = clean_scores.astype(np.float64)
    candidates = np.unique(clean_scores)[::-1]  # high -> low threshold
    fprs = np.array([np.mean(clean_scores >= t) for t in candidates], dtype=np.float64)

    valid = np.where(fprs >= target_fpr - 1e-12)[0]
    if len(valid) > 0:
        # Choose minimal achieved FPR among valid options.
        best_idx = valid[np.argmin(fprs[valid])]
        return float(candidates[best_idx])

    # Fallback: if all achieved FPR are below target (rare), use lowest threshold.
    return float(candidates[-1])

def _classification_stats(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
    fpr = fp / (fp + tn + 1e-8)
    return precision, recall, f1, fpr

def _metrics_from_scores(y_true, scores):
    # Higher score means more likely adversarial
    y_true = y_true.astype(np.int64)
    scores = scores.astype(np.float64)
    unique_thresholds = np.unique(scores)[::-1]

    max_f1 = 0.0
    for thr in unique_thresholds:
        y_pred = (scores >= thr).astype(np.int64)
        _, _, f1, _ = _classification_stats(y_true, y_pred)
        if f1 > max_f1:
            max_f1 = f1

    auc = roc_auc_score(y_true, scores)

    negative_scores = scores[y_true == 0]
    targets = [0.10, 0.05, 0.02, 0.01]
    out = {}
    for target in targets:
        thr = _threshold_for_target_fpr(negative_scores, target)
        y_pred = (scores >= thr).astype(np.int64)
        precision, recall, _, _ = _classification_stats(y_true, y_pred)
        out[target] = (precision, recall)

    return auc, max_f1, out

def main(args):
    clean_attention = _load_attention_group(args.clean_attention_dir)
    attacked_attention = _load_attention_group(args.attacked_attention_dir)

    if clean_attention.shape[0] != attacked_attention.shape[0]:
        raise ValueError(
            f"Question dimension mismatch: clean={clean_attention.shape[0]}, attacked={attacked_attention.shape[0]}"
        )

    attention_map = np.concatenate([clean_attention, attacked_attention], axis=1)
    y_true = np.concatenate([
        np.zeros(clean_attention.shape[1], dtype=np.int64),
        np.ones(attacked_attention.shape[1], dtype=np.int64),
    ])

    attention_map = _reshape_attention(attention_map)
    svm_models = _load_svm_models(args.svm_dir, args.svm_total_num)
    vote_scores = _aggregate_vote_scores(attention_map, svm_models)

    threshold = float(args.svm_alarm_num)
    if args.calib_clean_attention_dir:
        calib_clean_attention = _load_attention_group(args.calib_clean_attention_dir)
        if calib_clean_attention.shape[0] != attention_map.shape[0]:
            raise ValueError(
                f"Question dimension mismatch between calibration clean and test: calib={calib_clean_attention.shape[0]}, test={attention_map.shape[0]}"
            )
        calib_clean_attention = _reshape_attention(calib_clean_attention)
        calib_scores = _aggregate_vote_scores(calib_clean_attention, svm_models)
        threshold = _quantile_threshold_for_fpr(calib_scores, args.target_fpr)
    _ = threshold

    # Score-based metrics over the whole test set.
    auc, max_f1, fpr_metrics = _metrics_from_scores(y_true, vote_scores)
    precision_fpr_0_10, recall_fpr_0_10 = fpr_metrics[0.10]
    precision_fpr_0_05, recall_fpr_0_05 = fpr_metrics[0.05]
    precision_fpr_0_02, recall_fpr_0_02 = fpr_metrics[0.02]
    precision_fpr_0_01, recall_fpr_0_01 = fpr_metrics[0.01]
    print(f"AUC: {auc:.4f}")
    print(f"Max-F1: {max_f1:.4f}")
    print(f"Precision at FPR=0.10: {precision_fpr_0_10:.4f}, Recall at FPR=0.10: {recall_fpr_0_10:.4f}, F1 at FPR=0.10: {2 * precision_fpr_0_10 * recall_fpr_0_10 / (precision_fpr_0_10 + recall_fpr_0_10 + 1e-8):.4f}")
    print(f"Precision at FPR=0.05: {precision_fpr_0_05:.4f}, Recall at FPR=0.05: {recall_fpr_0_05:.4f}, F1 at FPR=0.05: {2 * precision_fpr_0_05 * recall_fpr_0_05 / (precision_fpr_0_05 + recall_fpr_0_05 + 1e-8):.4f}")
    print(f"Precision at FPR=0.02: {precision_fpr_0_02:.4f}, Recall at FPR=0.02: {recall_fpr_0_02:.4f}, F1 at FPR=0.02: {2 * precision_fpr_0_02 * recall_fpr_0_02 / (precision_fpr_0_02 + recall_fpr_0_02 + 1e-8):.4f}")
    print(f"Precision at FPR=0.01: {precision_fpr_0_01:.4f}, Recall at FPR=0.01: {recall_fpr_0_01:.4f}, F1 at FPR=0.01: {2 * precision_fpr_0_01 * recall_fpr_0_01 / (precision_fpr_0_01 + recall_fpr_0_01 + 1e-8):.4f}")

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean_attention_dir', required=True, type=str, help='Directory (recursive) of clean attention .npy files')
    parser.add_argument('--attacked_attention_dir', required=True, type=str, help='Directory (recursive) of attacked attention .npy files')
    parser.add_argument('--svm_dir', required=True)
    parser.add_argument('--svm_alarm_num', default=1, type=int)
    parser.add_argument('--svm_total_num', default=1, type=int)
    parser.add_argument('--calib_clean_attention_dir', default=None, type=str, help='Clean dev attention folder for threshold calibration')
    parser.add_argument('--target_fpr', default=0.02, type=float, help='Target FPR used with --calib_clean_attention_dir')
    args = parser.parse_args()
    if not (0.0 < args.target_fpr < 1.0):
        raise ValueError("--target_fpr must be between 0 and 1")
    main(args)
