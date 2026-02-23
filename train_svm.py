import numpy as np
from sklearn.svm import SVC
import os
import joblib
import argparse
from pathlib import Path

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

def main(args):
    clean_attention = _load_attention_group(args.clean_attention_dir)
    attacked_attention = _load_attention_group(args.attacked_attention_dir)

    if clean_attention.shape[0] != attacked_attention.shape[0]:
        raise ValueError(
            f"Question dimension mismatch: clean={clean_attention.shape[0]}, attacked={attacked_attention.shape[0]}"
        )

    attention_map = np.concatenate([clean_attention, attacked_attention], axis=1)
    svm_y = np.concatenate([
        np.zeros(clean_attention.shape[1], dtype=np.int64),
        np.ones(attacked_attention.shape[1], dtype=np.int64),
    ])

    question_num = np.size(attention_map, 0)
    sample_num = np.size(attention_map, 1)
    attention_map = attention_map.reshape(question_num, sample_num, -1)

    if args.question_index==-1:
        for i in range(question_num):
            print(f"training the {i}/{question_num} svm...")
            svm_classifier = SVC(kernel='linear', C=10)
            svm_classifier.fit(attention_map[i], svm_y)
            joblib.dump(svm_classifier, os.path.join(args.svm_dir, f"svm_question_{i}.pkl"))
    else:
        print(f"training the {args.question_index} svm...")
        svm_classifier = SVC(kernel='linear', C=10)
        svm_classifier.fit(attention_map[args.question_index], svm_y)
        joblib.dump(svm_classifier, os.path.join(args.svm_dir, f"svm_question_{args.question_index}.pkl"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean_attention_dir', required=True, type=str, help='Directory (recursive) of clean attention .npy files')
    parser.add_argument('--attacked_attention_dir', required=True, type=str, help='Directory (recursive) of attacked attention .npy files')
    parser.add_argument('--svm_dir', required=True, type=str, help='Output directory for trained svm models')
    parser.add_argument('--question_index', default=-1, type=int)
    args = parser.parse_args()

    Path(args.svm_dir).mkdir(parents=True, exist_ok=True)

    main(args)
