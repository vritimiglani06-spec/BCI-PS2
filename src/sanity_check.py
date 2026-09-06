# src/sanity_check.py
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from config import DATA_PROCESSED

def run_check(label_type="valence"):
    print(f"\n================ Running Within-Subject 5-Fold LDA ({label_type.upper()}) ================")
    accuracies = []
    
    for s in range(1, 24):
        X = np.load(DATA_PROCESSED / f"DREAMER_DE_S{s:02d}.npy")
        y = np.load(DATA_PROCESSED / f"DREAMER_labels_{label_type}_S{s:02d}.npy")

        classes, counts = np.unique(y, return_counts=True)
        if len(classes) < 2:
            print(f"Subject {s:02d}: Only one class present {classes}. Skipping.")
            continue

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_accs = []
        for tr_idx, te_idx in skf.split(X, y):
            clf = LinearDiscriminantAnalysis()
            clf.fit(X[tr_idx], y[tr_idx])
            fold_accs.append(clf.score(X[te_idx], y[te_idx]))

        subj_acc = np.mean(fold_accs) * 100
        accuracies.append(subj_acc)
        ratio_str = ", ".join([f"Class {c}: {cnt}" for c, cnt in zip(classes, counts)])
        print(f"Subject {s:02d} | N={X.shape[0]} windows | Accuracy: {subj_acc:.2f}% | ({ratio_str})")

    print(f"Mean {label_type.capitalize()} Within-Subject LDA Accuracy: {np.mean(accuracies):.2f}% +/- {np.std(accuracies):.2f}%\n")

if __name__ == "__main__":
    run_check("valence")
    run_check("arousal")