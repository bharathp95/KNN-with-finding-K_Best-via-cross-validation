# KNN Hyperparameter Tuning

Finds the best `k` for a K-Nearest Neighbors classifier on the Wine dataset using 10-fold cross-validation.

## What it does
1. Loads `wine_dataset.csv` (features + `target` column).
2. Scales features with `MinMaxScaler`.
3. Tests `k = 1` to `30`, computing mean 10-fold CV accuracy for each.
4. Plots accuracy vs. k.
5. Picks the `k` with highest CV accuracy (`best_k`).
6. Trains a final KNN model on a 70/30 train-test split using `best_k`.
7. Reports test accuracy, classification report, and confusion matrix.

## Requirements
```bash
pip install pandas scikit-learn matplotlib
```

## Usage
```bash
python knn_tuning.py
```

Make sure `wine_dataset.csv` is in the same directory, with a `target` column for labels.

## Output
- A plot of **k vs. cross-validated accuracy**
- Best `k` value
- Test accuracy, classification report, and confusion matrix for the tuned model

## Notes
- `cv=10` means each k is evaluated over 10 train/validation splits — this is what makes it "hyperparameter tuning" rather than a single lucky split.
- Final evaluation uses a **separate** train/test split so the reported accuracy isn't biased by the tuning process.
