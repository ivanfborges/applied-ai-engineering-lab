"""Synthetic classification: ranking, probability quality, and decision policy."""

from sklearn.calibration import calibration_curve
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from from_scratch import select_threshold, threshold_metrics


def report_scores(name, labels, scores):
    print(
        f"{name}: ROC-AUC={roc_auc_score(labels, scores):.4f}, "
        f"AP={average_precision_score(labels, scores):.4f}, "
        f"Brier={brier_score_loss(labels, scores):.4f}"
    )


def main():
    features, labels = make_classification(
        n_samples=5000,
        n_features=12,
        n_informative=5,
        n_redundant=2,
        weights=[0.95, 0.05],
        class_sep=1.3,
        flip_y=0.01,
        random_state=23,
    )
    x_dev, x_test, y_dev, y_test = train_test_split(
        features, labels, test_size=0.2, stratify=labels, random_state=23
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_dev, y_dev, test_size=0.25, stratify=y_dev, random_state=24
    )
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(x_train, y_train)
    val_scores = model.predict_proba(x_val)[:, 1]
    test_scores = model.predict_proba(x_test)[:, 1]
    threshold = select_threshold(y_val, val_scores, min_precision=0.60)

    print("Synthetic data; train/validation/test = 3000/1000/1000")
    print(f"Positive counts: {sum(y_train)}/{sum(y_val)}/{sum(y_test)}")
    report_scores("Test original", y_test, test_scores)
    report_scores("Test score^4", y_test, test_scores**4)
    for name, scores in (("original", test_scores), ("score^4", test_scores**4)):
        observed, predicted = calibration_curve(
            y_test, scores, n_bins=5, strategy="quantile"
        )
        print(f"{name} calibration bins (mean prediction, event rate):")
        print([(round(p, 3), round(o, 3)) for p, o in zip(predicted, observed)])
    print(f"Validation-selected threshold (precision >= 0.60): {threshold:.6f}")
    print(f"Validation decision: {threshold_metrics(y_val, val_scores, threshold)}")
    print(f"Test decision: {threshold_metrics(y_test, test_scores, threshold)}")


if __name__ == "__main__":
    main()

