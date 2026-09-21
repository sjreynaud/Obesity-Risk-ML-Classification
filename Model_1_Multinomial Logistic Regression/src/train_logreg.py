import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

from preprocessing import get_feature_types, build_logreg_pipeline

def main():

    # Load datasets
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    sample_submission = pd.read_csv("data/sample_submission.csv")

    # Separate target
    target_col = "NObeyesdad"
    X = train.drop(columns=[target_col])
    y = train[target_col]

    # Drop ID column
    if "id" in X.columns:
        X = X.drop(columns=["id"])
    test_ids = test["id"]
    X_test = test.drop(columns=["id"])

    # Identify feature types
    numeric_features, categorical_features = get_feature_types(X)

    # Build pipeline
    pipeline = build_logreg_pipeline(numeric_features, categorical_features)

    # Train/validation split
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Fit model
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred_valid = pipeline.predict(X_valid)
    print("Validation accuracy:", accuracy_score(y_valid, y_pred_valid))
    print("\nClassification report:\n", classification_report(y_valid, y_pred_valid))
    print("\nConfusion matrix:\n", confusion_matrix(y_valid, y_pred_valid))

    # Predict on test set
    test_preds = pipeline.predict(X_test)

    # Save model
    joblib.dump(pipeline, "models/logistic_regression_pipeline.pkl")

    # Save submission
    submission = pd.DataFrame({
        "id": test_ids,
        "NObeyesdad": test_preds
    })
    submission.to_csv("submissions/submission_logreg.csv", index=False)

    print("Model and submission saved.")

if __name__ == "__main__":
    main()
