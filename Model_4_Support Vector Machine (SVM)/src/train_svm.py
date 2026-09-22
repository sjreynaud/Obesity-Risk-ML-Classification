(SVM): Train/validation split

X_train_svm, X_valid_svm, y_train_svm, y_valid_svm = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Fit SVM pipeline
svm_pipeline.fit(X_train_svm, y_train_svm)
