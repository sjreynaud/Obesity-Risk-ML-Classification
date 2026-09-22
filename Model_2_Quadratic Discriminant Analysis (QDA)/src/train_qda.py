#Train/validation split

X_train_qda, X_valid_qda, y_train_qda, y_valid_qda = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

qda_pipeline.fit(X_train_qda, y_train_qda)
