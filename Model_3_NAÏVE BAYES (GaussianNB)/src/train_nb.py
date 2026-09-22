Train/validation split

X_train_nb, X_valid_nb, y_train_nb, y_valid_nb = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

nb_pipeline.fit(X_train_nb, y_train_nb)
