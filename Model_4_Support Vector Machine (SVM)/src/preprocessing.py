Preprocessing + SVM pipeline

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

svm_clf = SVC(
    kernel="rbf",          # nonlinear kernel
    C=1.0,                 # regularization strength
    gamma="scale",         # kernel coefficient
    probability=False,     # class labels only
    random_state=42
)

svm_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", svm_clf)
])
