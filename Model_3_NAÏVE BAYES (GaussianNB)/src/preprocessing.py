numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

nb_clf = GaussianNB()

nb_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", nb_clf)
])
