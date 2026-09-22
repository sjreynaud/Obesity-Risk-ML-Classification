
# Preprocessing + QDA pipeline

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

qda_clf = QuadraticDiscriminantAnalysis()

qda_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", qda_clf)
])
