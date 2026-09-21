import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

def get_feature_types(df):
    """Identify numeric and categorical columns."""
    numeric_features = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = df.select_dtypes(include=["object"]).columns.tolist()
    return numeric_features, categorical_features

def build_preprocessor(numeric_features, categorical_features):
    """Build ColumnTransformer for preprocessing."""
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )
    return preprocessor

def build_logreg_pipeline(numeric_features, categorical_features):
    """Build full pipeline: preprocessing + multinomial logistic regression."""
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    log_reg = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=1000,
        n_jobs=-1
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", log_reg)
        ]
    )
    return pipeline
