
import json
import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from scipy.stats import randint


DATA_PROCESSED = Path("data/processed")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_SIZE = 50_000
PARQUET_PATH = DATA_PROCESSED / f"enem_2023_amostra_{SAMPLE_SIZE}.parquet"

TARGET = "NU_NOTA_MT"


CATEGORICAL_FEATURES_DESEJADAS = ["Q006", "Q002", "TP_ESCOLA", "TP_COR_RACA"]
NUMERIC_FEATURES_DESEJADAS = ["NU_NOTA_CN", "NU_NOTA_LC", "NU_NOTA_CH", "NU_NOTA_RED"]
    #segunda etapa
def build_pipeline(cat_features_disponiveis, num_features_disponiveis):
    
    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", cat_transformer, cat_features_disponiveis),
            ("num", num_transformer, num_features_disponiveis)
        ],
        remainder="drop"
    )
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", KNeighborsRegressor())
    ])

def main():
    print(f"Carregando dados da amostra de: {PARQUET_PATH}")
    df = pd.read_parquet(PARQUET_PATH)
    
    df.dropna(subset=[TARGET], inplace=True)
    
    print(f"Iniciando treinamento para prever a nota de '{TARGET}'...")
    
    
    cat_features_disponiveis = [col for col in CATEGORICAL_FEATURES_DESEJADAS if col in df.columns]
    num_features_disponiveis = [col for col in NUMERIC_FEATURES_DESEJADAS if col in df.columns]
    
    
    features_faltantes = set(CATEGORICAL_FEATURES_DESEJADAS + NUMERIC_FEATURES_DESEJADAS) - set(df.columns)
    if features_faltantes:
        print(f"\nAVISO: As seguintes colunas não foram encontradas e serão ignoradas: {list(features_faltantes)}\n")

    features_disponiveis = cat_features_disponiveis + num_features_disponiveis
    X = df[features_disponiveis]
    y = df[TARGET]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    
    pipeline = build_pipeline(cat_features_disponiveis, num_features_disponiveis)
    
    param_dist = {
        'regressor__n_neighbors': randint(3, 15),
        'regressor__weights': ['uniform', 'distance'],
        'regressor__p': [1, 2]
    }
    
    search = RandomizedSearchCV(
        pipeline, param_distributions=param_dist, n_iter=10, cv=3,
        scoring='neg_root_mean_squared_error', n_jobs=-1, random_state=42, verbose=1
    )
    
    print("Iniciando a busca de hiperparâmetros...")
    search.fit(X_train, y_train)
    
    best_model = search.best_estimator_
    
    y_pred = best_model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    
    print("\n--- Resultados do Treinamento ---")
    print(f"Melhores parâmetros: {search.best_params_}")
    print(f"RMSE (Validação Cruzada): {-search.best_score_:.2f}")
    print(f"RMSE (Teste): {rmse:.2f}")
    
    model_path = MODEL_DIR / f"knn_{TARGET}.joblib"
    meta_path = MODEL_DIR / f"knn_{TARGET}_meta.json"
    
    joblib.dump(best_model, model_path)
    
    meta = {
        "target": TARGET,
        "features": {
            "categorical": cat_features_disponiveis,
            "numeric": num_features_disponiveis
        },
        "best_params": search.best_params_,
        "validation_rmse": -search.best_score_,
        "test_rmse": rmse
    }
    
    with open(meta_path, "w", encoding="utf-8") as f:
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer, np.int64)): return int(obj)
                if isinstance(obj, (np.floating, np.float64)): return float(obj)
                if isinstance(obj, np.ndarray): return obj.tolist()
                return super(NpEncoder, self).default(obj)
        json.dump(meta, f, ensure_ascii=False, indent=2, cls=NpEncoder)
        
    print(f"\nModelo salvo em: {model_path}")
    print(f"Metadados salvos em: {meta_path}")

if __name__ == "__main__":
    main()