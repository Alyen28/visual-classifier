import os
import joblib
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier


MODEL_CONFIGS = {
    "mlp_4": {
        "hidden_layer_sizes": (4,),
        "architecture": "784 x 4 x 10",
        "path": "models/mlp_4.pkl"
    },
    "mlp_8": {
        "hidden_layer_sizes": (8,),
        "architecture": "784 x 8 x 10",
        "path": "models/mlp_8.pkl"
    },
    "mlp_32": {
        "hidden_layer_sizes": (32,),
        "architecture": "784 x 32 x 10",
        "path": "models/mlp_32.pkl"
    }
}


def carregar_dados():
    train_data = pd.read_csv("data/fashion-mnist_train.csv")
    test_data = pd.read_csv("data/fashion-mnist_test.csv")

    X_train = train_data.drop("label", axis=1) / 255.0
    y_train = train_data["label"]

    X_test = test_data.drop("label", axis=1) / 255.0
    y_test = test_data["label"]

    return X_train, y_train, X_test, y_test


def treinar_modelo(nome_modelo, config, X_train, y_train, X_test, y_test):
    print(f"\nTreinando {nome_modelo}...")
    print(f"Arquitetura: {config['architecture']}")

    model = MLPClassifier(
        hidden_layer_sizes=config["hidden_layer_sizes"],
        random_state=42,
        tol=0.005,
        max_iter=300
    )

    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_pred_train) * 100
    test_accuracy = accuracy_score(y_test, y_pred_test) * 100

    artifacts = {
        "model": model,
        "y_pred_test": y_pred_test,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "architecture": config["architecture"],
        "hidden_layer_sizes": config["hidden_layer_sizes"],
        "model_name": nome_modelo
    }

    joblib.dump(artifacts, config["path"])

    print(f"Modelo salvo em: {config['path']}")
    print(f"Acurácia no treino: {train_accuracy:.2f}%")
    print(f"Acurácia no teste: {test_accuracy:.2f}%")


def main():
    os.makedirs("models", exist_ok=True)

    X_train, y_train, X_test, y_test = carregar_dados()

    for nome_modelo, config in MODEL_CONFIGS.items():
        treinar_modelo(
            nome_modelo,
            config,
            X_train,
            y_train,
            X_test,
            y_test
        )


if __name__ == "__main__":
    main()