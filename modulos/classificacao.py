"""Modelos de classificação para vendas acima ou abaixo da mediana."""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def executar_classificacoes(dados: pd.DataFrame) -> dict:
    """Treina regressão logística, árvore de decisão e KNN."""
    X = dados[["investimento_divulgacao", "preco_produto", "desconto_percentual"]]
    mediana = dados["unidades_vendidas"].median()
    y = (dados["unidades_vendidas"] > mediana).astype(int)

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    modelos = {
        "Regressão Logística": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)),
        "Árvore de Decisão": DecisionTreeClassifier(max_depth=4, random_state=42),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    }

    resultados = {}
    print("\n=== CLASSIFICAÇÃO ===")
    print(f"Regra: 1 = vendas acima da mediana ({mediana:.1f}); 0 = abaixo ou igual.")
    for nome, modelo in modelos.items():
        modelo.fit(X_treino, y_treino)
        previsoes = modelo.predict(X_teste)
        acuracia = accuracy_score(y_teste, previsoes)
        resultados[nome] = {"modelo": modelo, "acuracia": float(acuracia)}
        print(f"{nome}: acurácia = {acuracia:.3f}")

    return resultados
