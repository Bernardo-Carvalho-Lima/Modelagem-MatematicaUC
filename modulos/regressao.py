"""Modelos de regressão linear simples e múltipla."""
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


def _avaliar_regressao(X, y, semente: int = 42):
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.25, random_state=semente
    )
    modelo = LinearRegression()
    modelo.fit(X_treino, y_treino)
    previsoes = modelo.predict(X_teste)
    metricas = {
        "MAE": float(mean_absolute_error(y_teste, previsoes)),
        "RMSE": float(np.sqrt(mean_squared_error(y_teste, previsoes))),
        "R²": float(r2_score(y_teste, previsoes)),
        "previsoes": previsoes,
        "real": y_teste,
        "X_teste": X_teste,
    }
    return modelo, metricas


def executar_regressoes(dados: pd.DataFrame) -> dict:
    """Treina os modelos simples e múltiplo e compara o desempenho."""
    y = dados["unidades_vendidas"]
    X_simples = dados[["investimento_divulgacao"]]
    X_multiplo = dados[["investimento_divulgacao", "preco_produto", "desconto_percentual"]]

    modelo_simples, metricas_simples = _avaliar_regressao(X_simples, y)
    modelo_multiplo, metricas_multiplo = _avaliar_regressao(X_multiplo, y)

    print("\n=== REGRESSÃO LINEAR ===")
    print("Modelo simples:")
    print(f"  Intercepto: {modelo_simples.intercept_:,.2f}")
    print(f"  Coeficiente de divulgação: {modelo_simples.coef_[0]:,.4f}")
    print("  Métricas:", {k: round(v, 3) for k, v in metricas_simples.items() if isinstance(v, (float, int))})

    print("Modelo múltiplo:")
    print(f"  Intercepto: {modelo_multiplo.intercept_:,.2f}")
    for nome, coeficiente in zip(X_multiplo.columns, modelo_multiplo.coef_):
        print(f"  Coeficiente {nome}: {coeficiente:,.4f}")
    print("  Métricas:", {k: round(v, 3) for k, v in metricas_multiplo.items() if isinstance(v, (float, int))})

    exemplos = pd.DataFrame({
        "investimento_divulgacao": [10000, 20000, 30000],
        "preco_produto": [65, 60, 55],
        "desconto_percentual": [5, 10, 15],
    })
    previsoes_exemplo = modelo_multiplo.predict(exemplos)
    print("Previsões de exemplo:", np.round(previsoes_exemplo, 1).tolist())

    return {
        "modelo_simples": modelo_simples,
        "metricas_simples": metricas_simples,
        "modelo_multiplo": modelo_multiplo,
        "metricas_multiplo": metricas_multiplo,
    }
