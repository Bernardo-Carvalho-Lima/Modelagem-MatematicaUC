"""Funções de estatística descritiva."""
import pandas as pd
import numpy as np


def calcular_estatisticas(dados: pd.DataFrame) -> dict:
    """Calcula média, mediana, moda, variância e desvio padrão de vendas."""
    vendas = dados["unidades_vendidas"]
    moda = vendas.mode()
    resultado = {
        "média": float(vendas.mean()),
        "mediana": float(vendas.median()),
        "moda": moda.tolist(),
        "variância": float(vendas.var()),
        "desvio_padrão": float(vendas.std()),
        "mínimo": float(vendas.min()),
        "máximo": float(vendas.max()),
    }
    print("\n=== ESTATÍSTICA DESCRITIVA ===")
    for nome, valor in resultado.items():
        print(f"{nome.capitalize()}: {valor}")
    return resultado


def resumo_completo(dados: pd.DataFrame) -> pd.DataFrame:
    """Exibe um resumo estatístico organizado para todas as variáveis numéricas."""
    resumo = dados.select_dtypes(include=[np.number]).describe().T
    resumo["variância"] = dados.select_dtypes(include=[np.number]).var()
    print("\nResumo estatístico dos dados:")
    print(resumo.round(3).to_string())
    return resumo
