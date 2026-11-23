"""Cálculos de probabilidade e estudo de distribuição."""
import pandas as pd
import numpy as np
from scipy import stats


def analisar_probabilidade(dados: pd.DataFrame) -> dict:
    """Calcula probabilidades simples e verifica o ajuste aproximado à normal."""
    vendas = dados["unidades_vendidas"]
    mediana = vendas.median()
    media = vendas.mean()
    desvio = vendas.std()

    prob_acima_mediana = float((vendas > mediana).mean())
    limite_superior = media + desvio
    prob_um_desvio = float(((vendas >= media - desvio) & (vendas <= limite_superior)).mean())

    estatistica_ks, valor_p_normal = stats.kstest(vendas, "norm", args=(media, desvio))
    cdf_vendas = stats.norm.cdf(vendas, loc=media, scale=desvio)

    print("\n=== PROBABILIDADE E DISTRIBUIÇÃO ===")
    print(f"P(vendas > mediana): {prob_acima_mediana:.3f}")
    print(f"P(vendas entre uma distância de ±1 desvio da média): {prob_um_desvio:.3f}")
    print(f"Teste de normalidade KS — estatística={estatistica_ks:.4f}, p-valor={valor_p_normal:.4f}")
    print("Interpretação: p-valor baixo indica evidência contra a distribuição normal.")

    return {
        "probabilidade_acima_mediana": prob_acima_mediana,
        "probabilidade_area_central": prob_um_desvio,
        "media": float(media),
        "desvio": float(desvio),
        "cdf_vendas": cdf_vendas,
    }
