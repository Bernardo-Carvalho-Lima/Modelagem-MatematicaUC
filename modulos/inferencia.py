"""Amostragem, testes de hipótese e intervalos de confiança."""
import pandas as pd
import numpy as np
from scipy import stats


def realizar_inferencia(dados: pd.DataFrame, semente: int = 42) -> dict:
    """Demonstra uma amostra, teste t e intervalo de confiança de 95%."""
    vendas = dados["unidades_vendidas"]
    rng = np.random.default_rng(semente)
    indices = rng.choice(len(dados), size=40, replace=False)
    amostra = vendas.iloc[indices]

    hipotese_nula = 130.0
    estatistica_t, valor_p = stats.ttest_1samp(amostra, hipotese_nula)
    intervalo = stats.t.interval(
        0.95, len(amostra) - 1, loc=amostra.mean(), scale=stats.sem(amostra)
    )

    print("\n=== INFERÊNCIA ESTATÍSTICA ===")
    print("Hipótese nula (H0): a média das vendas da amostra é igual a 130 unidades.")
    print("Hipótese alternativa (H1): a média das vendas da amostra é diferente de 130 unidades.")
    print(f"Estatística t: {estatistica_t:.4f}")
    print(f"Valor-p: {valor_p:.4f}")
    print(f"Intervalo de confiança de 95% da média: [{intervalo[0]:.2f}; {intervalo[1]:.2f}]")
    if valor_p < 0.05:
        conclusao = "Há evidência estatística para rejeitar H0 ao nível de 5%."
    else:
        conclusao = "Não há evidência estatística suficiente para rejeitar H0 ao nível de 5%."
    print(conclusao)

    return {
        "amostra": amostra,
        "estatistica_t": float(estatistica_t),
        "valor_p": float(valor_p),
        "intervalo_confianca": intervalo,
    }
