"""Análise de correlação entre as variáveis comerciais."""
import pandas as pd
from scipy import stats


def investigar_relacao(dados: pd.DataFrame) -> dict:
    """Identifica variáveis dependentes e independentes e calcula Pearson."""
    independentes = ["investimento_divulgacao", "preco_produto", "desconto_percentual"]
    dependente = "unidades_vendidas"

    print("\n=== RELAÇÃO ENTRE VARIÁVEIS ===")
    print(f"Variável dependente: {dependente}")
    print(f"Variáveis independentes: {', '.join(independentes)}")

    coeficiente, valor_p = stats.pearsonr(
        dados["investimento_divulgacao"], dados["unidades_vendidas"]
    )
    correlacoes = dados[["investimento_divulgacao", "preco_produto",
                         "desconto_percentual", "unidades_vendidas"]].corr()

    print(f"Coeficiente de Pearson (divulgação × vendas): {coeficiente:.4f}")
    print(f"Valor-p: {valor_p:.4f}")
    print("A correlação observada não demonstra necessariamente causalidade.")
    print("\nMatriz de correlação:")
    print(correlacoes.round(3).to_string())

    return {"pearson": float(coeficiente), "valor_p": float(valor_p), "matriz": correlacoes}
