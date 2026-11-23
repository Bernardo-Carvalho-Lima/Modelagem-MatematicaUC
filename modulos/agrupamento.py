"""Agrupamento de registros comerciais com K-means."""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def executar_agrupamento(dados: pd.DataFrame) -> dict:
    """Padroniza os dados, cria três grupos e identifica seu comportamento."""
    variaveis = ["investimento_divulgacao", "preco_produto",
                 "desconto_percentual", "unidades_vendidas"]
    dados_padronizados = StandardScaler().fit_transform(dados[variaveis])

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    grupos = kmeans.fit_predict(dados_padronizados)

    analise = dados[variaveis].copy()
    analise["grupo"] = grupos
    perfil = analise.groupby("grupo")[variaveis].mean().round(2)

    print("\n=== AGRUPAMENTO K-MEANS ===")
    print(perfil.to_string())
    print("Os grupos representam perfis comerciais semelhantes, não classes previamente conhecidas.")

    return {"grupos": grupos, "perfil": perfil, "modelo": kmeans,
            "dados_padronizados": dados_padronizados}
