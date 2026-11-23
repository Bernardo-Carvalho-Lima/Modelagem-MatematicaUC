"""Criação e salvamento dos gráficos do projeto."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")


def criar_graficos(dados: pd.DataFrame, probabilidade: dict, correlacao: dict,
                   regressao: dict, agrupamento: dict, pasta_resultados: Path) -> list:
    """Gera todos os gráficos exigidos e salva em resultados/."""
    pasta_resultados.mkdir(parents=True, exist_ok=True)
    arquivos = []

    def salvar(fig, nome):
        caminho = pasta_resultados / nome
        fig.tight_layout()
        fig.savefig(caminho, dpi=150, bbox_inches="tight")
        plt.close(fig)
        arquivos.append(caminho)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(pd.to_datetime(dados["data"]), dados["unidades_vendidas"], marker="o", ms=2, color="#2878B5")
    ax.set(title="Histórico de vendas", xlabel="Data", ylabel="Unidades vendidas")
    fig.autofmt_xdate()
    salvar(fig, "01_historico_vendas.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=dados, x="investimento_divulgacao", y="unidades_vendidas", ax=ax)
    sns.regplot(data=dados, x="investimento_divulgacao", y="unidades_vendidas",
                scatter=False, color="#C44E52", ax=ax)
    ax.set(title="Divulgação × vendas", xlabel="Investimento em divulgação", ylabel="Unidades vendidas")
    salvar(fig, "02_dispersao_divulgacao_vendas.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(correlacao["matriz"], annot=True, cmap="coolwarm", center=0, fmt=".2f", ax=ax)
    ax.set_title("Matriz de correlação")
    salvar(fig, "03_matriz_correlacao.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(dados["unidades_vendidas"], kde=True, stat="density", ax=ax)
    eixo_x = np.linspace(dados["unidades_vendidas"].min(), dados["unidades_vendidas"].max(), 300)
    ax.plot(eixo_x, __import__("scipy").stats.norm.pdf(eixo_x, probabilidade["media"], probabilidade["desvio"]),
            color="#C44E52", label="Curva normal")
    ax.legend()
    ax.set(title="Distribuição das vendas", xlabel="Unidades vendidas", ylabel="Densidade")
    salvar(fig, "04_distribuicao_vendas.png")

    metricas = regressao["metricas_multiplo"]
    fig, ax = plt.subplots(figsize=(8, 5))
    real = np.asarray(metricas["real"])
    previsto = np.asarray(metricas["previsoes"])
    ax.scatter(real, previsto, alpha=0.75)
    limites = [min(real.min(), previsto.min()), max(real.max(), previsto.max())]
    ax.plot(limites, limites, "r--", label="Previsão perfeita")
    ax.set(title="Valores reais × previstos", xlabel="Vendas reais", ylabel="Vendas previstas")
    ax.legend()
    salvar(fig, "05_reais_previstos.png")

    residuos = real - previsto
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(previsto, residuos, alpha=0.75)
    ax.axhline(0, color="red", linestyle="--")
    ax.set(title="Gráfico de resíduos", xlabel="Vendas previstas", ylabel="Resíduo")
    salvar(fig, "06_residuos_regressao.png")

    dados_grupos = dados.copy()
    dados_grupos["grupo"] = agrupamento["grupos"]
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(data=dados_grupos, x="investimento_divulgacao", y="unidades_vendidas",
                    hue="grupo", palette="viridis", ax=ax)
    ax.set(title="Agrupamento K-means", xlabel="Investimento em divulgação", ylabel="Unidades vendidas")
    salvar(fig, "07_agrupamentos_kmeans.png")

    print("\nGráficos salvos em:", pasta_resultados.resolve())
    for arquivo in arquivos:
        print(" -", arquivo.name)
    return arquivos
