"""Arquivo principal do projeto de modelagem matemática da UC9."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

from modulos.estatistica import calcular_estatisticas, resumo_completo
from modulos.probabilidade import analisar_probabilidade
from modulos.correlacao import investigar_relacao
from modulos.inferencia import realizar_inferencia
from modulos.regressao import executar_regressoes
from modulos.classificacao import executar_classificacoes
from modulos.agrupamento import executar_agrupamento
from modulos.visualizacao import criar_graficos


CAMINHO_CSV = Path(__file__).resolve().parent / "dados" / "vendas_exemplo.csv"
PASTA_RESULTADOS = Path(__file__).resolve().parent / "resultados"


VERSAO_NUMPY_RECOMENDADA = (1, 26, 4)
versao_numpy_atual = tuple(int(parte) for parte in np.__version__.split(".")[:3])
if versao_numpy_atual[0] >= 2:
    print("=" * 68)
    print("ERRO DE AMBIENTE PYTHON")
    print(f"O NumPy {np.__version__} não corresponde às versões compatíveis deste projeto.")
    print("Use o ambiente virtual criado para o projeto:")
    print(r"  .\.venv\Scripts\python.exe main.py")
    print("Ou ative o ambiente e execute:")
    print(r"  .\.venv\Scripts\Activate.ps1")
    print("  python main.py")
    print("=" * 68)
    sys.exit(1)


def carregar_e_preparar_dados(caminho: Path) -> pd.DataFrame:
    """Carrega o CSV, valida as colunas e remove registros incompletos."""
    try:
        dados = pd.read_csv(caminho)
    except FileNotFoundError:
        print(f"Erro: arquivo não encontrado em {caminho}")
        sys.exit(1)
    except pd.errors.ParserError:
        print("Erro: não foi possível interpretar o arquivo CSV.")
        sys.exit(1)

    colunas_obrigatorias = {
        "data", "investimento_divulgacao", "preco_produto",
        "desconto_percentual", "unidades_vendidas"
    }
    faltantes = colunas_obrigatorias.difference(dados.columns)
    if faltantes:
        print("Erro: colunas ausentes no CSV:", ", ".join(sorted(faltantes)))
        sys.exit(1)

    dados = dados.copy()
    dados["data"] = pd.to_datetime(dados["data"], errors="coerce")
    dados = dados.dropna(subset=colunas_obrigatorias)

    colunas_numericas = ["investimento_divulgacao", "preco_produto",
                         "desconto_percentual", "unidades_vendidas"]
    dados[colunas_numericas] = dados[colunas_numericas].apply(pd.to_numeric, errors="coerce")
    dados = dados.dropna(subset=colunas_numericas)

    if len(dados) < 100:
        print("Aviso: foram encontrados menos de 100 registros válidos.")
    print(f"Dados preparados: {len(dados)} registros válidos.")
    return dados.reset_index(drop=True)


def apresentar_resumo_final(dados: pd.DataFrame, correlacao: dict,
                            regressao: dict, classificacao: dict,
                            agrupamento: dict) -> None:
    """Apresenta um resumo executivo dos principais achados."""
    print("\n=== RESUMO FINAL ===")
    print(f"Total de registros analisados: {len(dados)}")
    print(f"Correlação divulgação × vendas: {correlacao['pearson']:.3f}")

    simples = regressao["metricas_simples"]
    multiplo = regressao["metricas_multiplo"]
    print("Regressão simples — MAE: {:.2f}, RMSE: {:.2f}, R²: {:.3f}".format(
        simples["MAE"], simples["RMSE"], simples["R²"]))
    print("Regressão múltipla — MAE: {:.2f}, RMSE: {:.2f}, R²: {:.3f}".format(
        multiplo["MAE"], multiplo["RMSE"], multiplo["R²"]))

    for nome, resultado in classificacao.items():
        print(f"Classificação — {nome}: acurácia {resultado['acuracia']:.3f}")
    print("Tamanho dos grupos:")
    print(pd.Series(agrupamento["grupos"]).value_counts().sort_index().to_string())


def main() -> None:
    print("=" * 68)
    print("PROJETO DE MODELAGEM MATEMÁTICA ORIENTADA A DADOS — UC9")
    print("=" * 68)

    dados = carregar_e_preparar_dados(CAMINHO_CSV)
    calcular_estatisticas(dados)
    resumo_completo(dados)

    probabilidade = analisar_probabilidade(dados)
    correlacao = investigar_relacao(dados)
    inferencia = realizar_inferencia(dados)
    regressao = executar_regressoes(dados)
    classificacao = executar_classificacoes(dados)
    agrupamento = executar_agrupamento(dados)

    criar_graficos(dados, probabilidade, correlacao, regressao,
                   agrupamento, PASTA_RESULTADOS)
    apresentar_resumo_final(dados, correlacao, regressao,
                            classificacao, agrupamento)
    print("\nExecução concluída com sucesso.")


if __name__ == "__main__":
    main()
