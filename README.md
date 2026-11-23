# Projeto de Modelagem Matemática — UC9

Projeto escolar em Python que simula uma empresa fictícia e investiga quais fatores comerciais influenciam a quantidade de produtos vendidos.

## Objetivo

Demonstrar na prática estatística descritiva, probabilidade, correlação, inferência estatística, regressão, classificação e agrupamento usando dados fictícios e reproduzíveis.

## Estrutura

- `main.py`: integração e execução completa do projeto.
- `dados/vendas_exemplo.csv`: 120 registros fictícios com data, divulgação, preço, desconto e vendas.
- `modulos/estatistica.py`: média, mediana, moda, variância, desvio padrão e resumo.
- `modulos/probabilidade.py`: probabilidades simples, eventos e distribuição normal.
- `modulos/correlacao.py`: Pearson, matriz de correlação e interpretação.
- `modulos/inferencia.py`: amostragem, hipótese nula/alternativa, teste t e intervalo de confiança.
- `modulos/regressao.py`: regressão linear simples e múltipla, com MAE, RMSE e R².
- `modulos/classificacao.py`: regressão logística, árvore de decisão e KNN.
- `modulos/agrupamento.py`: segmentação por K-means.
- `modulos/visualizacao.py`: geração dos sete gráficos em `resultados/`.
- `RELACAO_COM_ATIVIDADE.txt`: correspondência detalhada com as etapas da UC9.

## Instalação no Windows

No terminal integrado do VS Code, entre na pasta do projeto e execute uma das opções abaixo:

```powershell
cd "C:\Users\bernardo63407536\Downloads\Modelagem_matematica\Projeto_Modelagem_Matematica_UC9"
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Se o PowerShell bloquear a ativação do ambiente, use diretamente o executável do ambiente virtual:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

## Execução

Execute sempre pelo ambiente virtual:

```powershell
.\.venv\Scripts\python.exe main.py
```

Para confirmar o ambiente correto:

```powershell
python -c "import sys; print(sys.executable)"
```

O caminho mostrado deve terminar em `Projeto_Modelagem_Matematica_UC9\.venv\Scripts\python.exe`.

## Bibliotecas

NumPy, Pandas, Matplotlib, Seaborn, SciPy, StatsModels e scikit-learn.

## Resultados esperados

Indicadores descritivos, probabilidades, coeficientes de correlação e inferência, métricas dos modelos, acurácias dos classificadores, perfis dos grupos e sete imagens PNG.
