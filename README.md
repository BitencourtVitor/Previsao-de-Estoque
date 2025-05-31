# Previsão de Estoque no Varejo

###### Estrutura do Projeto
- **modelos/**: scripts Python que utilizam Machine Learning para responder cada pergunta de negócio.
- **respostas/**: arquivos .md explicando as conclusões e interpretações de cada pergunta, seguindo um padrão de clareza e objetividade.
- **graficos/**: visualizações geradas pelos modelos, organizadas em subpastas por pergunta.
- **norteadores/**: dados, roteiros e insights iniciais para guiar a análise e o desenvolvimento dos modelos.

## Introdução
Este projeto foi desenvolvido a partir de uma base de dados pública do [Kaggle](https://www.kaggle.com/datasets/atomicd/retail-store-inventory-and-demand-forecasting/data), simulando o cenário de previsão de estoque em lojas de varejo. O objetivo é explorar, analisar e responder perguntas avançadas sobre o comportamento de vendas, demanda, promoções, sazonalidade e outros fatores que impactam a gestão de estoque.

## Sobre a Iniciativa
A proposta é ir além de análises descritivas, buscando responder questões complexas que envolvem relações entre variáveis, efeitos moderadores, interações e modelagem preditiva. As perguntas e modelos são lapidados com auxílio de Inteligência Artificial, especialmente Gemini 2.5 Pro e DeepThink R1 da DeepSeek, que têm sido essenciais para aprimorar a análise, sugerir abordagens e interpretar resultados.

## Passo a passo
- **OBJETIVO**: Analisar profundamente a base de dados, respondendo perguntas de negócio que envolvem estatística, visualização e machine learning.
- **EXTRAÇÃO**: A base foi obtida diretamente do Kaggle, sem necessidade de scraping ou coleta adicional.
- **TRANSFORMAÇÃO**: O tratamento dos dados e as análises são realizados em Python, utilizando bibliotecas como pandas, numpy, matplotlib, seaborn, scikit-learn e statsmodels.
- **ANÁLISE**: Cada pergunta é respondida em um script dedicado na pasta `modelos/`, com explicações detalhadas em markdown na pasta `respostas/` e visualizações em `graficos/`.

## Pontos de Atenção
A base é simulada, mas reflete situações realistas do varejo, incluindo promoções, descontos, sazonalidade, condições climáticas e até epidemias. As respostas e modelos aqui apresentados servem como referência para projetos reais de previsão de estoque e análise de demanda.

## Padrão para arquivos de resposta (.md)
Cada arquivo de resposta em `respostas/` segue o seguinte padrão:

- **Título da Pergunta**
- **Resumo da Resposta**: síntese objetiva do achado principal.
- **Metodologia**: breve descrição do(s) modelo(s) e abordagem utilizada.
- **Principais Resultados**: destaques quantitativos e qualitativos.
- **Visualizações**: referência aos gráficos gerados (com link ou caminho relativo para a subpasta em `graficos/`).
- **Discussão**: interpretação dos resultados, limitações e próximos passos.
