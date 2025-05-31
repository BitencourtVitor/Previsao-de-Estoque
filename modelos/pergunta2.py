# Existe interação significativa entre Seasonality, Weather Condition e Epidemic na variação do Demand? Caso sim, quais combinações de fatores geram maior ou menor demanda?

# ==============================================================================
# MODELO 7 - ANÁLISE DE INTERAÇÕES: SAZONALIDADE, CLIMA E EPIDEMIAS
# ==============================================================================

# Importação de bibliotecas
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.formula.api as smf
from statsmodels.formula.api import rlm
from scipy import stats
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
import patsy

# Carregamento e pré-processamento inicial
df = pd.read_csv('sales_data.csv')

# Renomear colunas e converter tipos
df = df.rename(columns={
    'Competitor Pricing': 'Competitor_Pricing',
    'Weather Condition': 'Weather_Condition',
    'Inventory Level': 'Inventory_Level'
})
df['Date'] = pd.to_datetime(df['Date'])

# Engenharia de features
df = df.sort_values(['Product ID', 'Date'])
df['Prev_Units_Sold'] = df.groupby('Product ID')['Units Sold'].shift(1)
df['Pct_Change_Units_Sold'] = (df['Units Sold'] - df['Prev_Units_Sold']) / df['Prev_Units_Sold'] * 100
df['Month'] = df['Date'].dt.month

# Filtrar dados válidos
df = df[df['Prev_Units_Sold'] > 0].dropna(subset=['Pct_Change_Units_Sold'])

# Tratamento de outliers
Q1 = df['Pct_Change_Units_Sold'].quantile(0.01)
Q99 = df['Pct_Change_Units_Sold'].quantile(0.99)
df_clean = df[(df['Pct_Change_Units_Sold'] > Q1) & (df['Pct_Change_Units_Sold'] < Q99)].copy()

# Transformações de variáveis
df_clean['log_Price'] = np.log(df_clean['Price'] + 0.001)
df_clean['Premium_Product'] = np.where(df_clean['Price'] > df_clean['Price'].quantile(0.75), 1, 0)
df_clean['Discount_Intensity'] = df_clean['Discount'] * df_clean['Promotion']

# Identificar meses críticos (Setembro, Outubro, Novembro)
df_clean['Critical_Month'] = df_clean['Month'].apply(lambda x: 1 if x in [9, 10, 11] else 0)

# Transformação Box-Cox para normalizar variável dependente
# Ajuste: garantir valores positivos
pct_min = df_clean['Pct_Change_Units_Sold'].min()
offset = 1 - pct_min if pct_min < 1 else 0
df_clean['Pct_Change_Units_Sold_boxcox'], lambda_val = stats.boxcox(df_clean['Pct_Change_Units_Sold'] + offset)

# Preparação dos dados
df_modelo_simplificado = df_clean.copy()

# Criar variável de demanda (se necessário)
df_modelo_simplificado['Demand'] = df_modelo_simplificado['Units Sold']  # Usando vendas como proxy de demanda

# Transformação para normalidade
df_modelo_simplificado['Demand_boxcox'], lambda_demand = stats.boxcox(df_modelo_simplificado['Demand'] + 1)

# Modelo com interações triplas
formula = (
    'Demand_boxcox ~ Seasonality * Weather_Condition * Epidemic'
    ' + Category + Region + Promotion + np.log(Prev_Units_Sold)'
)
modelo_simplificado = smf.ols(
    formula=formula,
    data=df_modelo_simplificado
).fit()

# ==============================================================================
# ANÁLISE DE RESULTADOS
# ==============================================================================

print("="*80)
print("MODELO 7 - INTERAÇÕES ENTRE SAZONALIDADE, CLIMA E EPIDEMIAS")
print("="*80)
print(modelo_simplificado.summary())

# Análise de efeitos marginais
print("\n" + "="*80)
print("COMBINAÇÕES QUE MAXIMIZAM/MINIMIZAM A DEMANDA")
print("="*80)

# Calcular combinações significativas
interactions = []
for param in modelo_simplificado.params.index:
    if 'Seasonality' in param and 'Weather' in param and 'Epidemic' in param:
        coef = modelo_simplificado.params[param]
        pval = modelo_simplificado.pvalues[param]
        if pval < 0.05:
            interactions.append((param, coef))

# Ordenar por impacto
interactions_sorted = sorted(interactions, key=lambda x: x[1], reverse=True)

print("\nTOP 5 COMBINAÇÕES QUE AUMENTAM A DEMANDA:")
for param, coef in interactions_sorted[:5]:
    print(f"{param}: {coef:.2f} (Efeito: +{np.exp(coef)-1:.1%})")

print("\nTOP 5 COMBINAÇÕES QUE REDUZEM A DEMANDA:")
for param, coef in interactions_sorted[-5:]:
    print(f"{param}: {coef:.2f} (Efeito: {np.exp(coef)-1:.1%})")

# Visualização das interações aprimorada
plt.figure(figsize=(12, 8))
sns.heatmap(pd.crosstab(
    index=[df_modelo_simplificado['Seasonality'], df_modelo_simplificado['Epidemic']],
    columns=df_modelo_simplificado['Weather_Condition'],
    values=df_modelo_simplificado['Demand'],
    aggfunc='mean'
), annot=True, fmt=".0f", cmap="YlGnBu")
plt.title("Demanda Média por Combinação de Fatores")
plt.show()

# Gráfico de interação marginal (opcional)
plt.figure(figsize=(12, 6))
sns.pointplot(
    data=df_modelo_simplificado,
    x='Seasonality',
    y='Demand',
    hue='Epidemic',
    dodge=True,
    markers=['o', 's', 'D', '^'],
    capsize=.1
)
plt.title('Interação entre Sazonalidade e Epidemia na Demanda')
plt.show()