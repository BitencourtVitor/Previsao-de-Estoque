# Como o efeito conjunto entre Promoção, Desconto e Competitor Pricing impacta o aumento percentual nas Units Sold, controlando por Categoria e Região?

# ==============================================================================
# MODELO 6 - REGRESSÃO ROBUSTA COM ENGENHARIA DE FEATURES
# ==============================================================================

# Importação de bibliotecas
import pandas as pd
import numpy as np
import statsmodels.api as sm
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

# Modelo 6 - Regressão Robusta (fórmula corrigida)
formula = (
    'Pct_Change_Units_Sold_boxcox ~ '
    'Promotion * log_Price + '
    'Discount_Intensity + '
    'Competitor_Pricing + '
    'C(Seasonality) + '
    'np.log(Prev_Units_Sold) + '
    'Inventory_Level + '
    'C(Weather_Condition) + '
    'C(Category):Premium_Product + '  # Interação estratégica
    'C(Region) + '
    'C(Critical_Month)'  # Usando a nova variável de mês crítico
)

model6 = rlm(
    formula,
    data=df_clean,
    M=sm.robust.norms.HuberT(t=1.345)  # Robustez a outliers
).fit()

# ==============================================================================
# ANÁLISE DE RESULTADOS
# ==============================================================================

# 1. Sumário do modelo
print("="*80)
print("MODELO 6 - RESULTADOS DA REGRESSÃO ROBUSTA")
print("="*80)
print(model6.summary())

# 2. Análise de elasticidade
price_coef = model6.params['log_Price']
promo_coef = model6.params['Promotion:log_Price']
mean_price = df_clean['Price'].mean()
mean_pct_boxcox = df_clean['Pct_Change_Units_Sold_boxcox'].mean()

# Elasticidade para produtos normais e premium
elasticity_normal = price_coef * (mean_price / mean_pct_boxcox)
elasticity_premium = (price_coef + promo_coef) * (mean_price / mean_pct_boxcox)

print("\n" + "="*80)
print("ELASTICIDADE-PREÇO DINÂMICA")
print("="*80)
print(f"Produtos Normais: {elasticity_normal:.4f}")
print(f"Produtos Premium: {elasticity_premium:.4f}")

# 3. Efeito das promoções por categoria premium
cate_effects = {}
for param in model6.params.index:
    if 'C(Category):Premium_Product' in param:
        parts = param.split('[')
        if len(parts) > 1:
            category_part = parts[1].split(']')[0]
            category = category_part.replace('T.', '')
            effect = model6.params[param]
            cate_effects[category] = effect

print("\n" + "="*80)
print("EFEITO PREMIUM POR CATEGORIA")
print("="*80)
if cate_effects:
    for category, effect in sorted(cate_effects.items(), key=lambda x: x[1], reverse=True):
        print(f"{category}: {effect:.2f}")
else:
    print("Nenhum efeito premium significativo encontrado por categoria")

# 4. Visualização de resíduos
plt.figure(figsize=(10, 6))
plt.scatter(model6.fittedvalues, model6.resid, alpha=0.3)
plt.axhline(y=0, color='r', linestyle='-')
plt.title('Análise de Resíduos')
plt.xlabel('Valores Ajustados')
plt.ylabel('Resíduos')
plt.grid(True)
plt.show()

# 5. Diagnóstico de multicolinearidade
print("\n" + "="*80)
print("DIAGNÓSTICO DE MULTICOLINEARIDADE (VIF)")
print("="*80)

# Preparar a matriz de design usando patsy
y, X = patsy.dmatrices(formula, df_clean, return_type='dataframe')

# Calcular VIF para cada variável explicativa
vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif_data.sort_values('VIF', ascending=False))

# 6. Exportar coeficientes para análise estratégica
coef_df = pd.DataFrame({
    'Variable': model6.params.index,
    'Coefficient': model6.params.values,
    'P-value': model6.pvalues
})
coef_df['Impact'] = coef_df['Coefficient'].abs()

print("\n" + "="*80)
print("PRINCIPAIS VARIÁVEIS PARA DECISÃO ESTRATÉGICA")
print("="*80)
significant_coefs = coef_df[coef_df['P-value'] < 0.05].sort_values('Impact', ascending=False)
print(significant_coefs.head(10))

# 7. Análise de resíduos
residuals = model6.resid
print("\n" + "="*80)
print("ANÁLISE DE RESÍDUOS")
print("="*80)
print(f"Assimetria (Skew): {residuals.skew():.4f}")
print(f"Curtose (Kurtosis): {residuals.kurtosis():.4f}")
print(f"Shapiro-Wilk p-value: {stats.shapiro(residuals)[1]:.4f}")

# 8. Efeito médio das promoções
print("\n" + "="*80)
print("EFEITO MÉDIO DAS PROMOÇÕES")
print("="*80)
mean_log_price = df_clean['log_Price'].mean()
promo_effect = model6.params['Promotion'] + model6.params['Promotion:log_Price'] * mean_log_price
print(f"Efeito médio de uma promoção: {promo_effect:.2f}%")