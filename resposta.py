
'''
## 1. Como o efeito conjunto entre Promoção, Desconto e Competitor Pricing impacta o aumento percentual nas Units Sold, controlando por Categoria e Região?

Vamos analisar o impacto conjunto dessas variáveis usando um modelo de regressão linear, controlando por Categoria e Região.
'''

# =====================
# CÓDIGO: Pergunta 1
# =====================
df = df.sort_values(['Product ID', 'Date'])
df['Prev_Units_Sold'] = df.groupby('Product ID')['Units Sold'].shift(1)
df['Pct_Change_Units_Sold'] = (df['Units Sold'] - df['Prev_Units_Sold']) / df['Prev_Units_Sold'] * 100
model1 = ols('Pct_Change_Units_Sold ~ Promotion * Discount * Q("Competitor Pricing") + C(Category) + C(Region)', data=df).fit()
print(model1.summary())

# --- Campo para comentário sobre o resultado da Pergunta 1 ---
'''
# Comentário sobre o resultado da Pergunta 1
# (Descreva aqui os principais efeitos e interações encontradas no modelo)
'''

# =====================
# MARKDOWN: Pergunta 2
# =====================
'''
## 2. Existe interação significativa entre Seasonality, Weather Condition e Epidemic na variação do Demand? Caso sim, quais combinações de fatores geram maior ou menor demanda?

Vamos usar ANOVA para testar interações e visualizar as médias de demanda por combinação de fatores.
'''

# =====================
# CÓDIGO: Pergunta 2
# =====================
model2 = ols('Demand ~ C(Seasonality) * C(Weather Condition) * C(Epidemic)', data=df).fit()
anova2 = sm.stats.anova_lm(model2, typ=2)
display(anova2)
pivot2 = df.pivot_table(index=['Seasonality', 'Weather Condition', 'Epidemic'], values='Demand', aggfunc='mean')
display(pivot2.sort_values('Demand', ascending=False).head(10))

'''
# Comentário sobre o resultado da Pergunta 2
# (Descreva aqui as interações significativas e as combinações de fatores que mais impactam a demanda)
'''

# =====================
# MARKDOWN: Pergunta 3
# =====================
'''
## 3. Qual o efeito moderador da variável Region na relação entre Inventory Level e Units Ordered? Ou seja, em quais regiões um baixo estoque leva a um maior volume de pedidos?
'''

# =====================
# CÓDIGO: Pergunta 3
# =====================
model3 = ols('Units Ordered ~ Inventory Level * C(Region)', data=df).fit()
print(model3.summary())
sns.lmplot(x='Inventory Level', y='Units Ordered', hue='Region', data=df, aspect=2)
plt.title('Efeito do Estoque sobre Pedidos por Região')
plt.show()

'''
# Comentário sobre o resultado da Pergunta 3
# (Explique em quais regiões o baixo estoque leva a mais pedidos e como a relação varia)
'''

# =====================
# MARKDOWN: Pergunta 4
# =====================
'''
## 4. É possível prever com acurácia as Units Sold utilizando um modelo de machine learning que incorpore variáveis exógenas como Weather Condition, Competitor Pricing e Seasonality? Quais são as variáveis com maior importância no modelo?
'''

# =====================
# CÓDIGO: Pergunta 4
# =====================
df_ml = df.copy()
df_ml = pd.get_dummies(df_ml, columns=['Category', 'Region', 'Weather Condition', 'Seasonality'])
X = df_ml.drop(['Units Sold', 'Date', 'Store ID', 'Product ID'], axis=1)
y = df_ml['Units Sold']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred)))
print('R2:', r2_score(y_test, y_pred))
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
display(importances.head(10))

'''
# Comentário sobre o resultado da Pergunta 4
# (Comente sobre a acurácia do modelo e as variáveis mais importantes)
'''

# =====================
# MARKDOWN: Pergunta 5
# =====================
'''
## 5. Existe evidência de que, em determinadas categorias, o Price influencia mais fortemente o Demand do que a presença de uma Promotion? Em quais categorias essa diferença é estatisticamente significativa?
'''

# =====================
# CÓDIGO: Pergunta 5
# =====================
results = {}
for cat in df['Category'].unique():
    sub = df[df['Category'] == cat]
    model = ols('Demand ~ Price + Promotion', data=sub).fit()
    results[cat] = model.params
    print(f'Categoria: {cat}')
    print(model.summary())

'''
# Comentário sobre o resultado da Pergunta 5
# (Destaque as categorias onde o preço tem maior efeito que promoção e se é significativo)
'''

# =====================
# MARKDOWN: Pergunta 6
# =====================
'''
## 6. Como o comportamento de Units Sold em dias de Epidemic difere em termos de elasticidade-preço quando comparado a dias normais, considerando a variabilidade por Categoria?
'''

# =====================
# CÓDIGO: Pergunta 6
# =====================
for cat in df['Category'].unique():
    for epi in [0, 1]:
        sub = df[(df['Category'] == cat) & (df['Epidemic'] == epi)]
        if len(sub) > 10:
            model = ols('Units Sold ~ Price', data=sub).fit()
            print(f'Categoria: {cat} | Epidemic: {epi}')
            print(model.params)

'''
# Comentário sobre o resultado da Pergunta 6
# (Comente sobre as diferenças de elasticidade-preço entre dias normais e de epidemia por categoria)
'''

# =====================
# MARKDOWN: Pergunta 7
# =====================
'''
## 7. Quais padrões sazonais podem ser identificados na relação entre Units Ordered e Inventory Level, considerando simultaneamente o efeito de Competitor Pricing e a ocorrência de Promoções?
'''

# =====================
# CÓDIGO: Pergunta 7
# =====================
model7 = ols('Units Ordered ~ Inventory Level * Q("Competitor Pricing") * Promotion * C(Seasonality)', data=df).fit()
print(model7.summary())

'''
# Comentário sobre o resultado da Pergunta 7
# (Descreva padrões sazonais e interações relevantes)
'''

# =====================
# MARKDOWN: Pergunta 8
# =====================
'''
## 8. Em que medida as condições climáticas extremas (e.g., tempestades, ondas de calor) impactam o estoque (Inventory Level) e as vendas (Units Sold), e como esse impacto varia entre regiões e categorias?
'''

# =====================
# CÓDIGO: Pergunta 8
# =====================
extremos = ['Storm', 'Heatwave', 'Snowy', 'Heavy Rain']
df_ext = df[df['Weather Condition'].isin(extremos)]
sns.boxplot(x='Weather Condition', y='Units Sold', hue='Region', data=df_ext)
plt.title('Vendas por Condição Climática Extrema e Região')
plt.show()
sns.boxplot(x='Weather Condition', y='Inventory Level', hue='Category', data=df_ext)
plt.title('Estoque por Condição Climática Extrema e Categoria')
plt.show()

'''
# Comentário sobre o resultado da Pergunta 8
# (Comente sobre o impacto das condições extremas em estoque e vendas)
'''

# =====================
# MARKDOWN: Pergunta 9
# =====================
'''
## 9. Existe um limiar de Discount a partir do qual a probabilidade de aumento nas Units Sold se estabiliza ou diminui? Esse ponto de inflexão varia conforme Category ou Region?
'''

# =====================
# CÓDIGO: Pergunta 9
# =====================
for cat in df['Category'].unique():
    sns.lmplot(x='Discount', y='Units Sold', data=df[df['Category'] == cat], lowess=True)
    plt.title(f'Units Sold vs Discount - {cat}')
    plt.show()

'''
# Comentário sobre o resultado da Pergunta 9
# (Descreva o ponto de inflexão e se ele varia por categoria ou região)
'''

# =====================
# MARKDOWN: Pergunta 10
# =====================
'''
## 10. Como a combinação de baixa Competitor Pricing com altas Promoções e Descontos influencia o risco de ruptura de estoque (Inventory Level próximo de zero), e quais são os clusters de lojas mais suscetíveis a esse risco?
'''

# =====================
# CÓDIGO: Pergunta 10
# =====================
df['Ruptura'] = df['Inventory Level'] < 10
from sklearn.cluster import KMeans
features = df[['Competitor Pricing', 'Promotion', 'Discount', 'Inventory Level']]
features = features.fillna(0)
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(features)
risk = df.groupby('Cluster')['Ruptura'].mean()
print(risk.sort_values(ascending=False))

'''
# Comentário sobre o resultado da Pergunta 10
# (Comente sobre os clusters mais suscetíveis à ruptura de estoque)
'''

# --- FIM DO ARQUIVO resposta.py --- 