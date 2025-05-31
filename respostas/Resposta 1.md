## Como o efeito conjunto entre Promoção, Desconto e Competitor Pricing impacta o aumento percentual nas Units Sold, controlando por Categoria e Região?

### Resumo da Resposta
Descontos maiores e preços de concorrentes mais altos aumentam as vendas. Promoções são especialmente eficazes para produtos mais caros. Os efeitos são aditivos, controlando por categoria e região.

### Metodologia
Utilizou-se regressão robusta (RLM) com engenharia de features, controle para categoria, região, preço logarítmico, intensidade de desconto e interações estratégicas (ex: produto premium).

### Principais Resultados
- Descontos e promoções aumentam significativamente as vendas.
- O efeito de promoção é maior para produtos de maior valor.
- Preços de concorrentes mais altos aumentam as vendas.
- Não há interação direta entre os três fatores principais.
- Controle para categoria, região e produto premium.

### Visualizações
- ![Gráfico de Resíduos](https://github.com/BitencourtVitor/Previsao-de-Estoque/blob/main/graficos/pergunta%201/Figure_3.png)

### Discussão
A estratégia ideal envolve combinar descontos, monitorar preços dos concorrentes e aplicar promoções, especialmente em produtos de maior valor. O modelo é robusto a outliers e controla para variáveis relevantes. Limitações incluem possíveis efeitos não modelados e dependência da qualidade dos dados.
