Em Machine Learning, treinar um modelo significa responder repetidamente a uma pergunta: como cada parâmetro contribuiu para o erro atual?

As derivadas medem essa sensibilidade. O gradiente reúne essa informação para todos os parâmetros, enquanto a regra da cadeia permite propagar o erro por várias transformações — exatamente o que acontece no backpropagation de uma rede neural.

O ponto mais importante é que o gradiente não “encontra a solução”. Ele oferece uma direção local. A qualidade do treinamento ainda depende da taxa de aprendizado, da escala dos dados, da estabilidade numérica e, principalmente, de uma função de perda alinhada ao problema real.

Documentei no GitHub a derivação dos gradientes de uma regressão linear, uma implementação manual de backpropagation e experimentos de verificação numérica.
