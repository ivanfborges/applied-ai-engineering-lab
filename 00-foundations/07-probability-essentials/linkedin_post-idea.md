Um modelo pode identificar 90% das fraudes e, ainda assim, a maioria dos alertas gerados por ele ser falsa.

Parece contraditório, mas é um efeito direto da probabilidade base.

Quando o evento que queremos detectar é raro, não basta perguntar:

“Qual é a chance de o modelo gerar um alerta quando existe fraude?”

Também precisamos responder:

“Qual é a chance de realmente existir fraude quando o modelo gera um alerta?”

As duas perguntas parecem semelhantes, mas representam probabilidades diferentes.

Esse é um dos pontos mais importantes do Teorema de Bayes e aparece diretamente em problemas de fraude, diagnóstico, detecção de anomalias, classificação e avaliação de sistemas de IA.

A principal conclusão prática é que métricas não devem ser interpretadas isoladamente. Prevalência, custo dos erros, calibração e contexto de produção alteram completamente a decisão.

Documentei os conceitos, fórmulas, armadilhas de interpretação e uma simulação em Python no meu repositório de estudos em Applied AI Engineering.
