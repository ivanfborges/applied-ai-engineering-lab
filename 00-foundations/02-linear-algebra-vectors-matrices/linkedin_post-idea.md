Nos últimos estudos, revisitei um fundamento que aparece em praticamente todo sistema moderno de dados e IA: vetores e matrizes.

Na prática, uma observação de uma base, um documento convertido em embedding, os parâmetros de um modelo e até os tokens processados por um transformer acabam representados numericamente.

O ponto mais interessante é perceber que operações aparentemente básicas têm impacto direto em decisões de arquitetura:

- produto escalar mede alinhamento, mas também considera magnitude;
- similaridade de cosseno compara principalmente a direção dos vetores;
- distância euclidiana pode ser dominada pela escala das variáveis;
- multiplicação de matrizes permite aplicar transformações e processar grandes lotes de dados de forma eficiente.

Essas diferenças não são apenas matemáticas. Elas influenciam a qualidade de um sistema de busca semântica, o comportamento de um modelo, o consumo de memória e a latência em produção.

Documentei no GitHub a parte teórica, exemplos com NumPy, uma implementação simplificada do zero e algumas perguntas comuns de entrevistas técnicas.

#DataScience #MachineLearning #AIEngineering #LinearAlgebra #AppliedAI