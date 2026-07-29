Nem toda dimensão de um dataset carrega a mesma quantidade de informação.

Ao estudar PCA e SVD com mais profundidade, um dos pontos que mais chama atenção é que redução de dimensionalidade não significa simplesmente excluir colunas. O objetivo é encontrar novas direções que concentrem a estrutura mais relevante dos dados.

Na prática, isso pode ajudar em visualização, compressão, redução de ruído e até na diminuição do tamanho de embeddings utilizados em sistemas de busca vetorial.

Mas existe um cuidado importante: preservar variância não significa necessariamente preservar aquilo que é relevante para uma classificação, previsão ou recuperação de documentos.

Por isso, em um sistema real, a escolha da dimensionalidade precisa considerar não apenas a variância explicada, mas também métricas do problema, custo computacional, latência e estabilidade da transformação.

Documentei a parte teórica, os principais trade-offs e implementações com NumPy e scikit-learn no meu repositório de Applied AI Engineering.
