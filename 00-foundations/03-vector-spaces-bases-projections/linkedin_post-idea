Por que embeddings são chamados de representações vetoriais?

Quando um modelo transforma um texto em uma sequência de centenas de números, ele está posicionando esse texto em um espaço vetorial aprendido durante o treinamento.

O mais importante normalmente não é o significado de cada número isoladamente, mas a geometria criada pelo conjunto:

- conteúdos semelhantes ficam próximos;
- direções podem representar padrões compartilhados;
- projeções podem reduzir dimensionalidade;
- distâncias e produtos internos permitem buscar conteúdos relacionados.

Essa mesma base matemática aparece em regressão linear, PCA, sistemas de recomendação e busca semântica.

Um ponto importante para sistemas de RAG é que reduzir a dimensão dos embeddings pode diminuir memória e latência, mas preservar variância não significa necessariamente preservar relevância semântica.

Por isso, uma compressão de embeddings deve ser avaliada com métricas de recuperação e qualidade da resposta, e não apenas com erro de reconstrução.

Aprofundei esse tema com exemplos de projeção e experimentos em Python no meu laboratório de Applied AI Engineering.