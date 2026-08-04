Nos últimos dias, revisitei um dos mecanismos mais fundamentais de Machine Learning: o Gradient Descent.

A ideia parece simples: calcular em qual direção o erro aumenta e atualizar os parâmetros na direção contrária.

Mas, quando implementamos o algoritmo do zero, alguns pontos ficam muito mais claros:

* por que uma taxa de aprendizado alta pode fazer o treinamento divergir;
* por que features em escalas muito diferentes prejudicam a convergência;
* como acompanhar não apenas a loss, mas também os gradientes e os parâmetros;
* e por que minimizar o erro de treino não significa necessariamente construir um bom modelo.

Implementei uma regressão linear usando apenas NumPy e comparei o resultado com o scikit-learn. Também visualizei a evolução da loss e dos coeficientes durante o treinamento.

É um exercício simples, mas ajuda a consolidar conceitos que aparecem em praticamente todo treinamento moderno, de regressões a redes neurais e Transformers.

A implementação e as anotações técnicas estão documentadas no meu repositório de estudos no GitHub.