# Applied AI Engineering Lab

[English](README.md) | **Português**

Laboratório público de estudos de Ciência de Dados e IA aplicada, com apoio de
IA na pesquisa, implementação e edição. Conecta teoria, código executável,
experimentos sintéticos, visualizações e testes.

**52 estudos implementados: 15 de fundamentos, 20 de ML clássico, 9 de aprendizado não supervisionado/recomendação e 8 de experimentação.** O roteiro
de 140 tópicos é um plano de aprendizado. As próximas áreas não são projetos
já entregues. [Inventário completo em inglês](README.md#implemented-studies)
e [roteiro](ROADMAP.md).

São quatro módulos com todos os tópicos implementados:
[Fundamentos](00-foundations/) (15/15),
[ML clássico](01-classical-machine-learning/) (20/20),
[Aprendizado não supervisionado e recomendação](02-unsupervised-recommender-systems/) (9/9)
e [Experimentação, causalidade e produto](03-statistics-experimentation/) (8/8).

Os estudos conectam implementação numérica, avaliação sem vazamento,
representações, ranking e decisões sob incerteza. Incluem experimentos
delimitados e código didático testado, não implantações em produção.
O tópico mais recente é o [Dia 52 — Uplift Modeling](03-statistics-experimentation/52-uplift-modeling/).
Os módulos 4–11 continuam planejados; o inventário em inglês detalha cada estudo.

Implementação, verificação automatizada e revisão de interpretações são estados
distintos. Itens explicitamente pendentes continuam pendentes; revisões técnicas
de escopo limitado estão identificadas no [registro de curadoria](docs/CURATION.pt-BR.md).

## Por onde começar

| Pergunta | Estudo | Evidência e limite |
|---|---|---|
| Como conectar dados, validação e inferência? | [Pipeline de ML](01-classical-machine-learning/16-end-to-end-ml-pipeline/) | Churn sintético, pré-processamento dentro dos folds, limiar e persistência testada. Sem implantação real de retenção. |
| O resultado pode estar inflado por vazamento? | [Validação e leakage](01-classical-machine-learning/17-validation-and-leakage/) | Controles com rótulos aleatórios, grupos e disponibilidade temporal. Uma semente não define o tamanho geral do efeito. |
| O que uma regressão ajustada demonstra? | [Regressão linear](01-classical-machine-learning/18-linear-regression-theory/) | Geometria de OLS, resíduos e testes. Identidades no treino não provam generalização ou causalidade. |
| Como funciona o otimizador? | [Gradiente do zero](00-foundations/06-gradient-descent-from-scratch/) | Implementação NumPy e diagnóstico de convergência com dados sintéticos. Finalidade didática; sem suíte própria de testes. |
| Por que associação não basta para intervir? | [Correlação e causalidade](00-foundations/13-correlation-causation/) | Geradores conhecidos para confundimento e seleção. Sem identificação causal em dados empresariais. |
| Preservar variância preserva informação preditiva? | [PCA na prática](02-unsupervised-recommender-systems/39-pca-in-practice/) | Comparação Wine dentro dos folds e contraexemplo com sinal de baixa variância. Variância retida não valida previsão. |
| O que uma métrica de ranking deixa de fora? | [Métricas de ranking](02-unsupervised-recommender-systems/44-ranking-metrics/) | Métricas por ID testadas, denominadores e universo de candidatos explícitos. NDCG condicionado aos candidatos não mede recall de recuperação. |
| Ajustar resolve sobreposição fraca? | [Propensity score e matching](03-statistics-experimentation/50-causal-inference-ii-propensity-score-matching/) | Matching, pesos, balanceamento, retenção e ESS em dados sintéticos. Correção algébrica não estabelece identificação causal. |

Em uma visita curta, leia a pergunta e os limites do estudo e examine seu código
e seus testes. Os materiais técnicos detalhados permanecem em inglês.
Interpretações do módulo de ML marcadas como pendentes de revisão do autor
continuam pendentes; implementação e testes não aprovam conclusões em seu nome.
Veja o [registro de curadoria](docs/CURATION.pt-BR.md).

Estudos aplicados com fontes externas estão em repositórios próprios:
[TopVistos](https://github.com/ivanfborges/ML_olympiad_for_students-topvistos_EUA)
e [Airbnb Rio](https://github.com/ivanfborges/eng_dados-analytics_engineering).

## Executar localmente

Python 3.11 ou superior. Na raiz do repositório:

```sh
python -m venv .venv
```

Ative com `.venv/Scripts/Activate.ps1` no PowerShell ou
`source .venv/bin/activate` no Linux/macOS. Instale as dependências declaradas:

```sh
python -m pip install -e ".[dev]"
python 00-foundations/01-ai-ml-genai-landscape/example.py
```

Para a suíte integral, inclua as bibliotecas opcionais utilizadas pelos estudos:

```sh
python -m pip install -e ".[dev,imbalance,explainability,boosting,manifold,experimentation]"
python scripts/validate_repo.py all
```

O último comando verifica sintaxe, links, testes isolados por arquivo e abertura
dos aplicativos selecionados via AppTest, sem servidor. Para notebooks, use o
extra `.[dev,notebooks]`. As faixas de dependências em `pyproject.toml` não são
um lock: versões numéricas podem variar entre instalações.
O `python -m pytest` padrão cobre Foundations, não todo o repositório.
Veja o [contrato de validação](docs/validation.md) para comandos, cobertura,
isolamento e limites. Notebooks e aparência no navegador exigem revisão separada.

Para explorar a interface do pipeline localmente:

```sh
streamlit run 01-classical-machine-learning/16-end-to-end-ml-pipeline/streamlit_app.py
```

## Autoria, evidências e limites

ChatGPT e Codex participam como ferramentas de estudo, programação e edição.
Testes verificam comportamentos específicos; não comprovam revisão pessoal de
todas as interpretações. O autor mantém a responsabilidade pelas decisões e
conclusões. [Metodologia](docs/methodology.pt-BR.md).

Os exemplos usam dados sintéticos ou definidos em código, além do
[Wine](02-unsupervised-recommender-systems/39-pca-in-practice/references.md) e dos
[dígitos manuscritos](02-unsupervised-recommender-systems/40-t-sne-umap/references.md)
públicos distribuídos com scikit-learn, com fontes documentadas. Valores observados
não representam ganhos em produção. Código didático não substitui bibliotecas
validadas; roteiros futuros não indicam experiência de implantação nessas áreas.
Os créditos e referências de cada estudo estão preservados. [Licença MIT](LICENSE).
