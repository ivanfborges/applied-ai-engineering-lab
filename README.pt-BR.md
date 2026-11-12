# Applied AI Engineering Lab

[English](README.md) | **Português**

Laboratório público de estudos de Ciência de Dados e IA aplicada, com apoio de
IA na pesquisa, implementação e edição. Conecta teoria, código executável,
experimentos sintéticos, visualizações e testes.

**50 estudos implementados: 15 de fundamentos, 20 de ML clássico, 9 de aprendizado não supervisionado/recomendação e 6 de experimentação.** O roteiro
de 140 tópicos é um plano de aprendizado. As próximas áreas não são projetos
já entregues. [Inventário completo em inglês](README.md#implemented-studies)
e [roteiro](ROADMAP.md).

O [módulo de ML clássico](01-classical-machine-learning/) tem os 20 tópicos
16–35 implementados: ciclo de ML, validação, regressão, classificação, ensembles,
SVM, engenharia de atributos, desbalanceamento, dados ausentes e explicabilidade.
O [inventário em inglês](README.md#implemented-studies) separa as evidências por
estudo. O [Dia 42 — Sistemas de Recomendação I](02-unsupervised-recommender-systems/42-recommender-systems-i/)
implementa vizinhos com notas explícitas, fallbacks e um laboratório visual offline.
O [Dia 43 — Sistemas de Recomendação II](02-unsupervised-recommender-systems/43-recommender-systems-ii/)
adiciona fatoração BPR e gradientes testados, comparação com SVD, conteúdo e
popularidade, seleção do peso híbrido na validação e avaliação de candidatos.
As interações são sintéticas, com cortes temporais globais. Um caso separado
verifica a entrada de item novo por metadados, sem medir sua relevância.
O [Dia 44 — Métricas de Ranking](02-unsupervised-recommender-systems/44-ranking-metrics/)
adiciona Precision@K, Recall@K, AP/MAP, RR/MRR e NDCG testados, múltiplos
relevantes por consulta, convenções explícitas e comparação com scikit-learn.
Os casos sintéticos separam efeitos da ordenação e do universo de candidatos.
O laboratório visual em Streamlit tem oito seções: rankings editáveis,
animações de AP/RR/NDCG, permutações exatas em 3D, simulação RAG com orçamento
de contexto e exportação de quatro GIFs. As interpretações aguardam revisão. O
[módulo de aprendizado não supervisionado e recomendação](02-unsupervised-recommender-systems/)
tem **9 de 9 tópicos implementados**; a revisão das interpretações continua pendente.
O [Dia 45 — A/B Testing do zero](03-statistics-experimentation/45-ab-testing/)
inicia o [módulo de experimentação, causalidade e produto](03-statistics-experimentation/),
agora com **6 de 8 tópicos implementados**. O Dia 45 inclui randomização sintética por usuário,
análise por intenção de tratar, teste Z com variância combinada sob a hipótese
nula, intervalo com variâncias separadas, SRM binomial exato, comparação com
SciPy e repetições A/A e de poder em horizonte fixo. Latência, custo e segurança
não foram medidos; as interpretações aguardam revisão.
O [Dia 46 — Poder estatístico e tamanho de amostra](03-statistics-experimentation/46-power-analysis/)
adiciona dimensionamento testado com SciPy/statsmodels e um laboratório
Streamlit com oito módulos: regiões de erro, curvas e superfície 3D de poder,
Monte Carlo com incerteza, correlação, maturação dos resultados e relevância
prática. Inclui três GIFs reproduzíveis; as interpretações aguardam revisão.
O [Dia 47 — Métricas de produto e guardrails](03-statistics-experimentation/47-product-metrics-guardrails/)
adiciona um scorecard sintético testado e laboratório Streamlit com nove seções:
incerteza de conversão, atividade coerente e sobrevivência de assinantes com
censura, caudas de latência, break-even, eventos raros de segurança, fronteira
de Pareto factível em 3D, política conservadora de rollout, armadilhas de
agregação e dois GIFs sob demanda. O estudo histórico com comparações pontuais
permanece separado; as interpretações aguardam revisão e não há autorização
de implantação.
O [Dia 48 — Testes múltiplos e falsos positivos](03-statistics-experimentation/48-multiple-testing/)
adiciona Bonferroni, Holm e Benjamini–Hochberg testados, comparação com
statsmodels/SciPy e simulações sintéticas de FWER, FDR e poder com erro
Monte Carlo. Análises interinas aninhadas comparam horizonte fixo, peeking
ingênuo e Bonferroni entre análises planejadas. As interpretações aguardam
revisão do autor.
O [Dia 49 — Inferência causal I: DAGs e confundidores](03-statistics-experimentation/49-causal-inference-i-dags-confounders/)
adiciona regras de d-separation e critério backdoor original testados,
equivalência FWL/OLS, padronização por estratos e comparação sintética repetida
entre ajustes por confundidor, mediador e collider. As interpretações aguardam
revisão; o grafo assumido não estabelece efeitos de um produto real.
O [tópico mais recente, Dia 50 — Inferência causal II: propensity score e matching](03-statistics-experimentation/50-causal-inference-ii-propensity-score-matching/)
adiciona matching por logit com reposição e caliper explícito, ponderação
normalizada para ATE/ATT, balanceamento e ESS por grupo testados. Um estudo
sintético repetido de menor sobreposição registra retenção, reutilização de
controles e erro de estimação. As interpretações aguardam revisão do autor.
O laboratório visual separado em Matplotlib tem nove capítulos, superfície
3D de propensão verdadeira/estimada, seis GIFs via Pillow, sliders nativos,
comparação repetida de ATE/ATT e confundimento oculto. Nesse gerador, efeitos
negativos significam minutos economizados; os registros do estudo anterior
permanecem separados e a revisão das interpretações continua pendente.
São quatro módulos iniciados: fundamentos (15/15), ML clássico (20/20),
aprendizado não supervisionado e recomendação (9/9) e experimentação (6/8).
Os Dias 51–52 e os módulos 4–11 continuam planejados. Os candidatos de
interpretação dos Dias 36–50 aguardam revisão do autor.
Implementação concluída não equivale a aprovação pessoal das interpretações.

## Por onde começar

| Pergunta | Estudo | Evidência e limite |
|---|---|---|
| Como conectar dados, validação e inferência? | [Pipeline de ML](01-classical-machine-learning/16-end-to-end-ml-pipeline/) | Churn sintético, pré-processamento dentro dos folds, limiar e persistência testada. Sem implantação real de retenção. |
| O resultado pode estar inflado por vazamento? | [Validação e leakage](01-classical-machine-learning/17-validation-and-leakage/) | Controles com rótulos aleatórios, grupos e disponibilidade temporal. Uma semente não define o tamanho geral do efeito. |
| O que uma regressão ajustada demonstra? | [Regressão linear](01-classical-machine-learning/18-linear-regression-theory/) | Geometria de OLS, resíduos e testes. Identidades no treino não provam generalização ou causalidade. |
| Como funciona o otimizador? | [Gradiente do zero](00-foundations/06-gradient-descent-from-scratch/) | Implementação NumPy e diagnóstico de convergência com dados sintéticos. Finalidade didática; sem suíte própria de testes. |
| Por que associação não basta para intervir? | [Correlação e causalidade](00-foundations/13-correlation-causation/) | Geradores conhecidos para confundimento e seleção. Sem identificação causal em dados empresariais. |
| Os atributos estavam disponíveis na hora da previsão? | [Engenharia de atributos](01-classical-machine-learning/32-feature-engineering/) | Cortes de evento/disponibilidade, transformações ajustadas no treino e testes. Alvo sintético favorável; sem simulação de deploy. |
| Balancear dados equivale a escolher uma política de ação? | [Dados desbalanceados](01-classical-machine-learning/33-imbalanced-data/) | Reamostragem só no treino e limiares escolhidos na validação. Custos ilustrativos; sem benefício empresarial validado. |
| O que uma explicação mede e o que ela não prova? | [Explicabilidade](01-classical-machine-learning/35-explainability-classical-ml/) | Permutação, PDP/ICE e reconstrução SHAP testados. Sem garantia causal ou de calibração. |

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
