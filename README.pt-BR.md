# Applied AI Engineering Lab

[English](README.md) | **Português**

Laboratório público de estudos de Ciência de Dados e IA aplicada, com apoio de
IA na pesquisa, implementação e edição. Conecta teoria, código executável,
experimentos sintéticos, visualizações e testes.

**35 estudos implementados: 15 de fundamentos e 20 de ML clássico.** O roteiro
de 140 tópicos é um plano de aprendizado. As próximas áreas não são projetos
já entregues. [Inventário completo em inglês](README.md#implemented-studies)
e [roteiro](ROADMAP.md).

O [módulo de ML clássico](01-classical-machine-learning/) tem os 20 tópicos
16–35 implementados: ciclo de ML, validação, regressão, classificação, ensembles,
SVM, engenharia de atributos, desbalanceamento, dados ausentes e explicabilidade.
O [inventário em inglês](README.md#implemented-studies) separa as evidências por
estudo. O [tópico mais recente](01-classical-machine-learning/35-explainability-classical-ml/)
conecta permutação, PDP/ICE e reconstrução SHAP a seus limites.
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
python -m pip install -e ".[dev,imbalance,explainability,boosting]"
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

Os exemplos usam dados sintéticos ou definidos em código. Valores observados
não representam ganhos em produção. Código didático não substitui bibliotecas
validadas; roteiros futuros não indicam experiência de implantação nessas áreas.
Os créditos e referências de cada estudo estão preservados. [Licença MIT](LICENSE).
