# Registro de curadoria e evidências

[English](CURATION.md) · [Entrada do laboratório](../README.pt-BR.md)

Curadoria de 18/09/2026, a partir da revisão `b369445`. Escopo: selecionar entradas,
corrigir navegação e caminhos de execução e explicitar o estado da revisão.
Não substitui os registros históricos, nem certifica todas as afirmações científicas
ou altera autoria. Nenhum experimento histórico foi reexecutado para a edição.

Atualização de navegação em 08/10/2026: o [índice de ML](../01-classical-machine-learning/)
cobre os tópicos 16–35 implementados, e a entrada da raiz inclui perguntas dos
dias 32, 33 e 35. A seleção e os testes datados abaixo continuam históricos,
não representam o escopo ou os resultados da validação atual. Veja o
[contrato de validação](validation.md).

Uma [validação de referência estável em 08/10/2026](validation.md#stable-reference-check--october-8-2026)
cobre o estado atual da revisão. Ela não substitui o registro de setembro
nem aprova interpretações pendentes.

## Revisão técnica delegada — 08/10/2026

O autor delegou explicitamente a curadoria técnica ao assistente de IA. O
assistente revisou nove itens de interpretação confrontando código local,
definições das métricas e registros públicos existentes, e adotou redações
delimitadas para os escopos abaixo. **A revisão foi feita pelo assistente sob
delegação, não é uma análise pessoal realizada pelo autor.** A responsabilidade
editorial continua com o autor, sem alterar quem realizou esta revisão.

| Estudo / itens | Escopo revisado | Decisão e limite preservado |
|---|---|---|
| 32 · 32-A/B | [Comparação de representações e demonstração aritmética](../01-classical-machine-learning/32-feature-engineering/notes.md#executed-experiments) | Restringir ranking ao gerador favorável e split registrado; distinguir nova história de mudanças na representação e regularização. Manter reutilização de parâmetros de treino como controle aritmético, não ganho preditivo. |
| 33 · 33-A/B/C | [Estratégias de treino e políticas de limiar registradas](../01-classical-machine-learning/33-imbalanced-data/notes.md#executed-experiment) | Manter o trade-off de limiar com scores fixos. Restringir balanceamento a AP/ROC-AUC/recall medidos, preservando confundimento por tamanhos/pesos e ausência de vencedor escolhido no teste. Brier maior não identifica má calibração nem seu mecanismo. |
| 35 · 35-A/B/C/D | [Registro da floresta](../01-classical-machine-learning/35-explainability-classical-ml/notes.md#executed-forest-experiment) e [controle linear](../01-classical-machine-learning/35-explainability-classical-ml/notes.md#executed-linear-control) | Restringir reliance conjunta ao protocolo de perturbação; distinguir quantidades de importância e gap de teste formal de suporte. Manter reconstrução/acordo e controle conhecido como verificações numéricas, não validação causal, de estabilidade ou deploy. |

Nenhum experimento foi reexecutado nesta revisão editorial. Tabelas numéricas,
configurações e ambientes permanecem históricos. Os visuais separados de
32/33/35 e candidatos de outros tópicos ficaram fora do lote e continuam pendentes.
Relatórios de futuras execuções começam pendentes: revisar uma interpretação
pública datada não aprova novas medições, configurações ou explicações.

## Seleção

As cinco entradas cobrem ciclo de ML (16), fronteiras de avaliação (17), premissas
do modelo (18), implementação numérica (6) e raciocínio causal (13). Todas têm código executável e explicações. Os tópicos 13 e 16–18 têm
testes públicos; o tópico 6 não possui arquivo próprio de testes automatizados
nesta revisão e foi selecionado para inspeção do código, sem igualar sua cobertura. Os demais estudos continuam no inventário;
nenhum foi removido por ser didático. O currículo mantém sua finalidade de formação.

## Revisão pessoal ainda pendente

| Estudo | Registro das interpretações propostas | Decisão do autor que falta |
|---|---|---|
| 16 · Pipeline | [README](../01-classical-machine-learning/16-end-to-end-ml-pipeline/README.md) e [notas](../01-classical-machine-learning/16-end-to-end-ml-pipeline/notes.md) | Adotar, restringir ou reescrever as explicações das simulações de churn, leakage, busca e drift, preservando limites sintéticos e da configuração. |
| 17 · Validação | [Notas](../01-classical-machine-learning/17-validation-and-leakage/notes.md) e [guia visual](../01-classical-machine-learning/17-validation-and-leakage/VISUAL_GUIDE.md) | Revisar otimismo da seleção de atributos, reconhecimento de entidades repetidas e desenho de validação; decidir novas prévias públicas separadamente. |
| 18 · Regressão | [Notas](../01-classical-machine-learning/18-linear-regression-theory/notes.md) e [guia visual](../01-classical-machine-learning/18-linear-regression-theory/VISUAL_GUIDE.md) | Revisar recuperação de coeficientes, ortogonalidade dos resíduos sob má especificação, compromissos de Ridge e imagens propostas. |

Essas pendências já estavam identificadas nos estudos. Testes aprovados e
permissão para editar o portfólio não equivalem a aprovação pessoal das conclusões.
Para promover uma interpretação, registrar a decisão do autor, escopo e ajustes.
Os tópicos não listados não recebem certificação de revisão científica integral.

## Correções e alcance da verificação

- O diretório do pipeline passou a `16-end-to-end-ml-pipeline`, mas comandos,
  cadastro do smoke test, regras de prévias no Git e links do perfil mantinham
  o nome antigo. As referências agora apontam ao diretório implementado.
- Seis links de imagens apontavam a arquivos ausentes. O README explica como
  gerar prévias localmente, sem exibi-las como publicadas. Figuras e métricas
  experimentais não foram regeneradas nesta correção editorial.
- Entrada, metodologia e curadoria disponíveis em inglês e português. Os textos
  técnicos individuais preservam seu idioma.
- `python scripts/validate_repo.py all` verifica sintaxe, links locais, testes
  isolados por arquivo e abertura de aplicativos Streamlit cadastrados sem servidor.
  Isso não certifica todos os controles, layout no navegador ou notebooks.
- As dependências em `pyproject.toml` usam faixas, sem congelamento numérico.
  Os registros históricos preservam suas configurações e não são apresentados
  como medições desta sessão.

A entrega não adiciona benchmark real, serviço em produção, impacto empresarial
ou aprovação humana de novas interpretações.

Verificado em 18/09/2026: sintaxe de 133 arquivos Python, 218 links locais em
85 arquivos Markdown, 276 testes em 25 arquivos isolados e os seis smoke tests
Streamlit aprovados no Windows com Python 3.11.14 e Streamlit 1.64.0.
Esses controles não aprovam as interpretações pendentes acima.
