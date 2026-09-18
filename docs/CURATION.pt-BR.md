# Registro de curadoria e evidências

[English](CURATION.md) · [Entrada do laboratório](../README.pt-BR.md)

Curadoria de 18/09/2026, a partir da revisão `b369445`. Escopo: selecionar entradas,
corrigir navegação e caminhos de execução e explicitar o estado da revisão.
Não substitui os registros históricos, nem certifica todas as afirmações científicas
ou altera autoria. Nenhum experimento histórico foi reexecutado para a edição.

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
