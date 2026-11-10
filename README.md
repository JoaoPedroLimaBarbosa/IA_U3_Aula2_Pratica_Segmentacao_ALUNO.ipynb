# Segmentação de Clientes com Aprendizado Não Supervisionado (K-Means e DBSCAN)

Atividade prática da disciplina de Inteligência Artificial (Unidade 3, Aula 2). O projeto segmenta a base de clientes de uma rede varejista que vende vinhos, carnes, peixes, frutas, doces e joias por loja física, site e catálogo, para que o marketing possa direcionar campanhas específicas a cada segmento.

## Objetivo

Aplicar K-Means e DBSCAN em uma base real, passando por pré-processamento, seleção de variáveis, escolha do número de grupos, interpretação dos segmentos, detecção de clientes atípicos e persistência do modelo.

## Dados

Base [*Customer Personality Analysis*](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis/data) (Kaggle): 2.240 clientes e 29 colunas, usadas no estado original, sem tratamento prévio.

Colunas principais:

- **Perfil:** ano de nascimento, escolaridade, estado civil, renda anual, filhos e adolescentes em casa, data de cadastro, recência da última compra.
- **Gastos nos últimos 2 anos:** vinhos, frutas, carnes, peixes, doces e joias.
- **Compras e canais:** compras com desconto, pelo site, por catálogo e em loja, e visitas ao site no último mês.
- **Campanhas e atendimento:** aceitação das campanhas 1 a 5 e da última campanha, e reclamações.

## Etapas do notebook

1. **Análise exploratória e pré-processamento:** inspeção da base, tratamento dos valores ausentes de `Income` (mediana), codificação ordinal de `Education`, *one-hot* em `Marital_Status` (agrupando categorias inconsistentes como `Alone`, `Absurd` e `YOLO`), criação de `Tempo_Cadastro` a partir de `Dt_Customer` e remoção de identificador e colunas constantes.
2. **Seleção de variáveis e número de grupos:** método do cotovelo e silhueta para k de 2 a 8, comparando duas versões do conjunto de variáveis (A: todas as colunas; B: sem campanhas, reclamações e estado civil). A restrição de negócio foi de 3 a 5 segmentos.
3. **Interpretação dos segmentos:** tabela de perfil, nomes dos segmentos, escolha de segmentos para campanhas específicas e discussão ética e legal (LGPD) sobre o uso de estado civil e idade em decisões de crédito.
4. **Detecção de clientes atípicos:** DBSCAN sobre a versão B padronizada, com ruído entre 1% e 5% da base.
5. **Persistência do modelo:** pipeline salvo em `modelo_segmentos.pkl`.
6. **Desafio final:** remoção dos atípicos, conjunto reduzido de 6 variáveis e k = 4.

## Principais resultados

- A **versão B** (sem campanhas, reclamações e estado civil) teve silhueta maior que a versão A para os mesmos valores de k (por exemplo, 0,268 contra 0,200 com k = 2). As variáveis de campanha são resultado de uma ação de marketing e não característica do cliente, então não devem entrar na construção dos segmentos.
- Com a restrição de 3 a 5 segmentos, o k escolhido foi **3** (silhueta de 0,203).

| Segmento | Clientes | Renda média | Gasto total médio | Perfil |
|---|---|---|---|---|
| 0 | 1.042 | US$ 35.144 | US$ 98 | Renda baixa, baixo gasto, com filhos |
| 1 | 595 | US$ 76.426 | US$ 1.351 | Renda alta, alto gasto, poucos filhos (maior consumo de vinhos) |
| 2 | 603 | US$ 57.910 | US$ 748 | Renda média-alta, gasto intermediário, mais compras com desconto e pelo site |

- **Campanhas sugeridas:** o clube de vinhos importados é mais adequado ao segmento 1 (maior renda e gasto com vinhos). Os cupons para o site fazem mais sentido para o segmento 2, com mais compras com desconto e visitas ao site.
- **DBSCAN** (`eps = 3.4`, `min_samples = 5`): 88 clientes classificados como ruído (3,93%). Entre eles há prováveis erros de cadastro (anos de nascimento 1893, 1899 e 1900 e uma renda de 666.666) e clientes legítimos com padrões de consumo fora do comum.
- **Desafio final:**
  - Sem os clientes atípicos, a silhueta sobe um pouco (com k = 3, de 0,203 para 0,216).
  - Com um conjunto reduzido de 6 variáveis (renda, idade, filhos, gasto total, total de compras e visitas ao site), a silhueta com k = 3 sobe para 0,277, e o modelo fica mais simples de implantar.
  - Com k = 4, o segmento 0 se divide em dois: um de clientes mais jovens e de menor renda e outro de clientes mais velhos com mais filhos em casa.

## Como executar

### Requisitos

- Python 3.9 ou superior
- Bibliotecas: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `joblib`, `kagglehub`

```bash
pip install numpy pandas matplotlib scikit-learn joblib kagglehub
```

### Passos

1. Clone o repositório e abra o notebook `IA_U3_Aula2_Pratica_Segmentacao_ALUNO.ipynb` no VS Code, Jupyter ou Google Colab.
2. Execute as células em ordem. A base é baixada automaticamente pelo `kagglehub` (pode ser pedido login no Kaggle na primeira vez).
3. Ao final, o arquivo `modelo_segmentos.pkl` é gerado na pasta do notebook.

## Usando o modelo salvo

O arquivo guarda um dicionário com o pipeline (padronização e K-Means), a lista ordenada das 18 colunas usadas no treinamento e os nomes dos segmentos. Para classificar um cliente, passe um DataFrame com essas colunas, na mesma ordem:

```python
import joblib
import pandas as pd

carregado = joblib.load("modelo_segmentos.pkl")
print(carregado["colunas"])  # colunas esperadas pelo modelo

# cliente: DataFrame com uma linha e as colunas acima, na escala original
segmento = int(carregado["modelo"].predict(cliente[carregado["colunas"]])[0])
print(segmento, "·", carregado["nomes"][segmento])
```

> O arquivo `.pkl` depende da versão do scikit-learn usada para gerá-lo. Para evitar erros ao carregar, use a mesma versão ou gere o arquivo de novo executando o notebook.

## Limitações

- A silhueta ficou baixa (cerca de 0,2 com k = 3), então os segmentos não são bem separados e há clientes na fronteira entre grupos.
- Os nomes dos segmentos foram gerados a partir da renda, do gasto e do número de filhos e devem ser revistos pelo marketing.
- O segmento foi criado para fins de marketing e **não deve ser usado para decisões de crédito** ou outros fins que envolvam estado civil ou idade, por riscos éticos e legais (LGPD).
- Há registros com erros evidentes de cadastro (anos de nascimento improváveis e uma renda muito acima do restante) que precisam ser validados com a área responsável.

## Estrutura do repositório

```
.
├── IA_U3_Aula2_Pratica_Segmentacao_ALUNO.ipynb
├── modelo_segmentos.pkl
└── README.md
```
