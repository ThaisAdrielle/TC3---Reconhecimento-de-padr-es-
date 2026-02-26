# Trabalho Computacional 3 - Reconhecimento de Padrões

Este repositório contém a implementação e os resultados do Trabalho Computacional 3 da disciplina de Reconhecimento de Padrões.

O objetivo é avaliar o desempenho de dois classificadores supervisionados aplicados a um conjunto de dados multivariado de 24 sensores, 
bem como analisar o impacto da redução de dimensionalidade utilizando a técnica PCA (Principal Component Analysis).

---

# Objetivo

Comparar os classificadores considerando:

- Desempenho preditivo
- Robustez estatística
- Desempenho por classe
- Impacto da redução de dimensionalidade
- Preservação da variância dos dados com PCA

---

# Observações Atendidas

## OBS 1

Identificação do problema, incluindo:

- Número de classes
- Dimensão do vetor de atributos
- Número de instâncias por classe
- Verificação da invertibilidade das matrizes de covariância

---

## OBS 2

Execução de 100 rodadas independentes com cálculo de:

- Acurácia global média
- Desvio padrão da acurácia global
- Acurácia média por classe
- Desvio padrão por classe

---

## OBS 3

Implementação e avaliação dos classificadores:

- Classificador Quadrático Gaussiano (CQG)
- Classificador de Distância Mínima ao Protótipo (DMP)

Os protótipos do DMP foram obtidos utilizando o algoritmo K-means aplicado separadamente em cada classe.

---

## OBS 4

Aplicação da técnica PCA:

- Geração do gráfico da variância explicada acumulada VE(q)
- Seleção automática do número adequado de componentes principais
- Avaliação do impacto da redução de dimensionalidade no desempenho dos classificadores

---

# Modelos Implementados

## Classificador Quadrático Gaussiano (CQG)

Baseado na modelagem probabilística das classes utilizando:

- Média
- Matriz de covariância
- Probabilidade a priori

---

## Classificador de Distância Mínima ao Protótipo (DMP)

Baseado em:

- Geração de protótipos com K-means
- Classificação baseada na menor distância Euclidiana

---

## PCA (Principal Component Analysis)

Utilizado para:

- Redução de dimensionalidade
- Preservação da variância dos dados
- Avaliação do impacto no desempenho dos classificadores

---

# Resultados Gerados

O código produz automaticamente:

- Identificação do problema
- Verificação de invertibilidade das matrizes de covariância
- Gráfico da variância explicada acumulada (VE(q))
- Acurácia média global
- Desvio padrão global
- Acurácia média por classe
- Desvio padrão por classe
- Resultados com e sem PCA

---

# Estrutura do Projeto
TC3-Reconhecimento-de-Padroes/
│
├── main.py
├── README.md
└── pca.png
---

# Tecnologias Utilizadas

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

---

# Como Executar

Execute o comando:

- python main.py

O código irá automaticamente:

- Carregar o dataset
- Executar os experimentos
- Gerar o gráfico PCA
- Exibir todos os resultados

---

# Dataset

Wall Following Robot Navigation Data

Disponível em:

https://archive.ics.uci.edu/dataset/194/wall+following+robot+navigation+data 
Obs: Usar o sensor_readings_24.data

---

# Autor

Thaís Rodrigues

Trabalho desenvolvido para fins acadêmicos.
