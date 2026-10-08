# People Analytics: Plataforma Inteligente de Prevenção de Turnover

Trabalho da disciplina de **Data Science** — Desafio Integrador, Squad 4 (People Analytics).

**Integrantes:** Gabriel Maia, João Lucca, Kevin Mascarenhas, Marina Gonçalves e Samuel Paulo

---

## Sobre o projeto

Usamos o histórico de 10 anos de colaboradores de uma rede varejista (2006 a 2015) para entender por que as pessoas saem da empresa e prever quem tem mais risco de **pedir demissão**.

O resultado é uma **nota de risco** para cada colaborador, calculada por um modelo XGBoost. Com ela, o RH monta uma lista de prioridade e os gestores podem conversar com essas pessoas antes que elas decidam sair, em vez de agir só depois do pedido de demissão.

Como proposta de evolução, mostramos como um **co-piloto de IA generativa** poderia usar a nota e os fatores do modelo para preparar o gestor para essa conversa (roteiro 1:1, ações de retenção e PDI). Nesta entrega, o co-piloto ainda não está conectado a uma IA generativa: o notebook monta automaticamente as informações e o prompt que seriam enviados.

**Fluxo da solução:**

Dados de RH → limpeza e análise → modelo preditivo (XGBoost) → nota de risco → lista de prioridade → co-piloto (proposta) → conversa 1:1 preventiva

## Principais resultados

* Os pedidos de demissão se concentram nas lojas, principalmente entre caixas e atendentes, em pessoas jovens e nos primeiros anos de empresa.
* Comparamos Árvore de Decisão, Random Forest e XGBoost. Escolhemos o XGBoost pela validação cruzada.
* Na base de teste, com o limiar ajustado, o modelo encontrou **48%** dos pedidos de demissão (Recall), e **23%** das pessoas apontadas realmente pediram demissão (Precision). Escolhendo pessoas ao acaso, esse número seria de menos de 1%.
* Um modelo que nunca prevê pedido de demissão teria 99,2% de acurácia, por isso avaliamos os modelos por Recall, Precision e PR-AUC.

## Arquivos

| Arquivo | Conteúdo |
| :--- | :--- |
| [`People Analytics Entrega Geral - Definitiva.ipynb`](./People%20Analytics%20Entrega%20Geral%20-%20Definitiva.ipynb) | Notebook principal: ingestão, Data Quality Report, EDA, Feature Engineering, treinamento e avaliação dos modelos, conclusão e proposta do co-piloto (seção 9). |
| [`ENTREGA_DESAFIO_INTEGRADOR_SQUAD4.md`](./ENTREGA_DESAFIO_INTEGRADOR_SQUAD4.md) | Resumo da entrega: problema, dados, modelo, co-piloto, métricas, riscos, resposta ao desafio adicional e roteiro da apresentação. |
| [`MFG10YearTerminationData.csv`](./MFG10YearTerminationData.csv) | Base de dados utilizada. |
| [`requirements.txt`](./requirements.txt) | Bibliotecas necessárias. |
| [`legado/`](./legado/) | Versões anteriores dos notebooks e documentos. |

## Como executar

**Google Colab:** faça upload do notebook e do arquivo `MFG10YearTerminationData.csv` (na mesma pasta do notebook) e rode todas as células em ordem.

**Localmente:**

1. Clone o repositório:
   ```bash
   git clone https://github.com/lucca3447/auladatascience.git
   cd auladatascience
   ```
2. (Opcional) Crie e ative um ambiente virtual:
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Linux/Mac:
   source venv/bin/activate
   ```
3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
4. Abra o notebook e rode todas as células (Kernel → Restart & Run All):
   ```bash
   jupyter lab "People Analytics Entrega Geral - Definitiva.ipynb"
   ```

O notebook procura o CSV na mesma pasta ou na subpasta `dataset/`. Durante a execução, a seção 5 gera o arquivo `dataset_preparado_feature_engineering.csv`, que é usado na seção 6 para treinar os modelos.
