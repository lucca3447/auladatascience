# Desafio Integrador — People Analytics
## Plataforma Inteligente de Prevenção de Turnover

**Squad 4:** Gabriel Maia, João Lucca, Kevin Mascarenhas, Marina Gonçalves e Samuel Paulo
**Disciplina:** Data Science
**Base de dados:** `MFG10YearTerminationData.csv` (registros anuais de colaboradores de 2006 a 2015)
**Notebook com todo o código:** [`People Analytics Entrega Geral - Definitiva.ipynb`](./People%20Analytics%20Entrega%20Geral%20-%20Definitiva.ipynb)

Este documento resume a nossa solução. Os detalhes, gráficos e números estão no notebook.

**Resumo da solução:** usamos o histórico de colaboradores para treinar um modelo (XGBoost) que dá uma nota de risco de pedido de demissão para cada colaborador. Com essa nota, o RH monta uma lista de prioridade. A proposta é que um co-piloto de IA generativa use a nota e os fatores do modelo para preparar o gestor para uma conversa 1:1 com quem está na lista. A ideia é sair de uma postura reativa (agir só depois do pedido de demissão) para uma postura preventiva.

---

## 1. Problema de Negócio

**Qual problema queremos resolver?**
Os pedidos de demissão em uma rede varejista. Na análise dos dados, eles se concentram nas lojas, principalmente entre caixas e atendentes, em pessoas jovens e nos primeiros anos de empresa.

**Quem usaria a solução?**
* **BPs de Gente & Gestão (RH):** acompanham o risco por loja, departamento e cargo e organizam os programas de retenção.
* **Gestores de loja e supervisores:** recebem a lista de prioridade e conversam com os colaboradores.

**Qual decisão queremos melhorar?**
* **Hoje (reativa):** o gestor só fica sabendo da insatisfação quando recebe o pedido de demissão e precisa abrir uma vaga às pressas.
* **Com a solução (preventiva):** o gestor sabe com antecedência quem tem mais risco e pode conversar sobre carreira e rotina antes que a pessoa decida sair.

**Qual o impacto do problema?**
Custos de rescisão, recrutamento e treinamento de quem entra, além de perda de experiência no atendimento e sobrecarga de quem fica.

---

## 2. Dados

**Dados utilizados:** idade, tempo de empresa, cargo, departamento, unidade de negócio (escritório central ou loja), loja, gênero e situação no ano (ativo ou desligado, com o motivo). Em uma versão real, esses dados viriam do sistema de RH e da folha de pagamento.

**Variável alvo:** `desligamento_voluntario`, igual a 1 quando o colaborador pediu demissão naquele ano. Deixamos de fora as aposentadorias (acontecem sempre aos 60 ou 65 anos, então são previsíveis só pela idade) e os layoffs (decisão da empresa, concentrada em 2014 e 2015). Só 0,78% dos registros são pedidos de demissão.

**Variáveis usadas no modelo:**
* Originais: idade e tempo de empresa.
* Criadas por nós: idade na admissão, faixa de tempo de casa, tamanho da loja no ano e proporção da vida adulta passada na empresa (sugerida com ajuda de IA generativa).
* Categóricas transformadas com One-Hot Encoding: cargo, departamento, unidade de negócio e gênero.

**Problemas de qualidade que tratamos:**
* A data `1/1/1900` era usada como "sem data de desligamento"; trocamos por valor ausente.
* O motivo `Resignaton` estava escrito errado (e o departamento `Accounts Receiveable`); corrigimos a grafia.
* 5 colaboradores tinham dois registros no mesmo ano (um ativo e outro desligado); mantivemos só o de desligamento.
* A data e o motivo do desligamento revelam a resposta, então não entraram no modelo (evitando vazamento de dados).

---

## 3. Modelo de IA Preditiva

**Tipo de problema:** classificação binária (pede ou não pede demissão no ano), com classes muito desbalanceadas.

**Modelos comparados:** Árvore de Decisão, Random Forest e XGBoost. Escolhemos o **XGBoost**, que teve a maior PR-AUC na validação cruzada (0,141). O Random Forest ficou muito perto (0,135).

**Como treinamos e validamos:**
* Separamos 80% para treino e 20% para teste com `StratifiedGroupKFold`, para que o mesmo colaborador nunca apareça no treino e no teste ao mesmo tempo.
* Para lidar com o desbalanceamento, usamos `scale_pos_weight = 128` no XGBoost (e `class_weight='balanced'` nos outros modelos).
* Escolhemos os hiperparâmetros com validação cruzada dentro do treino, olhando a diferença entre treino e validação para controlar o overfitting.
* Trocamos o limiar padrão de 0,5 por 0,92, que foi o limiar com maior F1 nas previsões fora da amostra do treino. O teste só foi usado no final.

**Resultados na base de teste (limiar 0,92):**

| Métrica | Resultado |
| :-- | :-- |
| Recall | 48% (encontrou 37 dos 77 pedidos de demissão) |
| Precision | 23% (dos 163 apontados, 37 pediram demissão) |
| F1-Score | 0,31 |
| ROC-AUC | 0,847 |
| PR-AUC | 0,179 (um chute aleatório teria 0,008) |

Não usamos a acurácia para escolher o modelo: um modelo que nunca prevê pedido de demissão tem 99,2% de acurácia e não encontra ninguém.

**Limitações:** a base vai só até 2015 e não tem salário, horas extras, avaliação de desempenho nem pesquisa de clima. As notas do modelo também não são probabilidades calibradas; servem para ordenar quem tem mais risco.

---

## 4. Co-piloto de IA Generativa (proposta)

**Papel do co-piloto:** transformar a nota de risco em uma orientação prática para o gestor: o que pesou no risco, como conduzir a conversa e o que oferecer ao colaborador.

**Situação atual:** o co-piloto ainda **não está conectado** a uma IA generativa. No notebook (seção 9), montamos automaticamente, a partir do modelo treinado, tudo o que seria enviado para ela.

**Informações enviadas ao co-piloto:**
* Perfil do colaborador: cargo, departamento, unidade, loja, idade e tempo de empresa (sem nome nem identificador).
* Nota de risco calculada pelo XGBoost, o limiar de alerta e se a pessoa está em alerta.
* Os fatores que mais aumentaram o risco daquela pessoa, calculados pelo próprio XGBoost (contribuição de cada variável na previsão).

Exemplo real gerado no notebook para o colaborador com maior nota da base de teste: caixa do Atendimento ao Cliente, 21 anos, 1 ano de empresa, loja 46 (Victoria), **nota de risco 0,974**. Os fatores que mais pesaram foram o tempo de empresa, a idade, a proporção da vida adulta na empresa e o tamanho da loja.

**Instruções (prompt) do co-piloto:**

```text
Você é o co-piloto de retenção de uma rede varejista. Seu papel é ajudar o gestor
a preparar uma conversa individual (1:1) com um colaborador que o modelo preditivo apontou
como prioridade.

Regras:
1. Use apenas as informações do pacote de dados. Não presuma salário, satisfação ou problemas pessoais.
2. A abordagem deve ser de acolhimento e desenvolvimento. Nunca sugira punição, desligamento ou corte de oportunidades.
3. Não use a nota de risco como rótulo na conversa: ela serve só para o gestor priorizar quem procurar.
4. Não trate idade ou gênero como problema do colaborador.
5. Lembre o gestor de que o modelo erra: a pessoa pode não ter intenção de sair.

Responda em quatro seções:
1. Síntese dos fatores associados ao risco
2. Roteiro para a conversa 1:1 (abertura, 3 perguntas e fechamento)
3. Ações imediatas de retenção
4. Sugestão de PDI de 90 dias
```

**Exemplo do tipo de resposta esperada** (escrito pelo grupo para ilustrar, a partir do exemplo acima):

> **1. Fatores:** pessoa no primeiro ano de empresa, em cargo de entrada no atendimento de uma loja grande. Pelo histórico, esse é o perfil que mais pede demissão.
>
> **2. Conversa 1:1:** abrir perguntando como está sendo o primeiro ano na loja. Perguntas: o que tem sido mais difícil na rotina? Que outra área da loja te interessa? O que eu posso fazer para melhorar seu dia a dia? Fechar combinando um próximo encontro.
>
> **3. Ações imediatas:** indicar um colega mais experiente como referência, alternar o caixa com outras tarefas quando possível e dar retorno frequente sobre o trabalho.
>
> **4. PDI de 90 dias:** mês 1, conversas quinzenais com o gestor; mês 2, acompanhar o supervisor em algumas rotinas; mês 3, conversar sobre próximos passos de carreira na loja.

**Como o gestor usaria:**
1. Todo mês o modelo calcula a nota dos colaboradores ativos e monta a lista de prioridade de cada loja.
2. Para cada pessoa da lista, o co-piloto gera a orientação.
3. O gestor revisa a orientação, conversa com o colaborador e registra o que foi combinado.
4. O RH acompanha os resultados.

---

## 5. Fluxo da Solução

1. **Dados de RH:** histórico de colaboradores.
2. **Preparação:** limpeza da base, análise exploratória e criação de features.
3. **Modelo preditivo (XGBoost):** calcula a nota de risco de pedido de demissão.
4. **Lista de prioridade:** colaboradores acima do limiar, com os fatores que pesaram na nota.
5. **Co-piloto (proposta):** gera a orientação para a conversa a partir da nota e dos fatores.
6. **Gestor:** conversa 1:1 preventiva, ações de retenção e PDI.

---

## 6. Métricas

**Métricas do modelo:** Recall (quantos pedidos de demissão o modelo encontra), Precision (quantos dos apontados realmente saem), F1 e PR-AUC. Na base de teste: Recall de 48%, Precision de 23% e PR-AUC de 0,179. Escolher 163 pessoas ao acaso encontraria pouco mais de 1 pedido de demissão; a lista do modelo encontrou 37.

**Métricas de negócio que acompanharíamos** (as metas dependem de dados que a base não tem, como custo de contratação e salário):
* Taxa de pedidos de demissão nas lojas que usam a plataforma, comparada com as que não usam.
* % de colaboradores da lista que continuam na empresa 6 meses depois da conversa.
* % de colaboradores abordados que começaram o PDI em até 30 dias.
* Recall e Precision acompanhados mês a mês, para perceber se o modelo está piorando.

---

## 7. Riscos e Limitações

1. **Qualidade dos dados:** faltam salário, desempenho, horas extras e clima. Isso limita o modelo.
2. **Viés de idade:** idade e tempo de casa são os fatores mais fortes. O modelo nunca deve ser usado para recusar candidatos jovens, negar promoções ou justificar desligamentos.
3. **Privacidade (LGPD):** a nota de risco é um dado pessoal. Só o BP responsável e o gestor direto deveriam ver, e o colaborador tem direito de saber e pedir revisão de decisões baseadas em tratamento automatizado.
4. **Segurança:** os dados enviados ao co-piloto precisam ficar em um ambiente seguro, sem uso para treinar modelos públicos.
5. **Erros da IA generativa:** ela pode inventar motivos que não estão nos dados (por exemplo, insatisfação com salário). Por isso o prompt limita a resposta ao pacote de dados e o gestor revisa tudo antes da conversa.
6. **Interpretação errada da nota:** se o gestor achar que a pessoa "já vai sair", pode parar de investir nela e provocar a saída.
7. **Dependência da ferramenta:** a decisão é sempre de uma pessoa. Conversar com a equipe não pode depender de um alerta.

---

## 8. Desafio Adicional

> *"Se esta solução fosse colocada em produção amanhã, qual seria o maior risco de utilizá-la para apoiar decisões reais?"*

Para nós, o maior risco é a **nota de risco ser interpretada de forma errada, junto com o viés de idade**.

* **Técnico:** com o limiar que escolhemos, o modelo encontra 48% dos pedidos de demissão, mas só 23% dos apontados realmente saem. Ou seja, de cada 4 pessoas na lista, cerca de 3 não pediriam demissão.
* **Gestão:** se o gestor usar a lista de forma punitiva (deixar a pessoa fora de treinamentos ou negar uma promoção porque "ela vai sair mesmo"), ele mesmo pode provocar a saída que o modelo previu.
* **Ético:** como o modelo dá muito peso à pouca idade, decisões apressadas prejudicariam justamente os colaboradores mais jovens.

Por isso, a solução foi pensada só para acolhimento e desenvolvimento. Ela não deve ser usada para justificar demissões, cortar oportunidades ou aplicar punições.

