# Documentação Projeto MLOps
> Projeto desenvolvido por: Wanderson R. Marques , David S. Monte, Daniela C. de Sena

![](header.png)

## Definição do projeto

O projeto consistiu em fazer a classificação de causa de incêndio na província de Alberta no Canadá, abordando os conceitos e práticas do ciclo de desenvolvimento de MLOps (Machine Learning Operations) para realizar análise, armazenamento, controle, manutenção e testes dos modelos. As fases de desenvolvimento consistiram na análise do conjunto de dados, pré-processamento do conjunto de dados, seleção de modelos para treinamento e avaliação, configuração do ambiente, rastreamento dos experimentos com gerenciamento do fluxo de trabalho e desenvolvimento de uma API para consumo do modelo que foi empregada nos testes realizados em todo o projeto. A seguir serão descritas cada uma dessas etapas.
 

## Dataframe:

As informações contidas no dataset abrangem detalhes específicos de cada ocorrência de incêndio registrada, tais como: área florestal, latitude, longitude, área atingida pelo fogo, origem do fogo, prováveis causas, data e hora de detecção, de início de combate, de controle e de extinção do incêndio.
> Para mais informações sobre o dataset, visite: https://wildfire.alberta.ca/resources/historical-data/historical-wildfire-database.aspx 

## Descrição dos arquivos

* alberta_fires_1996to2005.csv
    - Dados dos incêndios registrados na província de Alberta do ano de 1996 até 2005  
* alberta_fires_2006to2018.csv
    - Dados dos incêndios registrados na província de Alberta do ano de 2006 até 2018  
* features_alberta_canada.pdf
    - Descreve cada feature utilizada no projeto 
* final_project_trainiing_decision_tree.py
    - Código com pré-processamento e treinamento do Decision Tree 
* final_project_trainiing_random_forest.py
    - Código com pré-processamento e treinamento do Random Forest
* final_project_predict.py
    - Realiza a predição e criação do modelo
* final_project_test.py
    - Realiza o teste do modelo com a classificação
* fires_1996to2005_set_test.csv
    - Arquivo de teste de classificação
* artifacts/
    - Possui o modelo salvo que foi colocado em produção, os encoders utilizados no treinamento do modelo, a matriz de confusão, a curva ROC e a curva Precision Recall      
 
