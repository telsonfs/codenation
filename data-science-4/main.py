#!/usr/bin/env python
# coding: utf-8

# # Desafio 6
# 
# Neste desafio, vamos praticar _feature engineering_, um dos processos mais importantes e trabalhosos de ML. Utilizaremos o _data set_ [Countries of the world](https://www.kaggle.com/fernandol/countries-of-the-world), que contém dados sobre os 227 países do mundo com informações sobre tamanho da população, área, imigração e setores de produção.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


# In[3]:


# Algumas configurações para o matplotlib.
#%matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# In[4]:


countries = pd.read_csv("countries.csv")


# In[5]:


new_column_names = [
    "Country", "Region", "Population", "Area", "Pop_density", "Coastline_ratio",
    "Net_migration", "Infant_mortality", "GDP", "Literacy", "Phones_per_1000",
    "Arable", "Crops", "Other", "Climate", "Birthrate", "Deathrate", "Agriculture",
    "Industry", "Service"
]

countries.columns = new_column_names

countries.head()


# ## Observações
# 
# Esse _data set_ ainda precisa de alguns ajustes iniciais. Primeiro, note que as variáveis numéricas estão usando vírgula como separador decimal e estão codificadas como strings. Corrija isso antes de continuar: transforme essas variáveis em numéricas adequadamente.
# 
# Além disso, as variáveis `Country` e `Region` possuem espaços a mais no começo e no final da string. Você pode utilizar o método `str.strip()` para remover esses espaços.

# ## Inicia sua análise a partir daqui

# In[6]:


numeric_columns = [
    "Pop_density", "Coastline_ratio", "Net_migration", "Infant_mortality","Literacy", "Phones_per_1000",
    "Arable", "Crops", "Other", "Climate", "Birthrate", "Deathrate", "Agriculture",
    "Industry", "Service"]

for column in numeric_columns:
    countries[column] = countries[column].str.replace(',', '.').astype(float)

countries['Region'] = [s.strip() for s in countries.Region.to_list()]

countries.dtypes


# # Questão 1
# 
# Quais são as regiões (variável `Region`) presentes no _data set_? Retorne uma lista com as regiões únicas do _data set_ com os espaços à frente e atrás da string removidos (mas mantenha pontuação: ponto, hífen etc) e ordenadas em ordem alfabética.

# In[10]:


def q1():
    unique = [x for x in countries.Region.sort_values().unique()]
    return unique

q1()


# ## Questão 2
# 
# Discretizando a variável `Pop_density` em 10 intervalos com `KBinsDiscretizer`, seguindo o encode `ordinal` e estratégia `quantile`, quantos países se encontram acima do 90º percentil? Responda como um único escalar inteiro.

# In[7]:


def q2():
    

    est = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
    pop_density_discretizer = est.fit(countries[['Pop_density']])

    scores = pop_density_discretizer.transform(countries[['Pop_density']])

    return len(scores[scores==9])
q2()


# # Questão 3
# 
# Se codificarmos as variáveis `Region` e `Climate` usando _one-hot encoding_, quantos novos atributos seriam criados? Responda como um único escalar.

# In[8]:


def q3():
    countries_for_one_hot = countries[["Region", "Climate"]]
    countries_for_one_hot['Climate'].fillna(0, inplace=True)
    
    one_hot_encoder = OneHotEncoder(sparse=False, dtype=np.int)
    region_encoded = one_hot_encoder.fit_transform(countries_for_one_hot[["Region", "Climate"]])

    return region_encoded.shape[1]

q3()


# ## Questão 4
# 
# Aplique o seguinte _pipeline_:
# 
# 1. Preencha as variáveis do tipo `int64` e `float64` com suas respectivas medianas.
# 2. Padronize essas variáveis.
# 
# Após aplicado o _pipeline_ descrito acima aos dados (somente nas variáveis dos tipos especificados), aplique o mesmo _pipeline_ (ou `ColumnTransformer`) ao dado abaixo. Qual o valor da variável `Arable` após o _pipeline_? Responda como um único float arredondado para três casas decimais.

# In[38]:


test_country = [
    'Test Country', 'NEAR EAST', -0.19032480757326514,
    -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
    0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
    -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]

df_test_country = pd.DataFrame([dict(zip(new_column_names, test_country))])


# In[40]:


def q4():
    df_countries_numeric = countries.select_dtypes(include = ['int64', 'float64'])
    countries_numeric_columns = df_countries_numeric.columns.to_list()
    
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy='median')),
        ("standard", StandardScaler())
    ])
    
    num_pipeline.fit(df_countries_numeric)
    arable_id = countries_numeric_columns.index('Arable')
    arable_value = num_pipeline.transform(df_test_country[countries_numeric_columns])[0][arable_id]
    
    return float(round(arable_value, 3))
     
q4()


# ## Questão 5
# 
# Descubra o número de _outliers_ da variável `Net_migration` segundo o método do _boxplot_, ou seja, usando a lógica:
# 
# $$x \notin [Q1 - 1.5 \times \text{IQR}, Q3 + 1.5 \times \text{IQR}] \Rightarrow x \text{ é outlier}$$
# 
# que se encontram no grupo inferior e no grupo superior.
# 
# Você deveria remover da análise as observações consideradas _outliers_ segundo esse método? Responda como uma tupla de três elementos `(outliers_abaixo, outliers_acima, removeria?)` ((int, int, bool)).

# In[45]:


def q5():
    net_migration = countries['Net_migration']
    q1 = net_migration.quantile(.25)
    q3 = net_migration.quantile(.75)
    
    iqr = q3 - q1
    
    outliers_acima = len(net_migration[net_migration > q3 + 1.5*iqr])
    outliers_abaixo = len(net_migration[net_migration < q1 - 1.5*iqr])

    
    removeria = bool((outliers_abaixo + outliers_acima)/len(net_migration) < 0.2)
    return (outliers_abaixo, outliers_acima, removeria)
q5()


# ## Questão 6
# Para as questões 6 e 7 utilize a biblioteca `fetch_20newsgroups` de datasets de test do `sklearn`
# 
# Considere carregar as seguintes categorias e o dataset `newsgroups`:
# 
# ```
# categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
# newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
# ```
# 
# 
# Aplique `CountVectorizer` ao _data set_ `newsgroups` e descubra o número de vezes que a palavra _phone_ aparece no corpus. Responda como um único escalar.

# In[69]:


categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)


# In[65]:


def q6():
    
    vectorizer = CountVectorizer()
    vectorizer_transform = vectorizer.fit_transform(newsgroup['data'])
    
    word_list = vectorizer.get_feature_names();    
    count_list = vectorizer_transform.toarray().sum(axis=0)
    vectorize_dict = dict(zip(word_list,count_list))
    
    return vectorize_dict['phone']
    
q6()


# ## Questão 7
# 
# Aplique `TfidfVectorizer` ao _data set_ `newsgroups` e descubra o TF-IDF da palavra _phone_. Responda como um único escalar arredondado para três casas decimais.

# In[73]:


def q7():
    
    vectorizer = TfidfVectorizer()
    vectorizer_transform = vectorizer.fit_transform(newsgroup['data'])
    
    word_list = vectorizer.get_feature_names();    
    count_list = vectorizer_transform.toarray().sum(axis=0)
    vectorize_dict = dict(zip(word_list,count_list))
    
    return round(vectorize_dict['phone'], 3)
q7()


# In[ ]:




