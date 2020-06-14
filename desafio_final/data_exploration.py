import pandas as pd 

market = pd.read_csv('estaticos_market.csv')
p1 = pd.read_csv('estaticos_portfolio1.csv')
p2 = pd.read_csv('estaticos_portfolio2.csv')
p3 = pd.read_csv('estaticos_portfolio3.csv')


columns = ['de_natureza_juridica', 'sg_uf', 'natureza_juridica_macro', 'de_ramo', 'setor', 'nm_segmento', 'de_nivel_atividade']
dataset_p1 = p1[columns]

dataset_p2 = market.loc[market['id'].isin(p2['id'].to_list())][columns]
dataset_p3 = market.loc[market['id'].isin(p3['id'].to_list())][columns]



# p2 = p2[['fl_matriz', 'de_natureza_juridica', 'sg_uf', 'natureza_juridica_macro', 'de_ramo', 'setor', 'nm_segmento', 'de_nivel_atividade', 'qt_funcionarios']]
# p3 = p3[['fl_matriz', 'de_natureza_juridica', 'sg_uf', 'natureza_juridica_macro', 'de_ramo', 'setor', 'nm_segmento', 'de_nivel_atividade', 'qt_funcionarios']]

# print(p1.head())
print(dataset_p1.head())
print(dataset_p2.head())
print(dataset_p3.head())
# print(p3.head())


#colunas boas = fl_matriz, de_natureza_juridica, sg_uf, natureza_juridica_macro, de_ramo, setor, nm_segmento, de_nivel_atividade, qt_funcionarios,