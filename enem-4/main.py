import pandas as pd
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split


from src.preprocessing import Preprocessing
from src.visualization import Visualization
from src.experiments import Experiments


# Read Data
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# Select features
print(Preprocessing.data_info(train))
train.drop(['Unnamed: 0', 'NU_INSCRICAO'], axis=1, inplace = True)
numeric_features = list(test.select_dtypes(include='number').columns)
df_train = train[numeric_features]
df_train['IN_TREINEIRO'] = train[['IN_TREINEIRO']]
Preprocessing.select_features(train, 'IN_TREINEIRO', numeric_features)

Visualization.correlation_features(df_train)

columns = ['NU_IDADE']
df_train = Visualization.verify_outliers(df_train, 9, 120, columns)

df_train['IN_TREINEIRO'].value_counts()

x = df_train.drop('IN_TREINEIRO', axis=1)
target = df_train['IN_TREINEIRO']

Visualization.balancing_analysis(x, target)

x, target = Preprocessing.balancing_target(x, target)

x_train, x_test, y_train, y_test = train_test_split(x, target, test_size=0.2, random_state=25)

experiments_enem = Experiments(x_train, y_train, x_test, y_test)

logistic_regression, logistic_regression_measurement = experiments_enem.models_classification(LogisticRegression())
print(logistic_regression_measurement)

kneighbors_classifier, kneighbors_classifier_measurement = experiments_enem.models_classification(KNeighborsClassifier())
print(kneighbors_classifier_measurement)

random_forest_classifier, random_forest_classifier_measurement = experiments_enem.models_classification(RandomForestClassifier())
print(random_forest_classifier_measurement)

decision_tree_classifier, decision_tree_classifier_measurement = experiments_enem.models_classification(DecisionTreeClassifier())
print(decision_tree_classifier_measurement)

print(test.columns.to_list())

df_test = test[x.columns.to_list()]
df_test.fillna(0, inplace = True)

df_response = test[['NU_INSCRICAO']]
df_response['IN_TREINEIRO'] = decision_tree_classifier.predict(df_test)
df_response.to_csv('answer.csv', index=False)


