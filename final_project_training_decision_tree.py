#from asyncore import file_dispatcher
#from email import encoders
import numpy as np
import pandas as pd
import datetime as dt
from datetime import date
import seaborn as sn
import matplotlib.pyplot as plt

from sklearn import model_selection
#from sklearn.model_selection import RandomizedSearchCV
#from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
#from sklearn.model_selection import cross_val_score, cross_validate
#from sklearn.metrics import classification_report, accuracy_score
#from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
#from sklearn.metrics import roc_curve,  plot_roc_curve
from sklearn.preprocessing import LabelEncoder

import pickle

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt import space_eval
from hyperopt.pyll import scope

import mlflow
from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner

seed = 1275

@task
def read_dataframe():
    df_96_05 = pd.read_csv("fires_1996to2005.csv", encoding='cp1252')
    df_06_18 = pd.read_csv("fires_2006to2018.csv", encoding='cp1252')
    df_96_18 = pd.concat([df_96_05, df_06_18])

   # Remoção de espaço em branco nos valores das colunas categóricas
    columns_categorics = df_96_18.select_dtypes(include="object").columns.to_list()
    for columns in columns_categorics:
        df_96_18[columns] = df_96_18[columns].str.strip()

    void_reg_fire_type_index = list(df_96_18[df_96_18['fire_type'] == '']['fire_type'].index)

    # Remoção dos registros nos quais "fuel_type" está vazio
    df_96_18.drop(void_reg_fire_type_index, axis=0, inplace=True)

    # Busca dos indices para os quais o registro de "fuel_type" é igual a espaço vazio
    void_reg_fuel_type_index = list(df_96_18[df_96_18['fuel_type'] == '']['fuel_type'].index)

    # Remoção dos registros nos quais "fuel_type" está vazio
    df_96_18.drop(void_reg_fuel_type_index, axis=0, inplace=True)

    # Após exclusão de registros, reordena os index
    df_96_18.reset_index(drop=True, inplace=True)

    # Preenchimento dos NAN do atributo `fuel_type`
    for i,fuel in enumerate(df_96_18['fuel_type']):
        if pd.isna(fuel):
            if not pd.isna(df_96_18.loc[i,'other_fuel_type']):
                df_96_18.loc[i,'fuel_type'] = 'OF'               # OF = Other Fuel

    # Ajuste dos Tipos de Variáveis
    date_columns = ['fire_start_date', 'discovered_date', 'reported_date', 'start_for_fire_date', 'fire_fighting_start_date',
                    'bh_fs_date', 'uc_fs_date', 'ex_fs_date']

    for date in date_columns:
        df_96_18[date] = pd.to_datetime(df_96_18[date])

    # Remoção de Colunas
    removed_columns = ['fire_name', 'industry_identifier_desc', 'responsible_group_desc', 'activity_class', 'true_cause', 
                    'permit_detail_desc', 'other_fuel_type', 'to_fs_date', 'to_hectares', 'assessment_datetime', 'fire_year',
                    'calendar_year', 'fire_fighting_start_size', 'discovered_date', 'fire_fighting_start_date',
                    'reported_date', 'start_for_fire_date']
    df_96_18 = df_96_18.drop(removed_columns, axis=1)

    # Remoção de Linhas (Registros)
    df_96_18.dropna(axis=0, how='any', inplace=True, subset=df_96_18.columns)
    df_96_18.reset_index(drop=True, inplace=True)
    
    ### Adição de novos atributos
    df_96_18['fire_start_day'] = df_96_18['fire_start_date'].apply(lambda dt: dt.day)      
    df_96_18['fire_start_month'] = df_96_18['fire_start_date'].apply(lambda dt: dt.month)      
    df_96_18['fire_start_year'] = df_96_18['fire_start_date'].apply(lambda dt: dt.year)

    # Criando feature de turnos
    def define_day_period(dt):
        hour = dt.strftime("%H:%M:%S")
        if (hour >= '00:00') and (hour <= '12:00'):
            return 'morning'
        elif (hour > '12:00') and (hour <= '18:00'):
            return 'afternoon'
        else:
            return 'night'
    
    df_96_18['day_period'] = df_96_18['fire_start_date'].apply(lambda dt: define_day_period(dt))

    # Criando feature de estações do ano
    def define_season(dt):
        month = dt.month
        if (month >= 3) and (month <= 5):
            return 'spring'
        elif (month >= 6) and (month <= 8):
            return 'summer'
        elif (month >= 9) and (month <= 11):
            return 'autumn'
        else:
            return 'winter'
    
    df_96_18['seasons'] = df_96_18['fire_start_date'].apply(lambda dt: define_season(dt))

    # Criando feature de tempo de controle 
    df_96_18['control_time'] = (df_96_18['uc_fs_date'] - df_96_18['fire_start_date']).apply(lambda t: (t.components[0]*24*60) + (t.components[1]*60) + t.components[2])

    # Criando feature de tempo de extinção
    df_96_18['extinction_time'] = (df_96_18['ex_fs_date'] - df_96_18['fire_start_date']).apply(lambda t: (t.components[0]*24*60) + (t.components[1]*60) + t.components[2])

    # Criando feature delta de extinção: intervalo de tempo entre o controle do fogo e a extinção
    df_96_18['extinction_delta'] = df_96_18['extinction_time'] - df_96_18['control_time']

    # Criando feature eficiência de extinção: tempo de extinçao do incêndio / (area total queimada * 10000)
    # A multiplicação por 10000 converte hectares em metros quadrados
    df_96_18['extinction_efficiency'] = round(df_96_18['extinction_time'] / (df_96_18['ex_hectares'] * 10000), 4)

    # Criando feature causa do fogo: natural ou humana
    df_96_18['fire_cause'] = df_96_18['general_cause_desc'].apply(lambda c: 'natural' if c == 'Lightning' else 'human' )

    # Criando feature de área de proteção florestal
    def define_forest_protection_area(l):
        cod = str(l[0:3]).lower().strip()
        if (cod == 'cwf') or (cod[1:] == '01'):
            return 'Calgary'
        elif (cod == 'ewf') or (cod[1:] == '02'):
            return 'Edson'
        elif (cod == 'hwf') or (cod[1:] == '03'):
            return 'High Level'
        elif (cod == 'gwf') or (cod[1:] == '04'):
            return 'Grande Prairie'
        elif (cod == 'lwf') or (cod[1:] == '05'):
            return 'Lac La Biche'
        elif (cod == 'mwf'):
            return 'Fort McMurray'
        elif (cod == 'pwf') or (cod[1:] == '06'):
            return 'Peace River'
        elif (cod == 'rwf') or (cod[1:] == '07'):
            return 'Rocky'
        elif (cod == 'swf') or (cod[1:] == '08'):
            return 'Slave Lake'
        elif (cod == 'wwf') or (cod[1:] == '09'):
            return 'Whitecourt' 
        else:
            return 'Not Defined'
    
    df_96_18['forest_protection_area'] = df_96_18['fire_number'].apply(lambda l: define_forest_protection_area(l))

    # Excluindo registros com tempo de controle igual a zero
    zero_reg_control_time_index = list(df_96_18[df_96_18['control_time'] == 0].index)
    df_96_18.drop(zero_reg_control_time_index, axis=0, inplace=True)
    df_96_18.reset_index(drop=True, inplace=True)

    # Remoção de colunas após a criação das features
    df_96_18 = df_96_18.drop(['fire_number', 'general_cause_desc', 'det_agent', 'fire_start_date', 'bh_fs_date', 'uc_fs_date', 'ex_fs_date'], axis=1)

    return df_96_18

@task
def df_split(df):
    # Cria novo df para os modelos de ML 
    df_ml = df.drop(['fire_number', 'general_cause_desc', 'det_agent', 'fire_start_date', 'bh_fs_date', 'uc_fs_date', 'ex_fs_date'], axis=1)

    # Divisão entre 'features' e 'target'
    X = df_ml.drop(['fire_cause'], axis=1)
    y = df_ml['fire_cause']

    # Armazenando os nomes das variáveis categóricas e numéricas em listas
    cat_cols = X.select_dtypes(include="object").columns.to_list()
    #num_cols = X.select_dtypes(exclude="object").columns.to_list()
    
    dict_encoders = {}

    for atr in cat_cols:
        le = LabelEncoder()
        le.fit(X[atr])
        dict_encoders[atr] = le
        X[atr] = le.transform(X[atr])

    le_tree = LabelEncoder()
    X_tree = np.array(X)
    y_tree = le_tree.fit_transform(y)

    # Divisao da base de dados em treinamento, validacao e teste
    X_train, X_test, y_train, y_test = train_test_split(X_tree, y_tree, test_size=0.25, stratify=y_tree, random_state=seed)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.3333, stratify=y_train, random_state=seed)

    return X_train, X_valid, X_test, y_train, y_valid, y_test, dict_encoders
'''
@task
def evaluate(sk_model, x_test, y_test):
    eval_acc = sk_model.score(x_test, y_test)
    preds = sk_model.predict(x_test)
    auc_score = roc_auc_score(y_test, preds)

    mlflow.log_metric("eval_acc", eval_acc)
    mlflow.log_metric("auc_score", auc_score)

    print(f"Auc Score: {auc_score:.3%}")
    print(f"Eval Accuracy: {eval_acc:.3%}")

    roc_plot = plot_roc_curve(sk_model, x_test, y_test, name='Scikit-learn ROC Curve')
    plt.savefig("sklearn_roc_plot.png")
    plt.show()
    plt.clf()

    conf_matrix = confusion_matrix(y_test, preds)
    ax = sn.heatmap(conf_matrix, annot=True,fmt='g')
    ax.invert_xaxis()
    ax.invert_yaxis()
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title("Confusion Matrix")
    plt.savefig("sklearn_conf_matrix.png")

    mlflow.log_artifact("sklearn_roc_plot.png")
    mlflow.log_artifact("sklearn_conf_matrix.png")

    return auc_score
'''
@task
def train_model_search(train, valid, y_val, y_train):
    def objective(params):
        with mlflow.start_run():
            #mlflow.set_tag("model", "decision_tree")
            #mlflow.log_params(params)
            mlflow.sklearn.autolog()

            decisionTree = DecisionTreeClassifier(**params,random_state=seed)
            model_dt = decisionTree.fit(train, y_train)
            
            #y_pred = decisionTree.predict(valid)

            y_scores = model_dt.predict_proba(valid)
            auc = roc_auc_score(y_val, y_scores[:,1])
            mlflow.log_metric("auc", auc)
            mlflow.sklearn.log_model(model_dt, "log_dt_model")
        mlflow.end_run()

        return {'loss': -auc, 'status': STATUS_OK}

    search_space = {
        'max_depth': scope.int(hp.choice('max_depth', [2,3,4,5,6,7,8,9,10])),
        'criterion': hp.choice('criterion', ['gini', 'entropy']),
        'min_samples_split': hp.choice('min_samples_split', [7,8,9,10,12,15,18,20,22,24]),
        'min_samples_leaf': hp.choice('min_samples_leaf', [3,4,5,6,7,8,9,10,12,14,16,18]),
    }

    best_result = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=100,
        trials=Trials()
    )
    return space_eval(search_space, best_result)

@task
def train_best_model(X_train, X_val, y_train, y_val, dict_encoders, best_params):
    with mlflow.start_run():

        #mlflow.log_params(best_params)
        mlflow.sklearn.autolog()

        cat_cols = ['size_class', 'fire_origin', 'det_agent_type', 'initial_action_by', 'fire_type', 'fire_position_on_slope',
            'weather_conditions_over_fire', 'fuel_type', 'day_period', 'seasons', 'forest_protection_area']

        decisionTree = DecisionTreeClassifier(**best_params,random_state=seed)
        model_dt = decisionTree.fit(X_train, y_train)

        #auc_score = evaluate(model_dt, X_val, y_val)
        #mlflow.log_metric("auc", auc_score)
        y_scores = model_dt.predict_proba(X_val)
        auc = roc_auc_score(y_val, y_scores[:,1])
        mlflow.log_metric("auc", auc)
        mlflow.sklearn.log_model(model_dt, "log_dt_model")

        for col in cat_cols:
            file_name = 'preprocessors/' + col + '.pkl'
            file_output = open(file_name, 'wb')
            pickle.dump(dict_encoders[col], file_output)
            file_output.close()
            mlflow.log_artifact(file_name, artifact_path="encoder")

        #file_output = open("decision_tree_encoder.pkl", 'wb')
        #pickle.dump(label_encoder, file_output)
        #file_output.close()
        #mlflow.log_artifact("decision_tree_encoder.pkl", artifact_path="encoder")
    mlflow.end_run()

@flow(task_runner=SequentialTaskRunner())
def main():
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("decision-tree-alberta-wildfire-final")
    # mlflow.sklearn.autolog()

    df = read_dataframe()
    X_train, X_val, X_test, y_train, y_val, y_test, label_encoders = df_split(df)
    
    best_params = train_model_search(X_train, X_val, y_val, y_train)

    train_best_model(X_train, X_val, y_train, y_val, label_encoders, best_params)
    
    print(best_params)

main()