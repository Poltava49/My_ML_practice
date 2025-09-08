import matplotlib.pyplot as plt
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import seaborn as sns
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error


def show_all_regressor_metrics(y_pred, y_test, title):
    print(f'============================={title}========================================')
    print(f"R2 - {r2_score(y_test, y_pred)}")
    print(f"MSE - {mean_squared_error(y_test, y_pred)}")
    print(f"RMSE - {root_mean_squared_error(y_test, y_pred)}")
    print('=============================================================================')


def check_null(df):
    view_null = np.round(df.isna().sum().max()/df.shape[0] * 100)
    if view_null < 15:
        print(f"Доля {np.round(df.isna().sum().max()/df.shape[0] * 100)}% пропусков")
        if df.dropna().shape[0] < df.shape[0]*0.85:
             print('НО доля пропусков на тотале сильно больше 15%. Будем заполнять пропуски')
             return df
        else:
            print('Доля пропусков на тотале меньше  20%. Удаляем пропуски')
            return df.dropna()
    else:
        print('Меньше 20%')
        return df


def plots_correlation_numeric_features_with_target(data, num_col):
    num_col = num_col.to_list()
    num_col.remove('Listening_Time_minutes')
    target = 'Listening_Time_minutes'
    n_cols = 3
    n_rows = (len(num_col) + n_cols) // 3
    plt.figure(figsize=(18,10))
    
    for i, feature in enumerate(num_col):
        plt.subplot(n_rows, n_cols, i+1)
        plt.scatter(data[feature], data[target])
        plt.xlabel(feature)
        plt.ylabel("Время прослушивания")
        plt.title(f"Зависимость {target} от {feature}")   


def search_multicollinear(df, num_col):
    """Функция вычисляющая Фактор инфляции дисперсии (VIF) и сигнализирует о наличии и силы мультиколлинеарности у независимых фич"""
    df = df[num_col].drop(labels=['Listening_Time_minutes'], axis=1) # выбираем кол-ые признаки, удаляем таргет
    df = df.fillna(df.mean()) # обрабатываем пропуски в данных
    X_with_const = add_constant(df) #добавляем константу в датафрейм т.к. VIF не работает с пропусками
    X_features = X_with_const.columns  #определяем названия колонок фичей
    vif_df = pd.DataFrame() # формируем пустой датафрейм для организации расчетных  итоговых цифр 
    vif_df['feature'] = X_features # заполняем датафрейм названиями независимых фич
    vif_df['VIF'] = [variance_inflation_factor(X_with_const.values,i)  for i in range(X_with_const.shape[1])] #посредством list comprehension записываем итоговые VIF коэффициенты в датафрейм 
    #валидация итоговых результатов
    print('=================================================================')
    if vif_df.iloc[1:,:].max()[1] > 10:
        print("Присутствует сильная мультиколлинеарность")
    elif vif_df.iloc[1:,:].max()[1] < 10 and vif_df.iloc[1:,:].max()[1] > 5:
        print("Присутствует умеренная мультиколлинеарность")
    else:
        print("Мультиколлинеарность незначительная или отсутствует вовсе ")
    print('=================================================================')

    return vif_df # возвращаем VIF коэффициенты по каждой фиче


def categorial_feat_boxplots(data, cat_feat, target):
    plt.figure(figsize=(40,30))
    num_col = 3
    n_rows = len(cat_feat) // 3 + 1
    for ind, feature in enumerate(cat_feat):
        plt.subplot(n_rows, num_col, ind +1)
        sns.boxplot(data, x=feature, y='Listening_Time_minutes')
        plt.xlabel(feature)
        plt.xticks(rotation=45)
        plt.title(feature)
    plt.show()


def make_pipeline_preprocessor_and_model(num_col, cat_col, model_class, target=None, params=None, scaler=False, only_num_feat=False, only_cat_feat=False):
    num_col = num_col.to_list() #формируем 
    num_col.remove('Listening_Time_minutes') #удаляем таргет
    if scaler:
        num_pipe = Pipeline([('imputer', SimpleImputer(strategy='mean')),
                             ('scaler', StandardScaler())])
    else:
        num_pipe = Pipeline([('imputer', SimpleImputer(strategy='mean'))])
    if only_num_feat:
        model = Pipeline([('preproc', num_pipe),
                      ('model', model_class(**params))])
        return model
    cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                         ('ohe', OneHotEncoder(drop='first',handle_unknown='ignore', sparse_output=False))])
    preprocessor = ColumnTransformer([('num', num_pipe, num_col),
                                      ('cat', cat_pipe, cat_col)])
    if params is not None:
        model = Pipeline([('preprocessor', preprocessor),
                    #   ('feat_select'), Lasso() # feat_selector добавить 
                      ('model', model_class(**params)),
                      ])
    else:
        model = Pipeline([('preprocessor', preprocessor),
                      ('model', model_class())])
    if target is not None:
        imputer = SimpleImputer(strategy='mean')
        if isinstance(target, pd.Series):
            target = imputer.fit_transform(target.to_frame())
        else:
            target = imputer.fit_transform(target) #.values.reshape(-1,1)
        return model, imputer.fit_transform(target)
    return model


def looking_into_numeric_features(data, num_col):
    n_cols = 3
    n_rows = (len(num_col) + n_cols) // 3
    plt.figure(figsize=(18,10))
    
    for i, feature in enumerate(num_col):
        plt.subplot(n_rows, n_cols, i+1)
        sns.kdeplot(data, x=feature)
        plt.xlabel(feature)
        plt.ylabel("Плотность")
        plt.title("Распределение основны кол-ых фичей")    

#функция по добавлению результатов качества моделей
def append_model_results(data, title, y_test, y_train, y_pred_train, y_pred):
    data.loc[len(data)] = [title ,r2_score(y_train,y_pred_train), mean_squared_error(y_train,y_pred_train), 
                           root_mean_squared_error(y_train,y_pred_train),r2_score(y_test,y_pred), 
                           mean_squared_error(y_test,y_pred), root_mean_squared_error(y_test,y_pred)]