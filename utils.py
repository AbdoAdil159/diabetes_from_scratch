import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.impute import KNNImputer
import warnings
warnings.filterwarnings('ignore')


#############################################################
#                 Grab columns                              #
#############################################################

def grab_columns(dataframe: pd.DataFrame, num_cat_thresold=10, card_thresold=20):
    """ This function calssifies dataframe columns in categorical, numerical and cardinal

    Args:
        dataframe (pd.DataFrame): The data frame that the data is stored in
        num_cat_thresold (int, optional): The limit of unique values of numerical columns. Defaults to 10.
        card_thresold (int, optional): The limit of unique values of cardinal columns. Defaults to 20.
    
    Returns:
            cat_cols, num_cols, car_cols   (list): three lists, the first one is for categorical columns, 
            the second is for numerical columns, the third one is for cardinal columns. 
    
    """
   
    # getting categorical columns by the data type    
    cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ['category', 'object', 'bool']]    

    # get numerical but categorical columns by the number of unique values (if less than 10, so it is categorical)
    num_but_cat = [col for col in dataframe if str(dataframe[col].dtypes).startswith(('int', 'float')) and dataframe[col].nunique() < num_cat_thresold]

    # Numerical columns, they are either int or float, and they are not categorical
    num_cols = [col for col in dataframe.columns if col not in num_but_cat and str(dataframe[col].dtypes).startswith(('int', 'float'))]

    # Cardinal columns
    car_cols = [col for col in dataframe if str(dataframe[col].dtypes) in ['category', 'object'] and dataframe[col].nunique() > card_thresold]    
    
    # Excluding cardinal columns from categorical columns
    cat_cols = [col for col in cat_cols if not col in car_cols]    

    return cat_cols + num_but_cat, num_cols, car_cols

#############################################################
#                 Outliers                                  #
#############################################################

def outlier_threshold(dataframe: pd.DataFrame, col_name, q1 = 0.25, q3= 0.75):
    
    quartile_1 = dataframe[col_name].quantile(q1)
    quartile_3 = dataframe[col_name].quantile(q3)

    iqr = quartile_3 - quartile_1
    up_limit = quartile_3 + 1.5 * iqr
    low_limit = quartile_1 - 1.5 * iqr
    
    return low_limit, up_limit


def check_outliers(dataframe: pd.DataFrame, col_name, q1 = 0.25, q3= 0.75):
    
    low, up = outlier_threshold(dataframe, col_name, q1, q3)
    return dataframe[(dataframe[col_name] < low) | (dataframe[col_name] > up)].any(axis = None)

def get_outliners(dataframe: pd.DataFrame, col_name, q1 = 0.25, q3= 0.75)->pd.DataFrame:
    low, up = outlier_threshold(dataframe, col_name, q1, q3)
    return dataframe[(dataframe[col_name] < low) | (dataframe[col_name] > up)]

def print_outliers(dataframe: pd.DataFrame, col_name, index = False, q1 = 0.25, q3= 0.75):    
    
    if check_outliers(dataframe, col_name, q1, q3):
        print(f'Nummber of Outliers in {col_name} is {get_outliners(dataframe, col_name).shape[0]} \n')
        print(get_outliners(dataframe, col_name).head())
    
    if index:
        return  get_outliners(dataframe, col_name).index 
    
def replace_with_thresholds(dataframe: pd.DataFrame, col_name, q1 = 0.25, q3= 0.75):
    
    low, up = outlier_threshold(dataframe, col_name, q1=q1, q3=q3)
    
    dataframe.loc[dataframe[col_name] < low, col_name] = low
    dataframe.loc[dataframe[col_name] > up, col_name] = up


#############################################################
#                 Encoding                                  #
#############################################################

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first, dtype='int')
    return dataframe

#############################################################
#                 Dataset preparation                       #
#############################################################

def diabetes_data_prep(dataframe):
    
    # Imputing for zero values
    imputer = KNNImputer(n_neighbors=5)
    cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in cols_with_zero:
        dataframe.loc[dataframe[col] == 0, col] = np.nan
    dataframe = pd.DataFrame(imputer.fit_transform(dataframe), columns=dataframe.columns)
    
    # capitalizing column names
    dataframe.columns = [col.upper() for col in dataframe.columns]

    # Extracting new features
    # Glucose
    dataframe['NEW_GLUCOSE_CAT'] = pd.cut(x=dataframe['GLUCOSE'], bins=[-1, 139, 200], labels=["normal", "prediabetes"])
    # Age
    dataframe.loc[(dataframe['AGE'] < 35), "NEW_AGE_CAT"] = 'young'
    dataframe.loc[(dataframe['AGE'] >= 35) & (dataframe['AGE'] <= 55), "NEW_AGE_CAT"] = 'middleage'
    dataframe.loc[(dataframe['AGE'] > 55), "NEW_AGE_CAT"] = 'old'
    # BMI
    dataframe['NEW_BMI_RANGE'] = pd.cut(x=dataframe['BMI'], bins=[-1, 18.5, 24.9, 29.9, 100],
                                        labels=["underweight", "healty", "overweight", "obese"])
    # BloodPressure
    dataframe['NEW_BLOODPRESSURE'] = pd.cut(x=dataframe['BLOODPRESSURE'], bins=[-1, 79, 89, 123],
                                            labels=["normal", "hs1", "hs2"])

    # Encoding
    cat_cols, num_cols, car_cols = grab_columns(dataframe, num_cat_thresold=5, card_thresold=20)
    cat_cols = [col for col in cat_cols if "OUTCOME" not in col]
    dataframe = one_hot_encoder(dataframe, cat_cols, drop_first=True)
    
    dataframe.columns = [col.upper() for col in dataframe.columns]

    # Dealing with outliers
    cat_cols, num_cols, car_cols = grab_columns(dataframe, num_cat_thresold=5, card_thresold=20)
    replace_with_thresholds(dataframe, "INSULIN")
    
    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(dataframe[num_cols])
    dataframe[num_cols] = pd.DataFrame(X_scaled, columns=dataframe[num_cols].columns)

    # X,y
    y = dataframe["OUTCOME"]
    X = dataframe.drop(["OUTCOME"], axis=1)

    return X, y


#############################################################
#                 Base models                               #
#############################################################


def base_models(X, y, scoring="roc_auc"):
    print("Base Models....")
    classifiers = [('LR', LogisticRegression()),
                   ('KNN', KNeighborsClassifier()),
                   ("SVC", SVC()),
                   ("CART", DecisionTreeClassifier()),
                   ("RF", RandomForestClassifier()),
                   ('Adaboost', AdaBoostClassifier()),
                   ('GBM', GradientBoostingClassifier()),
                   ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
                   ('LightGBM', LGBMClassifier(verbose=-1)),
                   # ('CatBoost', CatBoostClassifier(verbose=False))
                   ]

    for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv=3, scoring=scoring)
        print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name}) ")