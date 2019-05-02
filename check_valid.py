import argparse
import pandas as pd
import numpy as np
from scipy import stats
from sklearn import metrics
from sklearn import svm
from sklearn import linear_model
from sklearn.manifold import Isomap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer
import seaborn as sns
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVC, SVR
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pickle
import numpy as np
import pandas as pd
import datetime as dt
import time
import pickle
from sklearn.preprocessing import Imputer, StandardScaler
from itertools import cycle
import scipy
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import ExtraTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, classification_report, accuracy_score, mean_squared_error
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')

def parse():
    '''
    This function is given to call for input: especially locations of the files
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--valid-file', type=str, dest='valid_file', default='/Users/lli51/Downloads/validationData.csv',
                        help='valid file csv')
    parser.add_argument('--parameters',type=str,  dest='parameters', default="/Users/lli51/Desktop/parameters.pkl",
                        help='Include this parameters for the output path')
    parser.add_argument('--load-model',type=str,  dest='load_model', default= "/Users/lli51/Desktop/best_estimator_so_far_latlon.pkl", #
                        help='Include this flag to load a previously model located in the output path')
    parser.add_argument('--load-model_bf',type=str,  dest='load_model_bf', default= "/Users/lli51/Desktop/best_estimator_so_far_bf.pkl",
                        help='Include this flag to load a previously model located in the output path')
    
    return parser.parse_args()


def Delete_Nan_Rows_Or_Columns_in_test_or_valid(params, X_test, y_test = None):
    '''
    This function is given to delete all NaN columns to rawfully reduce the dimensions
    Inputs:
    X                : train variables [always needs to be input]
    y                : target train variables
    X_test                : test variables
    y_test                : target test variables

    Outputs:
    model_scores     : X_train, X_test, X_valid
    '''
    [cols, rows] = params
    X_test = (X_test.loc[:,cols])
    y_test = y_test

    return X_test, y_test


def fix_skewness_in_test_with_boxcox(X_test, params):
    '''
    Inputs:
    X_test                : test variables
    params                : [min_value, lambda_optimal_for_train, scaler]

    Outputs:
    model_scores     : standardized_X_train
    '''
    [min_value, lambda_optimal_for_train, scaler] = params
    X_test = np.power(10, X_test/10) # Prepair for the skewness analysis
    X_test.fillna(min_value, inplace=True)
    
    for col in X_test:
        X_test.loc[:, col] = stats.boxcox(X_test.loc[:, col],lmbda= lambda_optimal_for_train.loc[col])
    
    standardized_X_test = scaler.transform(X_test)

    return standardized_X_test

start = time.time()
options = parse()
with open(options.parameters, 'rb') as f:
    [del_zeros_col_row, params_skeness, non_zeros_columns, pca, Ndim_reduce, new_label_dictionary] = pickle.load(f)
with open(options.load_model, 'rb') as f:
    best_estimator_so_far = pickle.load(f)
with open(options.load_model_bf, 'rb') as f:
    best_estimator_so_far_bf = pickle.load(f)
        
testframe = pd.read_csv(options.valid_file)
X_valid, y_valid = testframe.loc[:, 'WAP001':'WAP520'].replace(to_replace=100, value=np.nan), testframe[['LONGITUDE', 'LATITUDE', 'FLOOR', 'BUILDINGID']]
X_valid, y_valid = Delete_Nan_Rows_Or_Columns_in_test_or_valid(del_zeros_col_row, X_valid, y_valid)
X_valid.fillna(value=100, inplace=True)
X_valid = fix_skewness_in_test_with_boxcox(X_valid, params_skeness)
X_valid = pd.DataFrame(X_valid)
X_valid= X_valid.loc[:,[i for i, n in enumerate(non_zeros_columns) if n]]
X_valid_pca = pca.transform(X_valid)#[:,:Ndim_reduce]
valid_pca = np.array(X_valid_pca)
## ground true
y_valid_latlon = y_valid[['LONGITUDE', 'LATITUDE']]
y_valid_bf = y_valid[['BUILDINGID', 'FLOOR']]

def label_new(row):
    '''
    Inputs:
    row        : row of the pandas

    Outputs:
    row     : updated new labels
    '''
    return new_label_dictionary[(row['BUILDINGID'], row['FLOOR'])]

y_valid_bf['combine_build_floor'] = y_valid_bf.apply (lambda row: label_new(row), axis=1)

reverse_label_dictionary = {val:key for key, val in new_label_dictionary.items()}



predict_latlon = best_estimator_so_far.predict(valid_pca)
predict_bf = best_estimator_so_far_bf.predict(valid_pca)
predict_b_and_f = np.vectorize(reverse_label_dictionary.get)(predict_bf)
y_predict = pd.DataFrame({'predict_BUILDINGID':predict_b_and_f[0],'predict_Floor':predict_b_and_f[1]})


predict_error_latlon = np.sqrt(mean_squared_error(y_valid_latlon, predict_latlon))
predict_error_bf = accuracy_score(y_valid_bf['combine_build_floor'], predict_bf)
predict_error_b = accuracy_score(y_valid_bf['BUILDINGID'], y_predict['predict_BUILDINGID'])
predict_error_f = accuracy_score(y_valid_bf['FLOOR'], y_predict['predict_Floor'])
predict_error_b_report = metrics.classification_report(y_valid_bf['BUILDINGID'], y_predict['predict_BUILDINGID'])
predict_error_f_report = metrics.classification_report(y_valid_bf['FLOOR'], y_predict['predict_Floor'])
print('MSE Error (lat, lon):', predict_error_latlon)
print('Accuracy BuildingID and Floor:', predict_error_bf)
print('Accuracy BuildingID:', predict_error_b)
print('Accuracy Floor:', predict_error_f)
print('Total Time consuming:', time.time()-start)

print(predict_error_b_report)
print(predict_error_f_report)




