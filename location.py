import argparse
import pandas as pd
import numpy as np
from scipy import stats
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
    parser.add_argument('--train-file', type=str, dest='train_file', default='/Users/lli51/Downloads/trainingData.csv',
                        help='train file csv')
    parser.add_argument('--input-folder',type=str,  dest='input_folder', default="/Users/lli51/Desktop/",
                        help='Include this for the output path')
    parser.add_argument('--load-pre-process',type=int,  dest='load_preprocess', default=False,
                        help='Include this flag to load a previously pre-process data  located in the output path')
    parser.add_argument('--load-model',type=str,  dest='load_model', default= None,  #"/Users/lli51/Desktop/processed/best_estimator_so_far_latlon.pkl" , #
                        help='Include this flag to load a previously model located in the output path')
    parser.add_argument('--load-model_bf',type=str,  dest='load_model_bf', default= None, #"/Users/lli51/Desktop/processed/best_estimator_so_far_bf.pkl"
                        help='Include this flag to load a previously model located in the output path')
    parser.add_argument('--parameters',type=str,  dest='parameters', default="/Users/lli51/Desktop/parameters.pkl",
                        help='Include this parameters for the output path')
    
    return parser.parse_args()

def Obtain_Nan_Rows_Or_Columns_from_train(X, y=None):
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
    cols = X.count()!=0
    rows = X.count(axis=1) != 0

    X = (X.loc[:,cols]) # delete NaN columns
    X = X.loc[rows, :] # delete NaN rows
    if y is not None:
        y = y.loc[rows, :] # delete NaN rows
    return X, y, [cols, rows]

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

def lambda_optimal(x):
    _, k = stats.boxcox(x)
    return k
    
def find_skewness_parameters_in_train(X_train):
    '''
    Inputs:
    X_train                : train variables [always needs to be input]

    Outputs:
    min_value     : standardized_X_train, params
    '''
    X_train = np.power(10, X_train/10) # Prepair for the skewness analysis
    min_value = (X_train.apply(min).min())
    lambda_optimal_for_train = X_train.apply(lambda_optimal)
    for col in X_train:
        X_train.loc[:, col] = stats.boxcox(X_train.loc[:, col],lmbda= lambda_optimal_for_train.loc[col])
    
    scaler = StandardScaler().fit(X_train) # before PCA
    standardized_X_train = scaler.transform(X_train)
    return standardized_X_train, [min_value, lambda_optimal_for_train, scaler]

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

if not options.load_preprocess:
    trainframe = pd.read_csv(options.train_file)
    X, y = trainframe.loc[:, 'WAP001':'WAP520'].replace(to_replace=100, value=np.nan), trainframe[['LONGITUDE', 'LATITUDE', 'FLOOR', 'BUILDINGID']]
    X_train, X_test, y_train, y_test = train_test_split(X,  y, test_size=0.1, random_state=0) # split the train and the test files
    
    '''################################ 1. Preprocess :fix the datasets (NaN)  ################################'''
    X_train, y_train, del_zeros_col_row = Obtain_Nan_Rows_Or_Columns_from_train(X_train, y_train)
    X_test, y_test = Delete_Nan_Rows_Or_Columns_in_test_or_valid(del_zeros_col_row, X_test, y_test)

    print('After delete NANs:', X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    '''################################ 1. Preprocess :fix the datasets (Skewness)  ################################'''
    X_train.fillna(value=100, inplace=True)
    X_test.fillna(value=100, inplace=True)
    X_train, params_skeness = find_skewness_parameters_in_train(X_train)
    X_test = fix_skewness_in_test_with_boxcox(X_test, params_skeness)
    
    print('After delete boxcox:', X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    
    '''################################ 1. Preprocess :fix the datasets (remove zeros columns after boxcox)  ################'''
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    
    non_zeros_columns = list((X_train == 0).all()==False)
    X_train = X_train.loc[:,[i for i, n in enumerate(non_zeros_columns) if n]]
    X_test = X_test.loc[:,[i for i, n in enumerate(non_zeros_columns) if n]]
        
    print('After delete zeros columns:', X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    '''####################### 1. Preprocess :reduce the dimensions (PCA)  [LLE/ISOMAP/SpectrumEmbeding]#################'''
    Ndim_reduce = 200
    ###PCA##
    pca = PCA(n_components=Ndim_reduce)     ####ISOMAP#PCA(Ndim_reduce)
    pca.fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    ####ISOMAP
#     ISOMAP = Isomap(n_components=Ndim_reduce)
#     ISOMAP.fit(X_train)
#     X_train_pca = ISOMAP.transform(X_train)
#     X_test_pca = ISOMAP.transform(X_test)
#     
    print('After delete PCA:', X_train_pca.shape, y_train.shape, X_test_pca.shape, y_test.shape)
    
    
    '''################################  2. Learn the datasets  ################################'''
    
    train_pca, test_pca = np.array(X_train_pca), np.array(X_test_pca)
#     train_no_pre, test_no_pre, valid_no_pre = np.array(X_train_ready), np.array(X_test_ready), np.array(X_valid_ready)
    with open(options.input_folder+'train_pca.pickle', 'wb') as f:
        pickle.dump([train_pca, test_pca], f)
    with open(options.input_folder+'target.pickle', 'wb') as f:
        pickle.dump([y_train, y_test], f)
else:
    with open(options.input_folder+'train_pca.pickle', 'rb') as f:
        [train_pca, test_pca] = pickle.load(f)
    with open(options.input_folder+'target.pickle', 'rb') as f:
        [y_train, y_test] = pickle.load(f)
    
'''################################  2.1 Learn the lon, lat  ################################'''
    ## Proceed with only lat and long first?
y_train_latlon = y_train[['LONGITUDE', 'LATITUDE']]
y_test_latlon = y_test[['LONGITUDE', 'LATITUDE']]
y_train_bf = y_train[['BUILDINGID', 'FLOOR']]
y_test_bf = y_test[['BUILDINGID', 'FLOOR']]

BUILDINGIDs = y_train_bf.BUILDINGID.unique()
FLOORIDs = y_train_bf.FLOOR.unique()

'''################################  2.2.0 Build new label for the BuildID and Floor to learn them together ###############'''
new_label = np.arange(len(BUILDINGIDs)*len(FLOORIDs)).reshape(len(BUILDINGIDs),len(FLOORIDs))
new_label_dictionary = {(i,j): new_label[i][j] for i in BUILDINGIDs for j in FLOORIDs}
reverse_label_dictionary = {val:key for key, val in new_label_dictionary}

with open(options.parameters, 'wb') as f:
    pickle.dump([del_zeros_col_row, params_skeness, non_zeros_columns, pca, Ndim_reduce, new_label_dictionary], f)
    
def label_new(row):
    '''
    Inputs:
    row        : row of the pandas

    Outputs:
    row     : updated new labels
    '''
    return new_label_dictionary[(row['BUILDINGID'], row['FLOOR'])]

y_train_bf['combine_build_floor'] = y_train_bf.apply (lambda row: label_new(row), axis=1)
y_test_bf['combine_build_floor'] = y_test_bf.apply (lambda row: label_new(row), axis=1)
    
if not options.load_model:  
    # print(train_pca.shape, y_train_latlon.shape, test_pca.shape, y_test_latlon.shape, valid_pca.shape,  y_valid_latlon.shape )
    estimator_scores = {}
    estimator_dict = {} 
    estimator_para_dict = {}
    cache_models = {}
    
    def check_all_model(estimator_dict, estimator_para_dict, X_train, y_train, X_test, y_test, estimator_scores, cache_models):
        '''
        Inputs:
        estimator_dict        : List of estimators
        estimator_para_dict    : List of estimators parameters
        X_train                : train variables [always needs to be input]
        y_train                : target variable array
        X_test                : test variables
        y_test                : target variable array
        estimator_scores     : Dictionary to store the estimator's score
        cache_models  : store the best estimator
    
        Outputs:
        model_scores     : Updated dictionary of nested cross-validation scores
        '''
        for estimator_name in estimator_dict.keys():
            print('Start working on', estimator_name)
            start = time.time()
            estimator = estimator_dict[estimator_name]
            parameters = estimator_para_dict[estimator_name]
            model = GridSearchCV(estimator, parameters, cv=10, scoring='neg_mean_squared_error') # metrics.mean_squared_error
            model.fit(X_train, y_train)
            print('Time Consuming for' + estimator_name, time.time()-start)
            score = model.score(X_test, y_test)
    #         score = cross_val_score(model, X_train, y_train, cv=10, scoring='neg_mean_squared_error')
            scores = np.sqrt(np.abs(score))
            cache_models[estimator_name] = model #model.best_estimator_
    #         print(model.best_estimator_)
            estimator_scores[estimator_name] = [scores.mean(), scores.std()]
            
        return estimator_scores, cache_models
    '''# 2.1.1 Linear Regression'''
    #(1) Pipe Ridge
    pipe_ridge = Pipeline([('scale', StandardScaler()), ('regression', linear_model.Ridge(random_state=1))])
    pipe_ridge_para= {'regression__alpha':[0.01, 0.1, 1, 10]}
    estimator_dict['pipe_ridge'] = pipe_ridge
    estimator_para_dict['pipe_ridge'] = pipe_ridge_para
    #(2) Pipe lasso
    pipe_lasso = Pipeline([('scale', StandardScaler()), ('regression', linear_model.Lasso(random_state=1))])
    pipe_lasso_para= {'regression__alpha':[0.01, 0.1, 1, 10]}
    estimator_dict['pipe_lasso'] = pipe_lasso
    estimator_para_dict['pipe_lasso'] = pipe_lasso_para
    #(3) Pipe_ElasticNet
    pipe_ElasticNet = Pipeline([('scale', StandardScaler()), ('regression', linear_model.ElasticNet(random_state=1))])
    pipe_ElasticNet_para= {'regression__alpha':[0.01, 0.1, 1, 10]}
    estimator_dict['pipe_ElasticNet'] = pipe_ElasticNet
    estimator_para_dict['pipe_ElasticNet'] = pipe_ElasticNet_para
    #(4) Pipe_Poly_Ridge # add more metrics but too slow!!!!!!
    # pipe_Poly_ridge = Pipeline([('Polynomial', PolynomialFeatures(degree=2, interaction_only=True)),('scale', StandardScaler()),('PCA', PCA(n_components=50)), ('regression', linear_model.Ridge(random_state=1))])
    # pipe_Poly_ridge_para= {'regression__alpha':[0.01, 0.1, 1, 10]}
    # estimator_dict['Poly_pipe_ridge'] = pipe_Poly_ridge
    # estimator_para_dict['Poly_pipe_ridge'] = pipe_Poly_ridge_para
    '''# 2.1.2 Non-Linear Regression'''
    #(5) Pipe_knn
#     pipe_knn = Pipeline([('scale', StandardScaler()), ('knn', KNeighborsRegressor())])
#     pipe_knn_para= {'knn__n_neighbors':[2,3,5,7], 'knn__weights':['distance'], 'knn__metric':['euclidean', 'manhattan'], 'knn__n_jobs':[-1]}
#     estimator_dict['pipe_knn'] = pipe_knn
#     estimator_para_dict['pipe_knn'] = pipe_knn_para
    # 
    # #(6) Pipe_RandomForests
    pipe_RandomForests = Pipeline([('scale', StandardScaler()), ('rf', RandomForestRegressor(random_state=1))])
    pipe_RandomForests_para= {'rf__n_estimators':[100], 'rf__max_features':['auto', 'sqrt', 'log2']}
    estimator_dict['pipe_RandomForests'] = pipe_RandomForests
    estimator_para_dict['pipe_RandomForests'] = pipe_RandomForests_para
    
    
    # #(7) Pipe_updated_KNN
    # pipe_knn = Pipeline([('scale', StandardScaler()), ('knn', KNeighborsRegressor())])
    # pipe_knn_para= {'knn__n_neighbors':[2,3,5,7], 'knn__weights':['uniform', 'distance'], 'knn__metric':['euclidean', 'minkowski','manhattan'], 'knn__n_jobs':[-1]}
    # estimator_dict['pipe_knn'] = pipe_knn
    # estimator_para_dict['pipe_knn'] = pipe_knn_para
    
    '''##### Check the performance of the models ##### '''
    estimator_scores, cache_models = check_all_model(estimator_dict, estimator_para_dict, train_pca, y_train_latlon, test_pca, y_test_latlon, estimator_scores, cache_models)
    print(estimator_scores)
    
    best_estimator_so_far = cache_models[min([(val[0], key) for key,val in estimator_scores.items()])[1]]
        
    '''################################  2.2 Learn the BuildID and Floor ################################'''
    

    # print(BUILDINGIDs, FLOORIDs)
    def check_all_model_bf(estimator_dict, estimator_para_dict, X_train, y_train, X_test, y_test, estimator_scores, cache_models):
        '''
        Inputs:
        estimator_dict        : List of estimators
        estimator_para_dict    : List of estimators parameters
        X_train                : train variables [always needs to be input]
        y_train                : target variable array
        X_test                : test variables
        y_test                : target variable array
        estimator_scores     : Dictionary to store the estimator's score
        cache_models  : store the best estimator
    
        Outputs:
        model_scores     : Updated dictionary of nested cross-validation scores
        '''
        for estimator_name in estimator_dict.keys():
            print('Start working on', estimator_name)
            start = time.time()
            estimator = estimator_dict[estimator_name]
            parameters = estimator_para_dict[estimator_name]
            model = GridSearchCV(estimator, parameters, cv=10, scoring='accuracy') # metrics.mean_squared_error
            model.fit(X_train, y_train)
            print('Time Consuming for' + estimator_name, time.time()-start)
            score = model.score(X_test, y_test)
    #         score = cross_val_score(model, X_train, y_train, cv=10, scoring='neg_mean_squared_error')
            scores = np.sqrt(np.abs(score))
            cache_models[estimator_name] = model #model.best_estimator_
    #         print(model.best_estimator_)
            estimator_scores[estimator_name] = [scores.mean(), scores.std()]
            
        return estimator_scores, cache_models
    
    
    '''################################  2.2.2 Start Learning by classification ###############'''
    estimator_scores_bf = {}
    estimator_dict_bf = {} 
    estimator_para_dict_bf = {}
    cache_models_bf = {}
    #(1) Pipe LDA
#     pipe_LDA = Pipeline([('scale', StandardScaler()), ('LDA', LinearDiscriminantAnalysis())]) #shrinkage='auto'
#     pipe_LDA_para= {}
#     estimator_dict_bf['pipe_LDA'] = pipe_LDA
#     estimator_para_dict_bf['pipe_LDA'] = pipe_LDA_para
#     
#     pipe_SGD = Pipeline([('scale', StandardScaler()), ('SGD', linear_model.SGDClassifier())]) #shrinkage='auto'
#     pipe_SGD_para= {}
#     estimator_dict_bf['pipe_SDG'] = pipe_SGD
#     estimator_para_dict_bf['pipe_SDG'] = pipe_SGD_para
    
    pipe_NaiveBayes = Pipeline([('scale', StandardScaler()), ('NaiveBayes', GaussianNB())]) 
    pipe_NaiveBayes_para= {}
    estimator_dict_bf['pipe_NaiveBayes'] = pipe_NaiveBayes
    estimator_para_dict_bf['pipe_NaiveBayes'] = pipe_NaiveBayes_para
    
#     pipe_knn = Pipeline([('scale', StandardScaler()), ('knn', KNeighborsClassifier())])
#     pipe_knn_para= {'knn__n_neighbors':[2,3,5], 'knn__weights':['distance'], 'knn__metric':['manhattan'], 'knn__n_jobs':[-1]}
#     estimator_dict_bf['pipe_knn'] = pipe_knn
#     estimator_para_dict_bf['pipe_knn'] = pipe_knn_para
    
    pipe_randomforest = Pipeline([('scale', StandardScaler()), ('rf', RandomForestClassifier())])
    pipe_randomforest_para= {'rf__n_estimators':[100], 'rf__max_features':['auto', 'sqrt', 'log2']}
    estimator_dict_bf['pipe_randomforest'] = pipe_randomforest
    estimator_para_dict_bf['pipe_randomforest'] = pipe_randomforest_para
    
    
    
    estimator_scores_bf, cache_models_bf = check_all_model_bf(estimator_dict_bf, estimator_para_dict_bf, train_pca, y_train_bf['combine_build_floor'], test_pca, y_test_bf['combine_build_floor'], estimator_scores_bf, cache_models_bf)
    print(estimator_scores_bf)
    best_estimator_so_far_bf = cache_models_bf[max([(val[0], key) for key,val in estimator_scores_bf.items()])[1]]
        
    with open(options.input_folder+"best_estimator_so_far_latlon.pkl", 'wb') as f:
        pickle.dump(best_estimator_so_far, f)
    with open(options.input_folder+"best_estimator_so_far_bf.pkl", 'wb') as f:
        pickle.dump(best_estimator_so_far_bf, f)
else:
    
    '''################################  3. Postprocess (Evaluate our results)  ################################'''
    with open(options.load_model, 'rb') as f:
        best_estimator_so_far = pickle.load(f)
    with open(options.load_model_bf, 'rb') as f:
        best_estimator_so_far_bf = pickle.load(f)
      
#     best_estimator_so_far = pickle.load(options.load_model)
#     best_estimator_so_far_bf = pickle.load(options.load_model_bf)
       
def mean_and_variance(x):
    y =np.sqrt(np.abs(x))
    return np.mean(y, axis=1), np.std(y, axis=1) 
   
print(best_estimator_so_far.best_estimator_)
print(best_estimator_so_far_bf.best_estimator_)
# training_size_abs, train_scores, test_scores = learning_curve(estimator=best_estimator_so_far.best_estimator_, X=train_pca, y=y_train_latlon, train_sizes=np.linspace(0.1, 1.0, 10), cv=10,n_jobs=1, scoring='neg_mean_squared_error')
# train_scores_mean, train_scores_std = mean_and_variance(train_scores)
# test_scores_mean, test_scores_std = mean_and_variance(test_scores)
# 
# training_size_abs_bf, train_scores_bf, test_scores_bf =  learning_curve(estimator=best_estimator_so_far_bf.best_estimator_, X=train_pca, y= y_train_bf['combine_build_floor'], train_sizes=np.linspace(0.1, 1.0, 10), cv=10,n_jobs=1, scoring='accuracy')
# train_scores_mean_bf, train_scores_std_bf = np.mean(train_scores_bf, axis=1), np.std(train_scores_bf, axis=1) 
# test_scores_mean_bf, test_scores_std_bf = np.mean(test_scores_bf, axis=1), np.std(test_scores_bf, axis=1)  
# 
# plt.figure(1)
# plt.plot(training_size_abs, train_scores_mean, label='Train')
# plt.fill_between(training_size_abs, train_scores_mean+train_scores_std, train_scores_mean-train_scores_std, alpha=0.3)
# plt.plot(training_size_abs, test_scores_mean, label='Test')
# plt.fill_between(training_size_abs, test_scores_mean+test_scores_std, test_scores_mean-test_scores_std, alpha=0.3)
# plt.xlabel('Training Samples')
# plt.ylabel('MSE')
# plt.legend()
# 
# plt.figure(2)
# plt.plot(training_size_abs_bf, train_scores_mean_bf, label='Train')
# plt.fill_between(training_size_abs_bf, train_scores_mean_bf + train_scores_std_bf, train_scores_mean_bf- train_scores_std_bf, alpha=0.3)
# plt.plot(training_size_abs, test_scores_mean_bf, label='Test')
# plt.fill_between(training_size_abs_bf, test_scores_mean_bf + test_scores_std_bf, test_scores_mean_bf-test_scores_std_bf, alpha=0.3)
# plt.xlabel('Training Samples')
# plt.ylabel('Accuracy')
# plt.legend()
    
print('Total time consuming:', time.time()-start)
plt.show()