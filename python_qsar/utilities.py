import json
import csv
import inspect
import os
import re
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.linear_model import Lasso, Ridge, LinearRegression, ElasticNet
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import SelectKBest

from mordred import Calculator, descriptors

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


def smiles_dataset(dataset_df = None, smiles_loc = 'smiles', fp_radius = 3, fp_bits = 512):

    '''
    Use this function to generate the dataframe of fingerprint
    dataset_df: the input dataset should be a dataframe
    inchi_loc: the column name that consists of InChI strings
    fp_radius = the radius of Morgan fingerprint
    fp_bits = the number of fingerprint bits of Morgan fingerprint
    '''

    smiles = dataset_df[smiles_loc]
    smiles_list = np.array(smiles).tolist()

    mols = [Chem.MolFromSmiles(smile) for smile in smiles_list]
    mols = [mol for mol in mols if mol != None]
    # Hs have not been added in the KNIME preprocessing pipeline. They are not added here either, as it is best practice to not do so when working with fingerprints.
    # mols = [Chem.AddHs(smile) for smile in mols]

    mfpgen = rdFingerprintGenerator.GetMorganGenerator(includeChirality=False, radius=fp_radius, fpSize=fp_bits)
    morgans = [mfpgen.GetFingerprint(mol) for mol in mols]
    #AllChem.GetMorganFingerprintAsBitVect raised a deprecation warning. rdFingerprintGenerator.GetMorganGenerator should be used instead.
    #morgans = [AllChem.GetMorganFingerprintAsBitVect(mol, radius = fp_radius,
    #            nBits= fp_bits, useChirality = True) for mol in mols]
    morgan_bits =  [morgan.ToBitString() for morgan in morgans] 

    pattern = re.compile('.{1}')  # find every single digit
    morgan_bits = [','.join(pattern.findall(morgan)) for morgan in morgan_bits]

    fp_list = []
    for bit in morgan_bits:
        single_fp = bit.split(',')   # split the string by commas
        single_fp = [float(fp) for fp in single_fp] # transfer string to float32
        fp_list.append(single_fp)

    fp_df = pd.DataFrame(np.array(fp_list))
    fp_df.columns = fp_df.columns.astype(str)

    # rename the columns
    for i in range(fp_df.columns.shape[0]):
        fp_df.rename(columns = {fp_df.columns[i]:fp_df.columns[i] + "bit"}, inplace = True)

    return fp_df

def retrieve_df_name(dataframe):
    '''
    Use this function to retrieve the name of the dataframe
    '''
    for fi in reversed(inspect.stack()):
        df_names = [df_name for df_name, df_val in fi.frame.f_locals.items() if df_val is dataframe]
        if len(df_names) > 0:
            return df_names[0]

def save_dataset(dataframe, path = None, file_name = None, idx = False):
    '''
    Use this function to save the dataframe
    dataframe: DataFrame
    path: folder path in which the dataset stores, a string with ''
    file_name: please enter a string with ''
    idx: to control the creation of index or not
    '''
    if path is None:
        path = os.path.join(os.getcwd(), 'datasets')
    else:
        path = os.path.join(os.getcwd(), path)
    print('Current path is:', path)

    if os.path.exists(path) == True:
        pass
        print('Path already existed.')
    else:
        os.mkdir(path)
        print('Path created.')

    if file_name is None:

        dataframe.to_csv(path + '/' + retrieve_df_name(dataframe)+ '.csv', index = idx)
    else:
        dataframe.to_csv(path + '/' + file_name + '.csv', index = idx)

    print('Dataset saved successfully.')

def get_descriptors(df, smiles_loc, no_3D=True):
    calc = Calculator(descriptors, ignore_3D=no_3D)
    smiles = df[smiles_loc]
    smiles_list = np.array(smiles).tolist()
    mols = [Chem.MolFromSmiles(smile) for smile in smiles_list]
    df_descriptors = calc.pandas(mols)
    return df_descriptors

def drop_nonnumeric_columns(df=None):
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')
        df.dropna(axis=1, inplace=True)
    return df

def select_features(X_train, y_train, X_test, num_features='all'):
	# configure to select all features
	fs = SelectKBest(score_func=mutual_info_regression, k=num_features)
	# learn relationship from training data
	fs.fit(X_train, y_train)
	# transform train input data
	X_train_fs = fs.transform(X_train)
	# transform test input data
	X_test_fs = fs.transform(X_test)
	return X_train_fs, X_test_fs, fs

def get_parameters(path = None, print_dict = False):

    if path is None:
        raise ValueError('No path entered.')
    else:
        path = path
    print('json file path is:', path)

    f = open(path, 'r')
    line = f.read()
    dic = json.loads(line)

    if print_dict == False:
        pass
    elif print_dict == True:
        print(dic)

    return dic

def hp_opt(x_fp_df, x_desc_df, y_df, random_seeds = 42, mode = 'rf', scoring = None):

    '''
    x_df: the dataframe of the dataset, training set + cross validation set
    y_df: the dataframe of the target, training set + cross validation set

    scoring: 'None' to use r^2, 'mse_func' to use MSE for performance evaluation
    '''

    start = time.time()
    np.random.seed(random_seeds)

    # dataset x and y
    if mode in ['knn','ridge','lasso','elastic','dt']:
    #if mode == 'knn' or mode =='ridge' or mode == ...

        x_data = x_fp_df.values
        x_desc = x_desc_df.values
        y_data = y_df.values.ravel()

    elif mode in ['gradientboosting','adaboost','extratrees','rf', 'dt' ,'svr']:

        x_data = x_fp_df.values
        x_desc = x_desc_df.values
        y_data = y_df.values
        y_data = y_data.flatten()


    else:
        raise ValueError('Mode not found')


    dataframes = []
    best_estimators = []
    estimators = []
    param_grid = {}
    
    scaler = MinMaxScaler()
    x_desc = scaler.fit_transform(x_desc)
    
    x_join = np.concatenate((x_data, x_desc), axis=1)

    if mode == 'knn':
        regr = KNeighborsRegressor()
        param_grid = {'n_neighbors': [2,3,4,5,6,7,8,9,10],
                      'weights':['uniform', 'distance'],
                      'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                      'leaf_size': [20, 30, 40],
                      'p': [1,2]}
    if mode == 'ridge':
        regr = Ridge()
        param_grid = {'alpha': [0.001,0.01,0.1,1,10],
                       'solver': ['auto'],
                       'max_iter': [100000]}
    if mode == 'lasso':
        regr = Lasso()
        param_grid = {'alpha': [0.001,0.01,0.1,1,10],
                      'selection':['cyclic', 'random'],
                      'max_iter': [100000]}
    if mode == 'elastic':
        regr = ElasticNet()
        param_grid = {'alpha':[0.001,0.01,0.1,1,10],
                      'l1_ratio': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                      'max_iter': [100000]}
    if mode == 'gradientboosting':
        regr = GradientBoostingRegressor()
        param_grid = {'learning_rate': [0.001,0.01,0.1,1],
                         'min_samples_split': [2,3,4,5,6,7,8,9],
                         'loss': ['squared_error', 'absolute_error', 'huber', 'quantile'],
                         'criterion':['squared_error', 'friedman_mse']}
    if mode == 'adaboost':
        regr = AdaBoostRegressor()
        param_grid = {'learning_rate': [0.001,0.01,0.1,1],
                         'loss': ['linear', 'square', 'exponential']}
    if mode == 'extratrees':
        regr = ExtraTreesRegressor()
        param_grid = {'bootstrap': [True, False],
                      'min_samples_split': [2,3,4,5,6,7,8,9]}
    if mode == 'rf':
            regr = RandomForestRegressor()
            param_grid = {'bootstrap': [True, False],
                         'max_features': [1.0,'log2','sqrt'],
                         'min_samples_split': [2,3,4,5,6,7,8,9]}
    if mode == 'dt':
        regr = DecisionTreeRegressor()
        param_grid = {'criterion': ['squared_error', 'friedman_mse'],
                      'min_samples_split': [2,3,4,5,6,7,8,9]}
    if mode == 'svr':
        regr = SVR()
        param_grid = {'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
                      'gamma': ['scale', 'auto'],
                      'C': [0.001, 0.01, 0.1, 1, 10, 20, 50, 100]}


    regr_grid = GridSearchCV(regr, param_grid, scoring = None)
    regr_grid.fit(x_join, y_data)

    best_estimator = regr_grid.best_estimator_


    print('The best hyperparameters:', best_estimator)
    print('Time for optimization: %f seconds' %(time.time()-start), flush=True)
    print('{} Folds Were Tested'.format(regr_grid.n_splits_))
    print('*********************************')

    return best_estimator

def fit_result(train_fp, train_desc, train_target, test_fp, test_desc, test_target, estimator):

    values_train_fp, values_train_desc, values_train_target = train_fp.values, train_desc.values, train_target.values.ravel()
    values_test_fp, values_test_desc, values_test_target = test_fp.values, test_desc.values, test_target.values.ravel()

    scaler = MinMaxScaler()
    values_train_desc = scaler.fit_transform(values_train_desc)
    values_test_desc = scaler.transform(values_test_desc)

    values_train_join = np.concatenate((values_train_fp, values_train_desc), axis=1)
    values_test_join = np.concatenate((values_test_fp, values_test_desc), axis=1)
    
    regr = estimator
    regr.fit(values_train_join, values_train_target)
    train_pred = regr.predict(values_train_join)
    test_pred = regr.predict(values_test_join)


    train_score = regr.score(values_train_join, values_train_target)
    test_score = regr.score(values_test_join, values_test_target)

    train_mse = sum((x - y)*(x - y) for x, y in zip(train_pred, values_train_target))/ len(values_train_join)
    train_rmse = np.sqrt(train_mse)
    test_mse = sum((x - y)*(x - y) for x, y in zip(test_pred, values_test_target))/ len(values_test_join)
    test_rmse = np.sqrt(test_mse)

    result_dict = {'train_score':train_score,
                   'test_score':test_score,
                   'train_rmse':train_rmse,
                   'test_rmse':test_rmse}
        
    print('Model:', estimator)
    print('The train score:{:.3f}'.format(train_score))
    print('The test score:{:.3f}'.format(test_score))
    print('The train rmse:{:.3f}'.format(train_rmse))
    print('The test rmse:{:.3f}'.format(test_rmse))


    return result_dict

def extract_data_for_plot(y, interval = None):
    '''
    work with utils.plot_multi_learning_curves
    '''
    if isinstance(interval, int):

        index1 = list(np.arange(interval, len(y), interval))
        last = [len(y) - 1]
        index1.insert(0, 0)
        index1.extend(last)
        index2 = list(set(index1))
        index2.sort(key = index1.index)
        y = np.array(y)
        y_plot = list(y[index2])
        x_plot = [i + 1 for i in index2]

    elif interval is None:

        index2 = list(np.arange(0, len(y)))
        x_plot = [i + 1 for i in index2]
        y_plot = y

    else:
        raise ValueError('interval must be an integer or None')

    return x_plot, y_plot

def plot_multi_learning_curves(x_fp_df, x_desc_df, y_df, estimator1, estimator2, estimator3, random_seed = 42, testsize = 0.2, mode = 'r2', autosave = 'n', interval = 1, path = None, file_name = 'file_name'):

    x_fp = x_fp_df.values
    x_desc = x_desc_df.values
    y_data = y_df.values.ravel()
    
    x_train, x_test, y_train, y_test = train_test_split(x_fp, y_data, test_size = testsize, random_state = random_seed)
    x_train_desc, x_test_desc = train_test_split(x_desc, test_size = testsize, random_state = random_seed)

    scaler = MinMaxScaler()
    x_train_desc = scaler.fit_transform(x_train_desc)
    x_test_desc = scaler.transform(x_test_desc)

    x_train_join = np.concatenate((x_train, x_train_desc), axis=1)
    x_test_join = np.concatenate((x_test, x_test_desc), axis=1)

    train_errors1 = []
    test_errors1 = []
    train_errors2 = []
    test_errors2 = []
    train_errors3 = []
    test_errors3 = []
    range_start = 1
    estimator_str_list = [estimator1.__class__.__name__, estimator2.__class__.__name__, estimator3.__class__.__name__]
    estimator_list = [estimator1, estimator2, estimator3]

    if 'KNeighborsRegressor' in estimator_str_list:
        if getattr(estimator_list[estimator_str_list.index('KNeighborsRegressor')], 'n_neighbors') > 1:
            range_start = getattr(estimator_list[estimator_str_list.index('KNeighborsRegressor')], 'n_neighbors')
        elif mode == 'rmse':
            range_start = 1
        elif mode == 'r2':
            range_start = 2
    elif mode == 'rmse':
        range_start = 1
    elif mode == 'r2':
        range_start = 2

    if mode == 'rmse':   
        
        for m in range(range_start, len(x_train_join)):
    
            estimator1.fit(x_train_join[: m], y_train[: m])
    
            train_pred = estimator1.predict(x_train_join[: m])
            test_pred = estimator1.predict(x_test_join)
    
            train_errors1.append(mean_squared_error(y_train[: m], train_pred))
            test_errors1.append(mean_squared_error(y_test, test_pred))
    
            estimator2.fit(x_train_join[: m], y_train[: m])
    
            train_pred = estimator2.predict(x_train_join[: m])
            test_pred = estimator2.predict(x_test_join)
    
            train_errors2.append(mean_squared_error(y_train[: m], train_pred))
            test_errors2.append(mean_squared_error(y_test, test_pred))
    
            estimator3.fit(x_train_join[: m], y_train[: m])
    
            train_pred = estimator3.predict(x_train_join[: m])
            test_pred = estimator3.predict(x_test_join)
    
            train_errors3.append(mean_squared_error(y_train[: m], train_pred))
            test_errors3.append(mean_squared_error(y_test, test_pred))
    
        train_errors1_x, train_errors1 = extract_data_for_plot(train_errors1, interval = interval)
        test_errors1_x, test_errors1 = extract_data_for_plot(test_errors1, interval = interval)
        train_errors2_x, train_errors2 = extract_data_for_plot(train_errors2, interval = interval)
        test_errors2_x, test_errors2 = extract_data_for_plot(test_errors2, interval = interval)
        train_errors3_x, train_errors3 = extract_data_for_plot(train_errors3, interval = interval)
        test_errors3_x, test_errors3 = extract_data_for_plot(test_errors3, interval = interval)
    
    
        plt.figure(figsize=(16, 8))
        plt.xticks(size = 22)
        plt.yticks(size = 22)
        plt.xlabel('Number of Examples', fontproperties = 'Times New Roman', fontsize = 24)
        plt.ylabel('Root Mean Square Error (RMSE)', fontproperties = 'Times New Roman', fontsize = 24)
    
    
        plt.plot(train_errors1_x, train_errors1, color = 'blue',linestyle = '--', linewidth = 1, label = estimator1.__class__.__name__ + ' Train') #label = 'SVR Train R\u00b2'
        plt.plot(test_errors1_x, test_errors1, color = 'blue', linestyle = '-', linewidth = 1, label = estimator1.__class__.__name__ + ' Test')  # label = 'SVR Test R\u00b2'
        plt.plot(train_errors2_x, train_errors2, color = 'red', linestyle = '--', linewidth = 1, label = estimator2.__class__.__name__ + ' Train') # label = 'RF Train R\u00b2'
        plt.plot(test_errors2_x, test_errors2, color = 'red', linestyle = '-',linewidth = 1, label = estimator2.__class__.__name__ + ' Test') # label = 'RF Test R\u00b2'
        plt.plot(train_errors3_x, train_errors3, color = 'grey', linestyle = '--', linewidth = 1, label = estimator3.__class__.__name__ + ' Train') # label = 'DT Train R\u00b2'
        plt.plot(test_errors3_x, test_errors3, color = 'grey', linestyle = '-', linewidth = 1, label = estimator3.__class__.__name__ + ' Test') # label = 'DT Train R\u00b2'
        plt.legend()
    
    elif mode == 'r2':

        for m in range(range_start, len(x_train_join)):   # can not calculate the r^2 value below 2
            estimator1.fit(x_train_join[: m], y_train[: m])

            train_error1 = estimator1.score(x_train_join[: m], y_train[: m])
            test_error1 = estimator1.score(x_test_join, y_test)

            train_errors1.append(train_error1)
            test_errors1.append(test_error1)

            estimator2.fit(x_train_join[: m], y_train[: m])

            train_error2 = estimator2.score(x_train_join[: m], y_train[: m])
            test_error2 = estimator2.score(x_test_join, y_test)

            train_errors2.append(train_error2)
            test_errors2.append(test_error2)

            estimator3.fit(x_train_join[: m], y_train[: m])

            train_error3 = estimator3.score(x_train_join[: m], y_train[: m])
            test_error3 = estimator3.score(x_test_join, y_test)

            train_errors3.append(train_error3)
            test_errors3.append(test_error3)

        train_errors1_x, train_errors1 = extract_data_for_plot(train_errors1, interval = interval)
        test_errors1_x, test_errors1 = extract_data_for_plot(test_errors1, interval = interval)
        train_errors2_x, train_errors2 = extract_data_for_plot(train_errors2, interval = interval)
        test_errors2_x, test_errors2 = extract_data_for_plot(test_errors2, interval = interval)
        train_errors3_x, train_errors3 = extract_data_for_plot(train_errors3, interval = interval)
        test_errors3_x, test_errors3 = extract_data_for_plot(test_errors3, interval = interval)

        plt.figure(figsize=(16, 8))
        plt.xticks(size = 22)
        plt.yticks(size = 22)

        plt.xlabel('Number of Examples', fontproperties = 'Times New Roman', fontsize = 24)
        plt.ylabel('Coefficient of Determination (R\u00b2)', fontproperties = 'Times New Roman', fontsize = 24)

        plt.plot(train_errors1_x, train_errors1, color = 'blue',linestyle = '--', linewidth = 1, label = estimator1.__class__.__name__ + ' Train') #label = 'SVR Train R\u00b2'
        plt.plot(test_errors1_x, test_errors1, color = 'blue', linestyle = '-', linewidth = 1, label = estimator1.__class__.__name__ + ' Test')  # label = 'SVR Test R\u00b2'
        plt.plot(train_errors2_x, train_errors2, color = 'red', linestyle = '--', linewidth = 1, label = estimator2.__class__.__name__ + ' Train') # label = 'RF Train R\u00b2'
        plt.plot(test_errors2_x, test_errors2, color = 'red', linestyle = '-',linewidth = 1, label = estimator2.__class__.__name__ + ' Test') # label = 'RF Test R\u00b2'
        plt.plot(train_errors3_x, train_errors3, color = 'grey', linestyle = '--', linewidth = 1, label = estimator3.__class__.__name__ + ' Train') # label = 'DT Train R\u00b2'
        plt.plot(test_errors3_x, test_errors3, color = 'grey', linestyle = '-', linewidth = 1, label = estimator3.__class__.__name__ + ' Test') # label = 'DT Train R\u00b2'
        plt.legend(loc = 'lower right')
    else:
        raise ValueError('mode type either \'rmse\' or \'r2\'')
        
    # save figures
    if autosave == 'y':

        if path is None:
            path = os.path.join(os.getcwd(), 'figures')
        else:
            path = path
        print('Current path is:', path)

        if os.path.exists(path) == True:
            pass
            print('Path already existed.')
        else:
            os.mkdir(path)
            print('Path created.')

        plt.savefig(path + '/' + str(mode) + '_' + str(file_name) + '_compare_learning_curve.png')
        plt.show()
        print('Figure saved successfully.')

    elif autosave == 'n':
        pass
    else:
        raise ValueError('autosave rather \'n\' or \'y\'')
