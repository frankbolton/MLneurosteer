#imports

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
import neptune.new as neptune
    
# k-fold cross validation on combined data

def runModel(mean_value_subtraction, data_resampling, features_selected, standard_scaler, PCA_reduction, PCA_number_of_features, binary_classifier):
    run = neptune.init(project='frankbolton/Neurosteer-ML-v1', source_files=[__file__, 'environment.yaml'])
    # run = neptune.init(project='frankbolton/helloworld', source_files=[__file__, 'environment.yaml'])

    #preprocessing to select data to model
    data_params = { 'mean_value_subtraction': mean_value_subtraction,
                    'data_resampling': data_resampling, 
                    'features_selected': features_selected,
                    'standard_scaler': standard_scaler,
                    'PCA_reduction': PCA_reduction,
                    'PCA_number_of_features': PCA_number_of_features,
                    'binary_classifier':  binary_classifier
                    }

    # data_params = { 'mean_value_subtraction': False,
    #                 'data_resampling': 'average4', #options include 'last, 'average4', 'use4' and 'use7'
    #                 'features_selected': 'all', #options include 'all', '1070' and '4050'
    #                 'standard_scaler':False, #options include True and False
    #                 'PCA_reduction': False,
    #                 'PCA_number_of_features': 10,
    #                 'binary_classifier':  False
    #                 }

    run['parameters'] = data_params

    run["sys/tags"].add(['Naive Bayes', 'loop1'])

    #Data Preprocessing
    if (data_params['mean_value_subtraction']):
        data = pd.read_csv('preprocess_wide_2021_06_13_offsetZero.csv',index_col=0)
    else:
        data = pd.read_csv('preprocess_wide_2021_06_10.csv',index_col=0)

    #Data Resampling
    if (data_params['data_resampling']=='last'):
        #Drop all trials where there does not exist timepoints [0,1,2,3]
        for trial in data['uniqueTrialCounter'].unique():
            times = list(data.loc[data['uniqueTrialCounter'] == trial, 'timepoint'])
            if not((0 in times) and (1 in times) and (2 in times) and (3 in times)):
                data = data.drop(data[data['uniqueTrialCounter']==trial].index)
        # Drop any rows that are outside of the range 0s to 3s
        data = data.drop(data[data['timepoint']>3].index)
        data = data.drop(data[data['timepoint']<0].index)

    elif (data_params['data_resampling']=='average4'):
        #Drop all trials where there does not exist timepoints [0,1,2,3]
        for trial in data['uniqueTrialCounter'].unique():
            times = list(data.loc[data['uniqueTrialCounter'] == trial, 'timepoint'])
            if not((0 in times) and (1 in times) and (2 in times) and (3 in times)):
                data = data.drop(data[data['uniqueTrialCounter']==trial].index)
        # Drop any rows that are outside of the range 0s to 3s
        data = data.drop(data[data['timepoint']>3].index)
        data = data.drop(data[data['timepoint']<0].index)

    elif (data_params['data_resampling']=='use4'):
        #Drop all trials where there does not exist timepoints [0,1,2,3]
        for trial in data['uniqueTrialCounter'].unique():
            times = list(data.loc[data['uniqueTrialCounter'] == trial, 'timepoint'])
            if not((0 in times) and (1 in times) and (2 in times) and (3 in times)):
                data = data.drop(data[data['uniqueTrialCounter']==trial].index)
        # Drop any rows that are outside of the range 0s to 3s
        data = data.drop(data[data['timepoint']>3].index)
        data = data.drop(data[data['timepoint']<0].index)

    elif (data_params['data_resampling']=='use7'):
        #Drop all trials where there does not exist timepoints [-3,-2-1,0,1,2,3]
        for trial in data['uniqueTrialCounter'].unique():
            times = list(data.loc[data['uniqueTrialCounter'] == trial, 'timepoint'])
            if not((-3 in times) and (-2 in times) and (-1 in times) and \
                (0 in times) and (1 in times) and (2 in times) and (3 in times)):
                data = data.drop(data[data['uniqueTrialCounter']==trial].index)
        # Drop any rows that are outside of the range -3s to 3s
        data = data.drop(data[data['timepoint']>3].index)
        data = data.drop(data[data['timepoint']<-3].index)

    #Use binary data- drop label == 1
    if (data_params['binary_classifier']):
        data = data.drop(data[data['label']==1].index)
        run["sys/tags"].add(['binary'])

    
    #Data Feature Selection
    eeg_cols = []
    if (data_params['features_selected']=='1070'):
        eeg_cols += ["baf_"+str(x) for x in [*range(10,71)]]
        eeg_cols += ['VC_9', 'tf_16', 'tf_37']
    elif(data_params['features_selected']=='4050'):
        eeg_cols += ["baf_"+str(x) for x in [*range(40,51)]]
        eeg_cols += ['VC_9', 'tf_16', 'tf_37']
    else:
        eeg_cols += ["baf_"+str(x) for x in [*range(1,122)]]
        eeg_cols += ["tf_"+str(x) for x in [*range(1,104)]]
        eeg_cols += ["VC_"+str(x) for x in [*range(0,13)]]
        eeg_cols += ["A_"+str(x) for x in [*range(0,14)]]

    participants = data['participant'].unique()

    #Test out the accuracies with the different parameter settings
    # print(f"The data frame shape is {data.shape}")

    def generateXyLeaveOne(data, participant):
        data_test = data[data['participant'] == participant]
        data_train = data[data['participant'] != participant]
        # print(data_test.shape)
        # print(data_train.shape)
        X_train = list()
        X_test = list()
        y_train = list()
        y_test = list()
        
        if(data_params['data_resampling']=='use4'):
            for t in data_test['uniqueTrialCounter'].unique():
                y_test.append(data_test[data_test['uniqueTrialCounter']==t].label.values[0])
                X_test.append(data_test.loc[data_test['uniqueTrialCounter']==t, eeg_cols].transpose().values.flatten())
            for t in data_train['uniqueTrialCounter'].unique():
                y_train.append(data_train[data_train['uniqueTrialCounter']==t].label.values[0])
                X_train.append(data_train.loc[data_train['uniqueTrialCounter']==t, eeg_cols].transpose().values.flatten())

        elif(data_params['data_resampling']=='use7'):
            for t in data_test['uniqueTrialCounter'].unique():
                y_test.append(data_test[data_test['uniqueTrialCounter']==t].label.values[0])
                X_test.append(data_test.loc[data_test['uniqueTrialCounter']==t, eeg_cols].transpose().values.flatten())
            for t in data_train['uniqueTrialCounter'].unique():
                y_train.append(data_train[data_train['uniqueTrialCounter']==t].label.values[0])
                X_train.append(data_train.loc[data_train['uniqueTrialCounter']==t, eeg_cols].transpose().values.flatten())
        
        elif(data_params['data_resampling']=='average4'):
            for t in data_test['uniqueTrialCounter'].unique():
                y_test.append(data_test[data_test['uniqueTrialCounter']==t].label.values[0])
                X_test.append(data_test.loc[data_test['uniqueTrialCounter']==t, eeg_cols].mean())
            for t in data_train['uniqueTrialCounter'].unique():
                y_train.append(data_train[data_train['uniqueTrialCounter']==t].label.values[0])
                X_train.append(data_train.loc[data_train['uniqueTrialCounter']==t, eeg_cols].mean())
                
        elif (data_params['data_resampling']=='last'):
            for t in data_test['uniqueTrialCounter'].unique():
                y_test.append(data_test[data_test['uniqueTrialCounter']==t].label.values[0])
                X_test.append(data_test.loc[data_test['uniqueTrialCounter']==t, eeg_cols].values[-1])
            for t in data_train['uniqueTrialCounter'].unique():
                y_train.append(data_train[data_train['uniqueTrialCounter']==t].label.values[0])
                X_train.append(data_train.loc[data_train['uniqueTrialCounter']==t, eeg_cols].values[-1])
            
        return [np.asarray(X_train), np.asarray(X_test), np.asarray(y_train), np.asarray(y_test)]

    accuracies = list()
    train_acc_list = list()


    for p in participants:
        X_train, X_test, y_train, y_test= generateXyLeaveOne(data, p)

        if (data_params['standard_scaler']):
            scale = StandardScaler()
            scale.fit(X_train)
            X_train = scale.transform(X_train)
            X_test = scale.transform(X_test)

        if (data_params['PCA_reduction']):
            pca = PCA(n_components=data_params['PCA_number_of_features'])
            pca.fit(X_train)
            run['train/PCA_explained_variance_sum']= pca.explained_variance_ratio_.cumsum()
            X_train = pca.transform(X_train)
            X_test = pca.transform(X_test)
    
        model = GaussianNB()
        model.fit(X_train, y_train)  
        y_pred = model.predict(X_test)
        
        acc = ((y_test == y_pred).sum())/len(y_test)
        accuracies.append(acc)
        y_pred_train = model.predict(X_train)
        train_acc = ((y_train == y_pred_train).sum())/len(y_train)
        train_acc_list.append(train_acc)
        run['train/participant'].log(p)
        run['train/train_acc'].log(train_acc)
        run['test/acc'].log(acc)

    run['test/average_acc'] =  np.array(accuracies).mean()
    run['train/average_acc'] =  np.array(train_acc_list).mean()
    run.stop()
    return(np.array(accuracies).mean())