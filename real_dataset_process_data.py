import numpy as np
import pandas as pd
import random
from sklearn.utils import shuffle


def real_dataset_process_data(type_real_dataset):
    
    if(type_real_dataset==1):
        g=open('../statistical_learning_Github/real_datasets/lungCancer_train.data',"r")
        h=open('../statistical_learning_Github/real_datasets/lungCancer_test.data',"r")
        dict_type = {'Mesothelioma\r\n':-1,'ADCA\r\n':1}


    if(type_real_dataset==2):
        g=open('../statistical_learning_Github/real_datasets/leukemia_train.data',"r")
        h=open('../statistical_learning_Github/real_datasets/leukemia_test.data',"r")
        dict_type = {'ALL\r\r\n':-1,'AML\r\r\n':1}


    #READ TRAIN AND TEST
    X0, y = pd.DataFrame(), []

    for i in [g,h]:
        for line in i:
            line,data_line=line.split(",")[::-1],[]
            y.append(dict_type[str(line[0])])

            for aux in line[1:len(line)]:
                data_line.append(float(aux))
            X0=pd.concat([X0,pd.DataFrame(data_line).T])
                
    N,P = X0.shape
    X0.index = range(N)
    X0.columns = range(P)
    y = np.array(y)


    #NORMALIZE
    for i in range(P):
        X0.iloc[:,i] = X0.iloc[:,i]-X0.iloc[:,i].mean()
        X0.iloc[:,i] = X0.iloc[:,i]/float(X0.iloc[:,i].std())
        

    #TRAIN AND TEST -> SAME PROPORTION OF TWO CLASS
    idx_plus = np.where(y == 1)[0]
    idx_minus = np.where(y == -1)[0]

    N_plus_train = len(idx_plus)/2
    N_minus_train = len(idx_minus)/2

    X_plus = X0.loc[idx_plus]
    X_minus = X0.loc[idx_minus]

    X_plus.index = range(len(X_plus))
    X_minus.index = range(len(X_minus))


    #Xtrain
    X_train = pd.concat([X_plus.iloc[:N_plus_train] , X_minus.iloc[:N_minus_train]]).values
    y_train = np.concatenate([np.ones(N_plus_train),-np.ones(N_minus_train)])
    X_train, y_train = shuffle(X_train, y_train)


    #Xtest
    X_test = pd.concat([X_plus.iloc[N_plus_train:] , X_minus.iloc[N_minus_train:]]).values
    y_test = np.concatenate([np.ones(len(idx_plus) - N_plus_train),-np.ones(len(idx_minus) - N_minus_train)])
    X_test, y_test = shuffle(X_test, y_test)


    return X_train, y_train, X_test, y_test

