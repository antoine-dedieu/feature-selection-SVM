import numpy as np
import time

from Gurobi_l1_SVM import *
from simulate_data_classification import *


#CLASSIFICATION ERROR
def classification_error(y_train ,X_beta):
    return np.sum(y_train!=np.sign(X_beta))/float(len(X_beta))







#TRAIN ON THE WHOLE SET AFTER CV
def train_test_l1_SVM(X_train, X_test, y_train, y_test, C):
    
    N_train, N_test = X_train.shape[0], X_test.shape[0]
    time_limit = 100
    beta_l1_SVM, b0, _ , _  = l1_SVM(X_train, y_train, C, time_limit, 0)

    X_beta_l1_SVM = np.dot(X_train,beta_l1_SVM) + b0*np.ones(N_train)
    train_error = classification_error(y_train, X_beta_l1_SVM)

    X_beta_l1_SVM = np.dot(X_test,beta_l1_SVM) + b0*np.ones(N_test)
    test_error = classification_error(y_test, X_beta_l1_SVM)
    
    return train_error, test_error, beta_l1_SVM, b0






def cross_validation_l1_SVM(X_train, y_train, number_CV, C_list):

    start = time.time()
    time_limit = 20


    #---STEP 1: CREATE BALANCED TRAIN AND VALIDATION SETS FOR CV

    #Change type
    y_train = np.array(y_train)

    idx_plus = np.where(y_train == 1)[0]
    idx_minus = np.where(y_train == -1)[0]

    X_train_plus = X_train[idx_plus]
    X_train_minus = X_train[idx_minus]

    y_train_plus = y_train[idx_plus]
    y_train_minus = y_train[idx_minus]

    N_plus, P = X_train_plus.shape
    N_minus = X_train_minus.shape[0]


    #RESULTS
    X_train_Kfold, X_validation_Kfold = [], []
    y_train_Kfold, y_validation_Kfold = [], []



    for i in range(number_CV):

        #DEFINE THE K SAMPLES IN THE TWO CLASSES
        low_plus, high_plus = int(N_plus*(i/float(number_CV))), int(N_plus*((i+1)/float(number_CV)))
        low_minus, high_minus = int(N_minus*(i/float(number_CV))), int(N_minus*((i+1)/float(number_CV)))

        X_train_Kfold_i = np.concatenate([X_train_plus[:low_plus] , X_train_plus[high_plus:], X_train_minus[:low_minus], X_train_minus[high_minus:]])

        X_validation_Kfold_i = np.concatenate([X_train_plus[low_plus:high_plus], X_train_minus[low_minus:high_minus]])

        y_train_Kfold_i = np.concatenate([y_train_plus[:low_plus] , y_train_plus[high_plus:], y_train_minus[:low_minus], y_train_minus[high_minus:]])

        y_validation_Kfold_i = np.concatenate([y_train_plus[low_plus:high_plus], y_train_minus[low_minus:high_minus]])


        #SHUFFLE
        X_train_Kfold_i, y_train_Kfold_i = shuffle(X_train_Kfold_i, y_train_Kfold_i) 
        X_validation_Kfold_i, y_validation_Kfold_i = shuffle(X_validation_Kfold_i, y_validation_Kfold_i) 


        #STORE
        X_train_Kfold.append(X_train_Kfold_i)
        X_validation_Kfold.append(X_validation_Kfold_i)
        y_train_Kfold.append(y_train_Kfold_i)
        y_validation_Kfold.append(y_validation_Kfold_i)





    #---STEP 2: COMPUTE CV

    len_C_list = len(C_list)
    train_CV_errors = [[] for j in range(len_C_list)]
    validation_CV_errors = [[] for j in range(len_C_list)]

    support_list = [[] for j in range(len_C_list)]
    model_status_list = [[] for j in range(len_C_list)]
    times_list = [[] for j in range(len_C_list)]

    #We store one model for each C to train faster of the whole train set if the we have several minima
    Gurobi_model_list = [[] for j in range(len_C_list)]


    for i in range(number_CV):

        X_train_CV, X_validation_CV = X_train_Kfold[i], X_validation_Kfold[i]
        y_train_CV, y_validation_CV = y_train_Kfold[i], y_validation_Kfold[i]
        N_train_CV, N_validation_CV = X_train_CV.shape[0], X_validation_CV.shape[0]

        print N_train_CV, N_validation_CV 
        #Start with new model
        model =0

        for j in range(len_C_list): 
            start_time = time.time()
            beta_l1_SVM, b0, model_status, model  = l1_SVM(X_train_CV, y_train_CV, C_list[j], time_limit, model)

            X_beta_l1_SVM = np.dot(X_train_CV, beta_l1_SVM) + b0*np.ones(N_train_CV)
            train_CV_errors[j].append(classification_error(y_train_CV, X_beta_l1_SVM))

            X_beta_l1_SVM = np.dot(X_validation_CV, beta_l1_SVM) + b0*np.ones(N_validation_CV)
            validation_CV_errors[j].append(classification_error(y_validation_CV, X_beta_l1_SVM))

            support_list[j].append(np.where(beta_l1_SVM!=0)[0])
            times_list[j].append(round(time.time()-start_time,2))
            model_status_list[j].append(model_status)





    #---STEP 3: KEEP BEST BETA (if tie lower support)

    best_test_error_index = np.argmin(np.mean(validation_CV_errors,axis=1))

    end=round(time.time()-start,2)
    print 'Total time: ' +str(end)

    return best_test_error_index, np.round(validation_CV_errors,4), np.round(train_CV_errors,4), support_list, model_status_list, times_list, end






