import numpy as np
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn import svm

from l1_SVM_CV_test import *


def build_RFE_estimator(support, coefficients, P):

#Builds the estimator on the whole support    
    beta_RFE_SVM = np.zeros(P)
    
    aux = -1
    for idx in np.where(support==True)[0]:
        aux+=1
        beta_RFE_SVM[idx] = coefficients[aux]
    return beta_RFE_SVM




def SVM_RFE_CV_test(X_train, y_train, X_test, y_test, size, number_CV, C_list):

#SAME PARAMETERS THAN CV_l1_SVM
#SIZE: size of the dataset after reduction
#RETURN ESTIMATOR SINCE SCIKIT COMPUTES IT

    start = time.time()
    N_train, P = X_train.shape
    N_test = X_test.shape[0]
    
    
    beta_RFE_SVM_list = []
    support_list = []
    ranking_list = []
    validation_errors = []
    train_errors_all_dataset = []
    
        
    for C in C_list:
    
#---STEP 1 : REDUCED ACCORDING TO CORRELATION
        if(size<P):
            
            #We train a first RFE by removing half of the features at every iteration
            estimator = svm.LinearSVC(penalty='l2', loss= 'hinge', dual=True, C=C)
            selector = RFE(estimator, n_features_to_select=size, step=0.5)
            selector = selector.fit(X_train, y_train)

            
            #We compute the reduced data
            support_RFE_first_step = np.where(selector.support_==True)[0]
            X_train_reduced = []
            X_test_reduced = []

            for i in range(N_train):
                X_train_reduced.append(X_train[i,:][support_RFE_first_step])
            
            for i in range(N_test):
                X_test_reduced.append(X_test[i,:][support_RFE_first_step])

            X_train_reduced = np.array(X_train_reduced)
            X_test_reduced = np.array(X_test_reduced)
        
        else:
            X_train_reduced = np.array(X_train)
            X_test_reduced = np.array(X_test)
        
     
    
#---STEP 2 : ONE STEP SVM RFE TO BE MORE PRECISE AND SELECT THE GOOD NUMBER OF FEATURES

        P_RFE = X_train_reduced.shape[1]

        #Scikit classical SVM solved with liblinear
        estimator = svm.LinearSVC(penalty='l2', loss= 'hinge', dual=True, C=C)

        #Scikit uses Stratified CV
        selector_CV = RFECV(estimator, step=1, cv=number_CV)
        selector_CV = selector_CV.fit(X_train_reduced, y_train)


        #Scikit fits on the subset of columns to obtain an estimator for the whole test set
        support, coefficients = selector_CV.support_, selector_CV.estimator_.coef_[0]
        beta_RFE_SVM_reduced, b0  = build_RFE_estimator(support, coefficients, P_RFE), selector_CV.estimator_.intercept_[0]
        ranking_features_reduced = selector_CV.ranking_
        
        
        #We compute the train error for the estimator on the whole dataset
        X_beta_RFE_SVM = np.dot(X_train_reduced, beta_RFE_SVM_reduced) + b0*np.ones(N_train)
        train_error_all_dataset = classification_error(y_train, X_beta_RFE_SVM)

        
    
#---STEP 3 : COMPUTE THE ESTIMATORS ON THE WHOLE TRAINING SET AND STORE THE RESULTS
        
        if(size<P):
            beta_RFE_SVM_all_support = np.zeros(P)
            ranking_features_all_support = -np.ones(P)

            for i in range(len(support_RFE_first_step)):
                beta_RFE_SVM_all_support[support_RFE_first_step[i]] = beta_RFE_SVM_reduced[i]
                ranking_features_all_support[support_RFE_first_step[i]] = ranking_features_reduced[i]

        else:
            beta_RFE_SVM_all_support = beta_RFE_SVM_reduced
            ranking_features_all_support = ranking_features_reduced
        
        
        
        beta_RFE_SVM_list.append((beta_RFE_SVM_all_support,b0))
        support_list.append(np.where(beta_RFE_SVM_all_support!=0)[0])

        #We will keep the C associated to the highest validation error
        validation_errors.append(np.max(selector_CV.grid_scores_))
        train_errors_all_dataset.append(train_error_all_dataset)
        
        #We keep the rank of features for further analysis
        ranking_list.append(ranking_features_all_support)


        
#---STEP 4 : ESTIMATOR WITH SMALLER SIZE OF SUPPORT

    argmax = -1
    max_validation, min_size_support = -1, P
    min_train_error = 2

    for i in range(len(C_list)):
    
    #CHECK IF WE RAE AT THE OPTIMAL
        is_best_beta = False

        if(validation_errors[i] > max_validation):
            is_best_beta = True

        elif(validation_errors[i] == max_validation):
            
            if(train_errors_all_dataset[i]<min_train_error):
                is_best_beta = True

            elif(train_errors_all_dataset[i] == min_train_error):
                if(len(support_list[i])<min_size_support):
                    is_best_beta = True 
    
        #SUM UP
        if(is_best_beta==True):
            argmax = i
            max_validation = validation_errors[i]
            min_size_support = len(support_list[i])
            min_train_error = train_errors_all_dataset[i]


#---RESULTS TRAIN ERROR
    beta_RFE_SVM_opt, b0_opt = beta_RFE_SVM_list[argmax]
    C_optimal = C_list[argmax]
    ranking_features_opt = ranking_list[argmax]

    X_beta_RFE_SVM = np.dot(X_test,beta_RFE_SVM_opt) + b0_opt*np.ones(N_test)
    test_error = classification_error(y_test, X_beta_RFE_SVM)



    end=round(time.time()-start,2)

    return min_train_error, test_error, beta_RFE_SVM_opt, b0_opt, validation_errors, support_list, ranking_features_opt, C_optimal, end

