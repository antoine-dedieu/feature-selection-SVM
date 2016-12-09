import numpy as np
import datetime

from real_dataset_process_data import *
from l1_SVM_CV_test import *
from SVM_RFE_CV_test import *



def write_and_print(text,f):
    print text
    f.write('\n'+text)





def compare_l1_RFE_real_dataset(type_real_dataset, size):

## TYPE_REAL_DATASET = 1 : LUNG CANCER DATASET
# TYPE_REAL_DATASET = 2 : LEUKEMIA

#---GET DATA

	DT = datetime.datetime.now()
	dict_title ={1:'lung_cancer',2:'leukemia'}
	name = str(DT.year)+'_'+str(DT.month)+'_'+str(DT.day)+'-'+str(DT.hour)+'h'+str(DT.minute)+'-'+str(dict_title[type_real_dataset])
	pathname=r'../Results/'+str(name)
	f = open(pathname+'_results.txt', 'w')
	X_train, y_train, X_test, y_test = real_dataset_process_data(type_real_dataset)
	P = X_train.shape[1]

	write_and_print('DATA CREATED', f)
	write_and_print('Train size : '+str(X_train.shape) + '    +1 train size : '+str((len(y_train) - np.sum(y_train))/2), f)
	write_and_print('Test size  : '+str(X_test.shape)  + '    +1 test size  : '+str((len(y_test) - np.sum(y_test))/2), f)


	#---L1 SVM
	write_and_print('\n\nL1 SVM' ,f)

	number_CV = 5
	C_list = [2**i for i in range(-10,10)]


	best_index, validation_errors_l1, train_CV_errors, support_list, model_status_list, times_list, time_l1 = cross_validation_l1_SVM(X_train, y_train, number_CV, C_list)
	C_optimal_l1 = np.array(C_list)[best_index]
	train_error_l1, test_error_L1, beta_l1_SVM, b0 = train_test_l1_SVM(X_train, X_test, y_train, y_test, C_optimal_l1)

	write_and_print('\nTrain error for l1       : '+str(train_error_l1), f)
	write_and_print('Test error for l1        : '+str(test_error_L1), f)
	write_and_print('Time for l1              : '+str(time_l1), f)
	write_and_print('C optimal                : '+str(C_optimal_l1), f)
	write_and_print('Validation error l1      : '+str(np.round(validation_errors_l1,3)), f)



	#---SVM RFE
	write_and_print('\n\nSVM RFE' ,f)

	#Result includes correlations for further analysis
	train_error_RFE, test_error_RFE, beta_SVM_RFE, b0_opt, validation_errors_RFE, support_list, ranking_features_RFE, C_optimal_RFE, time_RFE = SVM_RFE_CV_test(X_train, y_train, X_test, y_test, size, number_CV, C_list)

	write_and_print('\nTrain error for RFE       : '+str(train_error_RFE), f)
	write_and_print('Test error for RFE        : '+str(test_error_RFE), f)
	write_and_print('Time for RFE              : '+str(time_RFE), f)
	write_and_print('C optimal                 : '+str(C_optimal_RFE), f)
	write_and_print('Validation error RFE      : '+str(np.round(validation_errors_RFE,3)), f)



	#---L1 SVM + SVM RFE
	write_and_print('\n\nL1 SVM + SVM RFE' ,f)

	X_train_reduced = []
	X_test_reduced = []

	support_beta_l1 = np.where(beta_l1_SVM!=0)[0]
	for i in support_beta_l1:
	    X_train_reduced.append(X_train[:,i])
	    X_test_reduced.append(X_test[:,i])

	X_train_reduced = np.array(X_train_reduced).T
	X_test_reduced = np.array(X_test_reduced).T


	train_error_l1_RFE, test_error_l1_RFE, beta_l1_SVM_RFE_reduced, b0_opt, validation_errors_l1_RFE, support_list, ranking_features_l1_RFE, C_optimal_l1_RFE, time_l1_RFE = SVM_RFE_CV_test(X_train_reduced, y_train, X_test_reduced, y_test, size, number_CV, C_list)


	beta_l1_RFE_SVM = np.zeros(P)
	aux=-1
	for i in support_beta_l1:
	    aux+=1
	    beta_l1_RFE_SVM[i] = beta_l1_SVM_RFE_reduced[aux]


	write_and_print('\nTrain error for l1 + RFE        : '+str(train_error_l1_RFE), f)
	write_and_print('Test error for l1 + RFE        : '+str(test_error_l1_RFE), f)
	write_and_print('Time for l1 + RFE              : '+str(time_l1_RFE), f)
	write_and_print('C optimal                      : '+str(C_optimal_l1_RFE), f)
	write_and_print('Validation error RFE           : '+str(np.round(validation_errors_l1_RFE,3)), f)





#---COMPARE VARIABLE SELECTION
	write_and_print('\n\nSUPPORT ANALYSIS :' ,f)
	compare_support_l1_RFE_real_dataset(X_train, y_train, beta_l1_SVM, beta_SVM_RFE, beta_l1_RFE_SVM, ranking_features_RFE, f)
	f.close()









def compare_support_l1_RFE_real_dataset(X_train, y_train, beta_l1_SVM, beta_SVM_RFE, beta_l1_SVM_RFE, ranking_features, f):
    

    N,P = X_train.shape

#---Support and norm
    support_l1 = np.where(beta_l1_SVM)[0]
    support_RFE = np.where(beta_SVM_RFE)[0]
    support_l1_RFE = np.where(beta_l1_SVM_RFE)[0]

    mean_correl_l1, std_correl_l1, mean_correl_y_l1, std_correl_y_l1 = mean_std_abs_correlation(X_train, y_train, support_l1)
    mean_correl_RFE, std_correl_RFE, mean_correl_y_RFE, std_correl_y_RFE = mean_std_abs_correlation(X_train, y_train, support_RFE)
    mean_correl_l1_RFE, std_correl_l1_RFE, mean_correl_y_l1_RFE, std_correl_y_l1_RFE = mean_std_abs_correlation(X_train, y_train, support_l1_RFE)


    RFE_rank_l1_features = [int(ranking_features[i]) for i in support_l1]
    RFE_rank_l1_RFE_features = [int(ranking_features[i]) for i in support_l1]


    
#---Print differences
    write_and_print('\n\nLen support l1      : '+str(len(support_l1)) ,f)
    write_and_print('Support l1            : '+str(support_l1) ,f)
    write_and_print('RFE rank l1 features  : '+str(RFE_rank_l1_features) ,f)
    write_and_print('Mean-std correl l1    : '+str(mean_correl_l1)+' '+str(std_correl_l1) ,f)
    write_and_print('Mean-std correl /y l1 : '+str(mean_correl_y_l1)+' '+str(std_correl_y_l1) ,f)


    write_and_print('\n\nLen support l1      : '+str(len(support_RFE)) ,f)
    write_and_print('Support l1            : '+str(support_RFE) ,f)
    write_and_print('Mean-std correl RFE    : '+str(mean_correl_RFE)+' '+str(std_correl_RFE)  ,f)
    write_and_print('Mean-std correl /y RFE : '+str(mean_correl_y_RFE)+' '+str(std_correl_y_RFE) ,f)


    write_and_print('\n\nLen support l1      : '+str(len(support_l1_RFE)) ,f)
    write_and_print('Support l1            : '+str(support_l1_RFE) ,f)
    write_and_print('RFE rank l1 RFE features  : '+str(RFE_rank_l1_RFE_features) ,f)
    write_and_print('Mean-std correl l1 RFE    : '+str(mean_correl_l1_RFE)+' '+str(std_correl_l1_RFE) ,f)
    write_and_print('Mean-std correl /y l1 RFE : '+str(mean_correl_y_l1_RFE)+' '+str(std_correl_y_l1_RFE) ,f)

    




def mean_std_abs_correlation(X, y, support):

#RETURN THE CORRELATION BETWEEN FEATURES + WITH OUTPUT

    N, P = X.shape
    P_reduced = len(support)

    #reduced matrix
    X_train_reduced = []
    for i in range(N):
        X_train_reduced.append(X[i,:][support])
    
    #mat_correl is of size P_reduced*P_reduced
    mat_correl = np.corrcoef(np.array(X_train_reduced).T)
    abs_ouput_correl = np.abs(np.corrcoef(np.array(X_train_reduced).T, y))
    abs_correl = []
    
    for i in range(P_reduced):
        for j in range(i+1,P_reduced):
            abs_correl.append(abs(mat_correl[i,j]))
    
    return round(np.mean(abs_correl),3), round(np.std(abs_correl),3), np.mean(abs_ouput_correl), np.std(abs_ouput_correl)

