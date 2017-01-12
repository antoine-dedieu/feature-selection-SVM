import numpy as np
import datetime

from simulate_data_classification import *
from l1_SVM_CV_test import *
from SVM_RFE_CV_test import *



def write_and_print(text,f):
    print text
    f.write('\n'+text)





def compare_l1_RFE(N,P,k0,rho,type_Sigma,size):

#---SIMULATE DATA
    DT = datetime.datetime.now()
    name = str(DT.year)+'_'+str(DT.month)+'_'+str(DT.day)+'-'+str(DT.hour)+'h'+str(DT.minute)+'-N'+str(N)+'_P'+str(P)+'_k0'+str(k0)+'_rho'+str(rho)+'_Sigma'+str(type_Sigma)
    pathname=r'../Results/'+str(name)
    f = open(pathname+'_results.txt', 'w')
    
    X_train, X_test, y_train, y_test, u_positive = simulate_data_classification(type_Sigma, N,P,k0,rho,f)

    write_and_print('+1 train size : '+str((len(y_train) - np.sum(y_train))/2), f)
    write_and_print('+1 test size  : '+str((len(y_test) - np.sum(y_test))/2), f)



#---L1 SVM
    write_and_print('\n\nL1 SVM' ,f)

    number_CV = 5
    C_list = [2**i for i in range(-10,0)]


    best_index, validation_errors_l1, train_CV_errors, support_list, model_status_list, times_list, time_l1 = cross_validation_l1_SVM(X_train, y_train, number_CV, C_list)
    C_optimal_l1 = np.array(C_list)[best_index]
    train_error_l1, test_error_L1, beta_l1_SVM, b0 = train_test_l1_SVM(X_train, X_test, y_train, y_test, C_optimal_l1)

    write_and_print('\nTrain error for l1 : '+str(train_error_l1), f)
    write_and_print('Test error for l1  : '+str(test_error_L1), f)
    write_and_print('Time for l1        : '+str(time_l1), f)
    write_and_print('C optimal          : '+str(C_optimal_l1), f)
    write_and_print('Validation error l1: '+str(np.round(validation_errors_l1,3)), f)



#---SVM RFE
    write_and_print('\n\nSVM RFE' ,f)

    #Result includes correlations for further analysis
    support_beta_l1 = np.where(beta_l1_SVM!=0)[0]
    train_error_RFE, test_error_RFE, beta_SVM_RFE, b0_opt, validation_errors_RFE, support_list, ranking_features_RFE, C_optimal_RFE, time_RFE = SVM_RFE_CV_test(X_train, y_train, X_test, y_test, len(support_beta_l1), number_CV, C_list)

    write_and_print('\nTrain error for RFE : '+str(train_error_RFE), f)
    write_and_print('Test error for RFE  : '+str(test_error_RFE), f)
    write_and_print('Time for RFE        : '+str(time_RFE), f)
    write_and_print('C optimal           : '+str(C_optimal_RFE), f)
    write_and_print('Validation error RFE: '+str(np.round(validation_errors_RFE,3)), f)



#---L1 SVM + SVM RFE
    write_and_print('\n\nL1 SVM + SVM RFE' ,f)

    X_train_reduced = []
    X_test_reduced = []

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


    write_and_print('\nTrain error for l1 + RFE : '+str(train_error_l1_RFE), f)
    write_and_print('Test error for l1 + RFE  : '+str(test_error_l1_RFE), f)
    write_and_print('Time for l1 + RFE        : '+str(time_l1_RFE), f)
    write_and_print('C optimal           : '+str(C_optimal_l1_RFE), f)
    write_and_print('Validation error RFE: '+str(np.round(validation_errors_l1_RFE,3)), f)



#---COMPARE VARIABLE SELECTION
    write_and_print('\n\nSUPPORT ANALYSIS :' ,f)
    compare_support_l1_RFE(X_train, y_train, beta_l1_SVM, beta_SVM_RFE, beta_l1_RFE_SVM, u_positive, ranking_features_RFE, f)
    f.close()









def compare_support_l1_RFE(X_train, y_train, beta_l1_SVM, beta_SVM_RFE, beta_l1_SVM_RFE, u_positive, ranking_features, f):
    

    N,P = X_train.shape

#---Support and norm
    support_l1 = np.where(beta_l1_SVM)[0]
    support_RFE = np.where(beta_SVM_RFE)[0]
    support_l1_RFE = np.where(beta_l1_SVM_RFE)[0]
    real_support = np.where(u_positive)[0]
    

    good_variables_l1 = list(set(real_support) & set(support_l1))
    good_variables_RFE = list(set(real_support) & set(support_RFE))
    good_variables_l1_RFE = list(set(real_support) & set(support_l1_RFE))
    

    variable_selection_error_l1 = list((set(real_support)-set(support_l1)) | (set(support_l1)-set(real_support)))
    variable_selection_error_RFE = list((set(real_support)-set(support_RFE)) | (set(support_RFE)-set(real_support))) 
    variable_selection_error_l1_RFE = list((set(real_support)-set(support_l1_RFE)) | (set(support_l1_RFE)-set(real_support))) 
    

    diff_norm_l1 = np.linalg.norm(beta_SVM_RFE-u_positive)
    diff_norm_RFE = np.linalg.norm(beta_l1_SVM-u_positive)
    diff_norm_l1_RFE = np.linalg.norm(beta_l1_SVM_RFE-u_positive)


    mean_correl_l1, std_correl_l1, mean_correl_y_l1, std_correl_y_l1 = mean_std_abs_correlation(X_train, y_train, support_l1)
    mean_correl_RFE, std_correl_RFE, mean_correl_y_RFE, std_correl_y_RFE = mean_std_abs_correlation(X_train, y_train, support_RFE)
    mean_correl_l1_RFE, std_correl_l1_RFE, mean_correl_y_l1_RFE, std_correl_y_l1_RFE = mean_std_abs_correlation(X_train, y_train, support_l1_RFE)

    
    
#---Work on ranking by correlation/ranking by RFE
    correlations = [np.corrcoef(y_train, X_train[:,i])[0,1] for i in range(P)]
    rank_column_by_correlation = np.argsort(np.argsort(np.abs(correlations))[::-1])
    

    RFE_rank_real_features = [int(ranking_features[i]) for i in real_support]
    correl_rank_real_features = [rank_column_by_correlation[i] for i in real_support]
    correlations_RFE_real_features = [correlations[i] for i in real_support]
    RFE_pairwise_correlations = [np.corrcoef(y_train, X_train[:,i])[0,1] for i in range(P)]
    rank_column_by_correlation = np.argsort(np.argsort(np.abs(correlations))[::-1])
    

    rank_features_added_l1 = [rank_column_by_correlation[i] for i in list(set(support_l1)-set(real_support))]
    correl_features_added_l1 = [correlations[i] for i in list(set(support_l1)-set(real_support))]
    rank_features_missed_l1 = [rank_column_by_correlation[i] for i in list(set(real_support)-set(support_l1))]
    correl_features_missed_l1 = [correlations[i] for i in list(set(real_support)-set(support_l1))]
    

    rank_features_added_RFE = [rank_column_by_correlation[i] for i in list(set(support_RFE)-set(real_support))]
    correl_features_added_RFE = [correlations[i] for i in list(set(support_RFE)-set(real_support))]
    rank_features_missed_RFE = [rank_column_by_correlation[i] for i in list(set(real_support)-set(support_RFE))]
    correl_features_missed_RFE = [correlations[i] for i in list(set(real_support)-set(support_RFE))]
    

    variables_dropped_l1_RFE = list((set(support_l1)-set(support_l1_RFE)))
    good_variables_dropped_l1_RFE = list((set(real_support) & set(variables_dropped_l1_RFE)))
    wrong_variables_dropped_l1_RFE = list((set(variables_dropped_l1_RFE) - set(real_support)))


    #Use RFE ranking
    RFE_ranking_features_missed = [int(ranking_features[i]) for i in list(set(real_support)-set(support_RFE))]

    idx = np.where(ranking_features == 1)[0]
    rank_correlation_top_RFE = []
    for i in range(2,len(support_l1)+len(real_support)):
        idx = np.where(ranking_features == i)[0]
        if len(idx==1):
            rank_correlation_top_RFE.append(int(rank_column_by_correlation[idx]))
        else:
            rank_correlation_top_RFE.append(rank_column_by_correlation[idx])
    
    
#---Print differences
    write_and_print('\nReal support               :'+str(real_support) ,f)
    write_and_print('RFE rank real features     : '+str(RFE_rank_real_features) ,f)
    write_and_print('Correl rank real features  : '+str(correl_rank_real_features) ,f)
    write_and_print('correlations_RFE real features : '+str(np.round(correlations_RFE_real_features,3)) ,f)


    
    write_and_print('\n\nLen support l1        : '+str(len(support_l1)) ,f)
    write_and_print('Good variable l1      : '+str(np.sort(good_variables_l1)) ,f)
    write_and_print('L2 norm error l1      : '+str(round(diff_norm_l1,2)) ,f)
    write_and_print('Var selection l1      : '+str(len(variable_selection_error_l1))+'    Features : '+str(np.sort(variable_selection_error_l1)) ,f)
    write_and_print('Mean-std correl l1    : '+str(mean_correl_l1)+' '+str(std_correl_l1) ,f)
    write_and_print('Mean-std correl /y l1 : '+str(mean_correl_y_l1)+' '+str(std_correl_y_l1) ,f)


    write_and_print('\nRank/correl added l1  : '+str(np.sort(rank_features_added_l1))+str(np.sort(np.round(correl_features_added_l1,3))[::-1]) ,f)
    write_and_print('Rank/correl missed l1 : '+str(np.sort(rank_features_missed_l1))+str(np.sort(np.round(correl_features_missed_l1,3))[::-1]) ,f)
    write_and_print('correlations_RFE real features : '+str(np.round(correlations_RFE_real_features,3)) ,f)
    
    
    write_and_print('\n\nLen support RFE         : '+str(len(support_RFE)) ,f)
    write_and_print('Good variable RFE       : '+str(np.sort(good_variables_RFE)) ,f)
    write_and_print('L2 norm error RFE       : '+str(round(diff_norm_RFE,2)) ,f)
    write_and_print('Var selection RFE       : '+str(len(variable_selection_error_RFE))+'    Features : '+str(np.sort(variable_selection_error_RFE)) ,f)
    write_and_print('Mean-std correl RFE    : '+str(mean_correl_RFE)+' '+str(std_correl_RFE)  ,f)
    write_and_print('Mean-std correl /y RFE : '+str(mean_correl_y_RFE)+' '+str(std_correl_y_RFE) ,f)

    write_and_print('\nRank/correl added RFE   : '+str(rank_features_added_RFE)+str(np.round(correl_features_added_RFE,3)) ,f)
    write_and_print('Rank/correl missed RFE  : '+str(rank_features_missed_RFE)+str(np.round(correl_features_missed_RFE,3)) ,f)
    write_and_print('RFE ranking missed      : '+str(RFE_ranking_features_missed) ,f)

    write_and_print('\nRanking correlation top RFE (decreasing) : '+str(rank_correlation_top_RFE) ,f)


    write_and_print('\n\nLen support l1 RFE      : '+str(len(support_l1_RFE)) ,f)
    write_and_print('Good variable l1 RFE      : '+str(np.sort(good_variables_l1_RFE)) ,f)
    write_and_print('L2 norm error l1 RFE      : '+str(round(diff_norm_l1_RFE,2)) ,f)
    write_and_print('Var selection l1 RFE      : '+str(len(variable_selection_error_l1_RFE))+'    Features : '+str(np.sort(variable_selection_error_l1_RFE)) ,f)
    write_and_print('Mean-std correl l1 RFE    : '+str(mean_correl_l1_RFE)+' '+str(std_correl_l1_RFE) ,f)
    write_and_print('Mean-std correl /y l1 RFE : '+str(mean_correl_y_l1_RFE)+' '+str(std_correl_y_l1_RFE) ,f)


    write_and_print('\nGood variables dropped l1 RFE  : '+str(np.sort(good_variables_dropped_l1_RFE)) ,f)
    write_and_print('Wrong variables dropped l1 RFE : '+str(np.sort(wrong_variables_dropped_l1_RFE)) ,f)
    




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






    