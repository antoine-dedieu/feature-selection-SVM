import numpy as np
import random
from scipy.stats import norm
from compare_l1_RFE import *


def write_and_print(text,f):
    print text
    f.write('\n'+text)


def shuffle(X, y):

#X: array of size (N,P)
#y: list of size (N,)

    N, P = X.shape
    aux = np.concatenate([X,np.array(y).reshape(N,1)], axis=1)
    np.random.shuffle(aux)

    X = [aux[i,:P] for i in range(N)]
    y = [aux[i,P:] for i in range(N)]

    X = np.array(X).reshape(N,P)
    y = np.array(y).reshape(N,)

    return X,y



def simulate_data_classification(type_Sigma, N,P,k0,rho,f):

#------------BETA-------------
    u_positive = np.zeros(P)

    if(type_Sigma==1):
        #equi-spaced k0
        index = [(2*i+1)*P/(2*k0) for i in range(k0)]
        u_positive[index] = np.ones(k0)

    elif(type_Sigma==2):
        index = random.sample(xrange(P),k0)
        u_positive = np.zeros(P)
        index = [(2*i+1)*P/(2*k0) for i in range(k0)]
        u_positive[:k0] = np.cumsum(0.1*np.ones(k0))
        
    elif(type_Sigma==3):
        u_positive[:k0] = np.ones(k0)
        zeros_P = np.zeros(P)
    
    u_negative = -u_positive


#------------SIGMA-------------
    Sigma = np.zeros(shape=(P,P))

    if(type_Sigma==1 or type_Sigma==3):
        for i in range(P):
            for j in range(P):
                Sigma[i,j]=rho**(abs(i-j))

    elif(type_Sigma==2):
        Sigma = rho*np.ones(shape=(P,P)) + (1-rho)*np.identity(P)



    X_train = np.zeros(shape=(N,P))
    y_train = []
    
    X_test=np.zeros(shape=(N,P))
    y_test = []
    
    
#---CASE 1 AND 2
    if(type_Sigma<=2):
    #------------X_train-------------
        for i in range(int(N/2)):
            X_train[i,:] = np.random.multivariate_normal(u_positive,Sigma,1)
            y_train.append(1)

        for i in range(int(N/2), N):
            X_train[i,:] = np.random.multivariate_normal(u_negative,Sigma,1)
            y_train.append(-1)

        #SHUFFLE
        X_train, y_train = shuffle(X_train, y_train)


    #------------X_test-------------
        for i in range(int(N/2)):
            X_test[i,:] = np.random.multivariate_normal(u_positive,Sigma,1)
            y_test.append(1)

        for i in range(int(N/2), N):
            X_test[i,:] = np.random.multivariate_normal(u_negative,Sigma,1)
            y_test.append(-1)

        #SHUFFLE
        X_test, y_test = shuffle(X_test, y_test)


#---CASE 3
    if(type_Sigma==3):
    #------------X_train-------------
        for i in range(N):
            X_train[i,:] = np.random.multivariate_normal(zeros_P,Sigma,1)
            X_train_u_positive = np.dot(X_train, u_positive)

            probas = [norm.cdf(X_train_u_positive[i]) for i in range(N)]
            randoms = np.random.rand(P)
            y_train = [2*(randoms[i]<probas[i]).astype(int)-1 for i in range(N)]
            
            
    #------------X_test-------------
        for i in range(N):
            X_test[i,:] = np.random.multivariate_normal(zeros_P,Sigma,1)
            X_test_u_positive = np.dot(X_test, u_positive)
            
            probas = [norm.cdf(X_test_u_positive[i]) for i in range(N)]
            randoms = np.random.rand(P)
            y_test = [2*(randoms[i]<probas[i]).astype(int)-1 for i in range(N)]



    write_and_print('DATA CREATED for N='+str(N)+', P='+str(P)+', k0='+str(k0)+' Rho='+str(rho)+' Sigma='+str(type_Sigma), f)

    return X_train, X_test, y_train, y_test, u_positive


