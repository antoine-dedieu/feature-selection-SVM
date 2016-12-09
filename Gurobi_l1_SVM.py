import numpy as np
from gurobipy import *


def l1_SVM(X, y, C, time_limit, model):

#ALPHA : parameter C
#MORE WARM START / POSSIBILITY TO USE THE PREVIOUS MODEL


#---DEFINE A NEW MODEL IF NO PREVIOUS ONE
    N,P = X.shape

    if(model == 0):
    
    #---VARIABLES
        l1_SVM=Model("l1_svm")
        l1_SVM.setParam('TimeLimit', time_limit)

        beta = np.array([l1_SVM.addVar(lb=-GRB.INFINITY, name="beta_"+str(i)) for i in range(P)])
        b0 = l1_SVM.addVar(lb=-GRB.INFINITY, name="b0")
        

    #UPDATE PARAMETER : to speed up CV
        C0 = l1_SVM.addVar(name="C0")

    
        #Absolut values
        abs_beta = np.array([l1_SVM.addVar(lb=0, name="abs_beta_"+str(i)) for i in range(P)])
        #Hinge loss
        u = np.array([l1_SVM.addVar(lb=0, name="hinge_loss_"+str(i)) for i in range(N)])
        l1_SVM.update()


    #---OBJECTIVE VALUE WITH HINGE LOSS AND L1-NORM
        l1_SVM.setObjective(C0*quicksum(u) + quicksum(abs_beta), GRB.MINIMIZE)


    #---CONSTRAINTS
        #Contraint for updating lambda 
        l1_SVM.addConstr(C0 == C, name='coefficient')
    
        #Define absolute value constraints
        for i in range(P):
            l1_SVM.addConstr(abs_beta[i] >= beta[i], name='abs_beta_sup_'+str(i))
            l1_SVM.addConstr(abs_beta[i] >= -beta[i], name='abs_beta_inf_'+str(i))

        #Define max constraint
        for i in range(N):
            l1_SVM.addConstr(u[i] >= 1-y[i]*(b0 + quicksum([X[i][k]*beta[k] for k in range(P)])), 
                                name="slack_"+str(i))

            
        
        
#---IF PREVIOUS MODEL JUST UPDATE THE PENALIZATION
    else:
        l1_SVM = model.copy()
        constraint = l1_SVM.getConstrByName('coefficient')
        C0 = l1_SVM.getVarByName('C0')
        l1_SVM.remove(constraint)
        l1_SVM.addConstr(C0 == C, name='coefficient')
    


    
#---SOLVE
    l1_SVM.optimize()
                    
    beta_l1_SVM = [l1_SVM.getVarByName("beta_"+str(i)).x for i in range(P)]
    b0_l1_SVM = l1_SVM.getVarByName("b0").x
    
    model_status = l1_SVM.status
    if (l1_SVM.status == GRB.Status.OPTIMAL):
        model_status = 'Optimal'
    elif (l1_SVM.status == GRB.Status.TIME_LIMIT):
        model_status = 'Time Limit'

    
    return np.round(beta_l1_SVM,6) , np.round(b0_l1_SVM), model_status, l1_SVM


