from compare_l1_RFE import *
from compare_l1_RFE_real_dataset import *


#N_P_k0_list = [(50,200,5),(50,500,10),(50,1000,10), (100,1000,10), (100,2000,10)]
type_Sigma_list = [1]
rho_list = [0,0.2,0.5,0.8]


#for type_Sigma in type_Sigma_list:
    #for NPk0 in N_P_k0_list:
    	#for rho in rho_list:
        	#N,P,k0 = NPk0
        	#compare_l1_RFE(N,P,k0,float(rho),type_Sigma, max(200,P/5) )

for i in range(10):
	for rho in rho_list:
		compare_l1_RFE(100,200,10,float(rho),1, max(200,P/5) )

#compare_l1_RFE(50,1000,10,0.5,3,200)

#compare_l1_RFE(100,1000,10,0.2,3,200)

#compare_l1_RFE_real_dataset(1,200)