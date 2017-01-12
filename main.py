from compare_l1_RFE import *
from compare_l1_RFE_real_dataset import *


#N_P_k0_list = [(50,200,5),(50,500,10),(50,1000,10), (100,1000,10), (100,2000,10)]
type_Sigma_list = [1]
rho_list = [0.2,0.5,0.8]


#for type_Sigma in type_Sigma_list:
    #for NPk0 in N_P_k0_list:
    	#for rho in rho_list:
        	#N,P,k0 = NPk0
        	#compare_l1_RFE(N,P,k0,float(rho),type_Sigma, max(200,P/5) )




#or i in range(3):
	#compare_l1_RFE(100,2000,10,0.3,2, 400)
	#compare_l1_RFE(100,2000,10,0.5,1, 400)
	#compare_l1_RFE(100,2000,10,0.8,1, 400)


#compare_l1_RFE(100,2000,10,0.8,2, 400)
#compare_l1_RFE(100,2000,10,0.5,2, 400)


#compare_l1_RFE(100,2000,10,0.2,1, 400)


#for i in range(2):
	#for rho in rho_list:
		#compare_l1_RFE(100,500,10,float(rho),3, 200)

#for i in range(5):
#compare_l1_RFE(100,2000,10,0.8,1, 400)

#compare_l1_RFE_real_dataset(1, 400)
#compare_l1_RFE_real_dataset(2, 400)

compare_l1_RFE(50,500,10,0.5,2,200)

#compare_l1_RFE(100,1000,10,0.2,3,200)

#compare_l1_RFE_real_dataset(2,200)