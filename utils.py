import scipy.stats as st
from scipy.stats import beta
import numpy.random as rnd
import numpy as np

class my_spdf(st.rv_continuous):
    def _pdf(self, x, S50, F50, S51, F51, trial, r):        
        trial *= 1.0        
        a, b = S50, F50 
        bt = beta.ppf(1-1.0/np.sqrt(trial), S51, F51)
        pi_t1 = beta.pdf(x, a, b)
                
        if x >= bt:
            F_t2 =beta.cdf(bt, a, b, loc=0, scale=1)
            ans = (1 - 1/r * F_t2)/(1-F_t2) * pi_t1            
        else:
            ans = 1/r * pi_t1
        return ans

def reg_cal_worst_case(args, r):
# repitation 
    max_mean = max(args.means)
    regret5 = np.zeros(args.T0+1)
    print("Begin BUCB demo")
    print("The goal is to maximize payout from three machines")
    probs5 = np.zeros(args.N+1)

    S5 = np.zeros(args.N, dtype=np.int)
    F5 = np.zeros(args.N, dtype=np.int)
    Arms5 = np.zeros(args.N)

    for trial in range(1, args.T0):
        if trial < args.T0 + 1:
            for i in range(args.N): 
                arra5 = np.zeros(1)
                    # generate the beta list to get arra for each method 
                if i == 1:
                    arra5[0] = beta.ppf((1-1.0/(trial+1)), S5[i] + 1, F5[i] + 1) 
                else: # i ==0 
                    my_cv = my_spdf(a=0, b=1, name='my_pdf')
                    S50, F50, S51, F51 = int(S5[0]), int(F5[0]), int(S5[1]), int(F5[1])

                    arra5[0] = my_cv.ppf( 1-1.0/(trial+1), S50 + 1, F50 + 1, S51 + 1,
                                         F51 + 1, (trial+1), r)

                probs5[i] = arra5[0]          

            machine5 = np.argmax(probs5)
            regret5[trial] = regret5[trial-1] + (max_mean - args.means[machine5])
            Arms5[machine5] = Arms5[machine5]+1
            p5 = rnd.binomial(1, args.means[machine5])
            if p5 == 1: S5[machine5] += 1
            else: F5[machine5] += 1
    return regret5



def reg_cal(args):
# repitation
    
    max_mean = np.max(args.means) 
    rep1 = 2 * args.T0

    regret1 = np.zeros(args.T0+1)
    regret2 = np.zeros(args.T0+1)
    regret3 = np.zeros(args.T0+1)
    regret4 = np.zeros(args.T0+1)
    regret5 = np.zeros(args.T0+1)
    print("Begin EBUCB demo")

    probs1 = np.zeros(args.N+1)
    probs2 = np.zeros(args.N+1)
    probs3 = np.zeros(args.N+1)
    probs4 = np.zeros(args.N+1)
    probs5 = np.zeros(args.N+1)
    
    S1 = np.zeros(args.N, dtype=np.int)
    S2 = np.zeros(args.N, dtype=np.int)
    S3 = np.zeros(args.N, dtype=np.int)
    S4 = np.zeros(args.N, dtype=np.int)
    S5 = np.zeros(args.N, dtype=np.int)
    
    F1 = np.zeros(args.N, dtype=np.int)
    F2 = np.zeros(args.N, dtype=np.int)
    F3 = np.zeros(args.N, dtype=np.int)
    F4 = np.zeros(args.N, dtype=np.int)
    F5 = np.zeros(args.N, dtype=np.int)
    
    Arms1 = np.zeros(args.N)
    Arms2 = np.zeros(args.N)
    Arms3 = np.zeros(args.N)
    Arms4 = np.zeros(args.N)
    Arms5 = np.zeros(args.N)
    #rnd = np.random

    for trial in range(1, args.T0):
        if trial < args.T0 + 1:
            for i in range(args.N): 
                arra1 = np.zeros(rep1)
                arra2 = np.zeros(rep1)
                arra3 = np.zeros(rep1)
                arra4 = np.zeros(rep1)
                arra5 = np.zeros(rep1)
                
                for j in range(rep1):
                    # generate the beta list to get arra for each method 
                    if rnd.beta(1, 1) < args.alpha1:
                        arra1[j] = rnd.beta(S1[i] + 1, F1[i] + 1) 
                        arra2[j] = rnd.beta(S2[i] + 1, F2[i] + 1) 
                        arra3[j] = rnd.beta(S3[i] + 1, F3[i] + 1) 
                        arra4[j] = rnd.beta(S4[i] + 1, F4[i] + 1) 
                        arra5[j] = rnd.beta(S5[i] + 1, F5[i] + 1) 
                    else:
                        arra1[j] = rnd.beta((S1[i] + 1)*0.5, (F1[i] + 1)*0.5) #*2
                        arra2[j] = rnd.beta((S2[i] + 1)*0.5, (F2[i] + 1)*0.5) 
                        arra3[j] = rnd.beta((S3[i] + 1)*0.5, (F3[i] + 1)*0.5) 
                        arra4[j] = rnd.beta((S4[i] + 1)*0.5, (F4[i] + 1)*0.5) 
                        arra5[j] = rnd.beta((S5[i] + 1)*0.5, (F5[i] + 1)*0.5) 

                probs1[i] = np.quantile(arra1, 1-1/((trial)**(1/2))) #EBUCB c = 0
                probs2[i] = np.quantile(arra2, 1-1/((trial)**(1))) #BUCB c = 0
                probs3[i] = np.quantile(arra3, 1-1/((trial)**(1/2)*(np.log(args.T0))**(5))) #'BUCB (c=5)'
                probs4[i] = np.quantile(arra4, 1-1/((trial)**(1)*(np.log(args.T0))**(5))) # 'EBUCB (c=5)' 
                probs5[i] = arra5[0] #'Thompson Sampling'          

            machine1 = np.argmax(probs1)
            machine2 = np.argmax(probs2)
            machine3 = np.argmax(probs3)    
            machine4 = np.argmax(probs4)
            machine5 = np.argmax(probs5)
            
            
            regret1[trial] = regret1[trial-1] + (max_mean-args.means[machine1])
            regret2[trial] = regret2[trial-1] + (max_mean-args.means[machine2])  
            regret3[trial] = regret3[trial-1] + (max_mean-args.means[machine3])
            regret4[trial] = regret4[trial-1] + (max_mean-args.means[machine4])
            regret5[trial] = regret5[trial-1] + (max_mean-args.means[machine5])
            
            
            Arms1[machine1] = Arms1[machine1]+1
            Arms2[machine2] = Arms2[machine2]+1
            Arms3[machine3] = Arms3[machine3]+1
            Arms4[machine4] = Arms4[machine4]+1
            Arms5[machine5] = Arms5[machine5]+1

            p1 = rnd.binomial(1, args.means[machine1])  # [0.0, 1.0)
            p2 = rnd.binomial(1, args.means[machine2])
            p3 = rnd.binomial(1, args.means[machine3])
            p4 = rnd.binomial(1, args.means[machine4])
            p5 = rnd.binomial(1, args.means[machine5])

            if p1 == 1:
            #  print(" -- win")
                S1[machine1] += 1
            else:
            #  print(" -- lose")
                F1[machine1] += 1
                
            if p2 == 1: S2[machine2] += 1
            else: F2[machine2] += 1
                
            if p3 == 1: S3[machine3] += 1
            else: F3[machine3] += 1
                
            if p4 == 1: S4[machine4] += 1
            else: F4[machine4] += 1   
                
            if p5 == 1: S5[machine5] += 1
            else: F5[machine5] += 1

    return regret1, regret2, regret3, regret4, regret5