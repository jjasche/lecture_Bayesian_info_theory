import numpy as np

def plot_posterior(P,case):
    from IPython.core.pylabtools import figsize
    from matplotlib import pyplot as plt
    figsize(11, 9)
    colours = ["#348ABD", "#A60628"]	

    # For the already prepared, I'm using Binomial's conj. prior.
    k=0
    for pp in P:
        sx = plt.subplot(len(P) / 2, 2, k + 1)
        plt.bar([0, .3], [P[k],1.-P[k]], alpha=0.70, width=0.25,
            color=[colours[1],colours[0]], label= str(k+1)+ " trials",
            lw="3", edgecolor=[colours[1],colours[0]])

        sx.set_ylim([0.,1.])
    
        if k in [len(P) - 2, len(P) - 1]: 
            plt.xticks([0.005, .3], ["Cancer", "No Cancer"]) 
        else:
            plt.xticks([0.005, .3], [" ", " "]) 
    
        #plt.title("Posterior probability of of Cancer")
        plt.ylabel("Probability")
        plt.legend()

        #plt.autoscale(tight=True)
        k+=1

    txt = 'Patient has cancer'
    if case==False:
        txt = 'Patient has no cancer'
        
    plt.suptitle("Bayesian updating of posterior probabilities for the case: " + txt,
             y=1.02,
             fontsize=14)

    plt.tight_layout()
    plt.show()


def generate_patient_data(probs,cancer=True,ndata=10):
    data=np.zeros(ndata)
    if (cancer==True):
        for i in range(ndata):
            data[i]=np.random.choice([1, 0], p=[1-probs['P(Neg|Cancer)'], probs['P(Neg|Cancer)']])
    else:
        for i in range(ndata):
            data[i]=np.random.choice([1, 0], p=[probs['P(Pos|No Canver)'], 1.-probs['P(Pos|No Canver)']])
        
    return data

def get_posterior(probs,data):
    
    Ntot=len(data)
    Npos=np.cumsum(data)
    Nneg=1+np.arange(Ntot)-Npos
  
    P_prior      = probs['P(Cancer)']
    P_likelihood = probs['P(Neg|Cancer)']**Nneg*(1.-probs['P(Neg|Cancer)'])**Npos
    P_a          = probs['P(Cancer)']*probs['P(Neg|Cancer)']**Nneg*(1.-probs['P(Neg|Cancer)'])**Npos 
    P_b          = (1.-probs['P(Cancer)'])*(1.-probs['P(Pos|No Canver)'])**Nneg*probs['P(Pos|No Canver)']**Npos 
    P_norm       = P_a+P_b                                                              
    P            = P_prior * P_likelihood/(P_norm)
     
    return P


