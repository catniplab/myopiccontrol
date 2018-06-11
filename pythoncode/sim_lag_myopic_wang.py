# the tensorflow version of the wong/wang dynamics that only uses a TF version of the controller
#the rest is in numpy

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
from tf_funs import * #EKF, and helper in tensorflow
from wang_dynamics import * #the gradients for wong/wang dynamics
import sys
import pickle

#tf.set_random_seed(101)
pi = math.pi
#np.random.seed(101)

#sess = tf.Session()
session_conf = tf.ConfigProto(
      intra_op_parallelism_threads=4,
      inter_op_parallelism_threads=4)
sess = tf.Session(config=session_conf)


#build graphs
xdim = 2 #state dimension
udim = 2 #control dimension

X_est = tf.placeholder(shape=(xdim),dtype=tf.float32,name='X_est') #the state estimate
PI_est = tf.placeholder(shape = (xdim,xdim),dtype=tf.float32, name = 'PI_est') #estimated covariance
Y_tp1 = tf.placeholder(shape=(xdim),dtype=tf.float32, name = 'Y_tp1') #the most recent observation
#Q  = tf.placeholder(dtype=tf.float32)
#R = tf.placeholder(dtype=tf.float32)
Control = tf.placeholder(shape = udim, dtype=tf.float32, name='Control')

#params for experiment
#wong dynamics
T = 1000 #number of steps
mu0 = 30. #stimulus strength
coh = -6.0 #coherence
gamma = 1e-4 #regularization term. 1e-3 too weak

#koulakov dynamics
#set up analogous stimulus to koulakov line
ifact = 7.5e-4 #scaling factor, found by inspection
I_k = [0,0]
I_kbase = ifact*(1+abs(coh)/100.0);
if coh < 0:
    I_k = [-I_kbase,I_kbase]
elif coh > 0:
     I_k = [I_kbase,-I_kbase]

#define the noise for the system
dt = 1.0e-3
sigsstate = (1./dt)*5e-8 #not sure how strong this should be yet
if len(sys.argv) < 5:
    sigsobs = 1.0e-9 #default value
else:
    sigsobs = float(sys.argv[4])

Q = sigsstate*np.eye(xdim)
Q_tf = tf.constant(Q,dtype=tf.float32, name = 'Q') #state noise covariance
R = sigsobs*np.eye(xdim)
R_tf = tf.constant(R,dtype=tf.float32, name = 'R') #observation noise covariance  

#graphs for updating state and observation
true_model_est = wonggrad(X_est,Control,mu0,coh) #state est. gradient, full myopic
true_model_est_null = wonggrad(X_est,[0.,0.],mu0,coh)#state est. gradient null control
target_model_est = koulakov_line(X_est,I_k) #state est. target dynamics

#the non-tensorflow anonymous functions, for generalizations
true_nontf = lambda x,c: wonggrad_nontf(x,c,mu0,coh)
target_nontf = lambda x: koulakov_line_nontf(x,I_k) 

X_plus,PI_plus = EKF(X_est,Y_tp1,PI_est,true_model_est,true_model_est_null,Q_tf,R_tf,xdim,dt)

#myopic controller
#graphs for controller

useMO = int(sys.argv[3]) #handle to use mean only. If ==1, then use mean-only. otherwise full
if useMO ==1:
    print('using mean-only control')
    Cnew = myopicController_meanonly(
        X_est,PI_est,Control,gamma,true_model_est,
        true_model_est_null,target_model_est,xdim,udim)
else:
    print('using full myopic control')
    Cnew = myopicController(
        X_est,PI_est,Control,gamma,true_model_est,
        true_model_est_null,target_model_est,xdim,udim)

#covariance prediction update graph
Ak = dynamics_linearized(X_est,true_model_est_null,xdim)

#the full loss function, not just loss of mean values
loss_tf = loss_full(X_est,PI_est,true_model_est,target_model_est)

ns = 500 #number of samples

#make these numpy version
statenoise = np.random.normal(0,sigsstate**0.5,[xdim,T,ns])
obsnoise = np.random.normal(0,sigsobs**0.5,[xdim,T,ns])
G = dt**(0.5)*np.eye(xdim) #system noise matrix, for covariance prediction

import pickle
fname = "noise_bad_lag10_wang"
index = ['statenoise','obsnoise']
alldata = [index,statenoise,obsnoise]
pickle.dump( alldata, open( fname, "wb" ) )


x_estvec = np.zeros((xdim,T,ns))
xvec = np.zeros((xdim,T,ns))
yvec = np.zeros((xdim,T,ns))
x_targvec = np.zeros((xdim,T,ns))
PI_estvec = np.zeros((xdim,xdim,T,ns))
contall = np.zeros((udim,T,ns))

loss = np.zeros((4,T,ns))
loss_nocont = np.zeros((4,T,ns))
loss_true = np.zeros((T,ns))
initvals = np.zeros((xdim,ns))


lag = int(sys.argv[2]) #how many steps in the past will we receive observations

init = tf.global_variables_initializer()
#sess.run(init)

for m in range(ns):
    sess.run(init)
    #x_init = [0.9,0.76] #initial state
    #x_init = np.array([ 0.79408495,  0.20955779])
    x_init = np.random.uniform(0.1,0.2,(2,))
    initvals[:,m] = x_init
    print(x_init)
    PI_init = [[1.0e-6,0.],[0.,1.0e-6]] #initial covariance
    c_init = [0.,0.]

    xest_k = x_init
    pi_k = PI_init
    c_k = c_init
    x_k = x_init
    x_targ_k = x_init
    ykp1 = np.array(x_init)

    x_estvec[:,0,m] = x_init
    xvec[:,0,m] = x_init
    x_targvec[:,0,m] = x_init
    PI_estvec[:,:,0,m] = PI_init

    print(m)
    
    #go ahead and propagate lag-steps ahead before starting state estimation and such
    for k in range(1,lag):
        #update actual dynamics
        grad_cont = true_nontf(xvec[:,k-1,m],c_init)
        grad_targ = target_nontf(x_targvec[:,k-1,m])

        xvec[:,k,m] = xvec[:,k-1,m] + grad_cont + statenoise[:,k,m]
        x_targvec[:,k,m] = x_targvec[:,k-1,m] + grad_targ + statenoise[:,k,m]
        yvec[:,k,m] = xvec[:,k,m] + obsnoise[:,k,m]

        #set estimates in beginning lags to initial state
        x_estvec[:,k,m] = x_init
        PI_estvec[:,:,k,m] = PI_init
    for k in range(max(1,lag),T): 
        #update actual dynamics
        grad_cont = true_nontf(xvec[:,k-1,m],contall[:,k-1,m])
        grad_targ = target_nontf(x_targvec[:,k-1,m])
        xvec[:,k,m] = xvec[:,k-1,m] + grad_cont + statenoise[:,k,m]
        x_targvec[:,k,m] = x_targvec[:,k-1,m] + grad_targ + statenoise[:,k,m]
        yvec[:,k,m] = xvec[:,k,m] + obsnoise[:,k,m]

        #run state estimator to update estimate of state k-lag
        test = sess.run([X_plus,PI_plus],
                        {X_est:x_estvec[:,k-lag,m], 
                         PI_est:PI_estvec[:,:,k-lag,m], 
                         Control:contall[:,k-lag,m],Y_tp1:yvec[:,k-lag+1,m]})
        x_estvec[:,k-lag+1,m] = test[0]
        PI_estvec[:,:,k-lag+1,m] = test[1]
        
        #predit lag states in the future to calculate control
        x_est_n = x_estvec[:,k-lag+1,m]
        PI_est_n = PI_estvec[:,:,k-lag+1,m]
        
        for n in range(1,lag):
            #state prediction step
            grad_cont = true_nontf(x_est_n,contall[:,k-lag+n,m])

            #covariance prediction step. calculate jacobian
            Ak_n= sess.run(Ak,
                        {X_est: x_est_n, PI_est: PI_est_n,
                         Control: contall[:,0,m], Y_tp1:yvec[:,0,m]})

            x_est_n = x_est_n + grad_cont
            PI_est_n = np.matmul(Ak_n,PI_est_n) + np.matmul(PI_est_n,np.transpose(Ak_n)) + np.matmul(
                np.matmul(G,Q),np.transpose(G))

        #run myopic controller using predicted state estimated. cov, doesnt matter
        #find control for time k
        c_k = sess.run(Cnew,{X_est:x_est_n, PI_est:PI_est_n,
                             Control:contall[:,k-1,m], Y_tp1:yvec[:,k,m]})
        #do quick comparison for jumps in control due to a singularity in dynamics

        if abs(np.linalg.norm(c_k)) > 100.:
            contall[:,k,m] = contall[:,k-1,m]
            print('dynamics likely got singular. hold tight')
            print(k)
        else:
             contall[:,k,m] = c_k
        
        loss_true[k-lag+1,m] = np.linalg.norm(true_nontf(xvec[:,k-lag+1,m],contall[:,k-lag+1,m])-
        target_nontf(xvec[:,k-lag+1,m]))**2
        
        ltest = sess.run(loss_tf,{X_est:x_estvec[:,k-lag+1,m],
                                               PI_est:PI_estvec[:,:,k-lag+1,m],
                                               Control:contall[:,k-lag+1,m]
                                              })
        loss[:,k-lag+1,m] = ltest
        
        #loss_nocont[k-lag,m] = np.linalg.norm(
        #    true_nontf(x_estvec[:,k-lag,m],[0.,0.])-
        #    target_nontf(x_estvec[:,k-lag,m]))

        ltest = sess.run(loss_tf,{X_est:x_estvec[:,k-lag+1,m],
                                               PI_est:PI_estvec[:,:,k-lag+1,m],
                                               Control:np.array([0.,0.])
                                              })
        loss_nocont[:,k-lag+1,m] = ltest
        
        
    #set final lag estimate values to esimate
    for k in range(lag-1):
        x_targvec[:,T-lag+1+k,m] = x_targvec[:,T-lag,m]
        x_estvec[:,T-lag+1+k,m] = x_estvec[:,T-lag,m]
        PI_estvec[:,:,T-lag+1+k,m] = PI_estvec[:,:,T-lag,m]

#end simulation section
fname = sys.argv[1]
#index = ['x_estvec','x_targvec','PI_estvec','contall_nolag','contall_meanonly_nolag','loss','loss_nocont','loss_meanonly_nolag','loss_meanonly','loss_nolag']
#alldata = [index,x_estvec,x_targvec,PI_estvec,contall_nolag,contall_meanonly_nolag,loss,loss_nocont,loss_meanonly_nolag,loss_meanonly,loss_nolag]

fname = sys.argv[1]
index = ['x_estvec','x_targvec','PI_estvec','contall','loss','loss_nocont','loss_true','xvec','yvec']
alldata = [index,x_estvec,x_targvec,PI_estvec,contall,loss,loss_nocont,loss_true,xvec,yvec]
pickle.dump( alldata, open( fname, "wb" ) )

