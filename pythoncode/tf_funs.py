import tensorflow as tf
import numpy as np
import math

def mvMul(m,v):
    #Function for matrix vector multiplication of form Mv
    #where v is rank 1 tensor and M is rank 2
    return tf.squeeze(tf.matmul(m,tf.expand_dims(v,axis=1)))

def vmMul(v,m):
    #Function for matrix vector multiplication of form vM
    #where v is rank 1 tensor and M is rank 2
    return tf.squeeze(tf.matmul(tf.expand_dims(v,axis=0),m))

def grad_elemwise(F,X):
    #return the elementwise gradient of F w.r.t X in the following form
    # G = dF/dX, where G[i,j] = dF_i/dX_j
    #F needs to be a rank 1 tensor for this to work,
    #so a reshape operation is needed beforehand
    
    #turn into iterable list of tensors
    Ftemp = tf.unstack(F)
    grads = [tf.gradients(f, X) for f in Ftemp] # if F is a python list
    #convert back to rank 2 tensor with list dimension squeezed out
    G = tf.squeeze(tf.stack(grads, axis=0),axis=1)
    return G

def hess_elemwise(F,X):
    #return the elementwise hessian of F w.r.t X in the following form
    #H = d^2F/dX, where H[i,j,k] = dF_i/dx_j dx_k
    #F needs to be a rank 1 tensor for this to work,
    #so a reshape operation is needed beforehand
    
    #turn into iterable list of tensors
    Ftemp = tf.unstack(F)
    hess = [tf.hessians(f, X) for f in Ftemp] # if F is a python list
    #convert back to rank 2 tensor with list dimension squeezed out
    H = tf.squeeze(tf.stack(hess, axis=0),axis=1)
    return H

def dynamics_linearized(X,model,xdim):
    #calculate the jacobian A(x) = dF/dx and converts from continuous dynamics to discrete.
    #conversion takes dX/dt = Ax --> X_t+1 = (I + A)X_t
    #used for covariance estimate prediction
    #currently only supports 2-dimensional state dynamics
    dF1dx = tf.gradients(model[0],X)
    dF2dx = tf.gradients(model[1],X)
    A = tf.stack([dF1dx[0],dF2dx[0]])
    A_ekf = tf.constant(np.eye(xdim),dtype=tf.float32)+A #convert continuous form to discrete

    return A_ekf

def EKF(X_est,Y_tp1,PI_est,true_model_est,true_model_est_null,Q,R,xdim,dt):
    #extended Kalman filter model
    #X_est: state estimate variable at t
    #Y_tp1: recent observation at t+1
    #PI_est: covariance estimate
    #true_model_est: gradient variable using state estimate, not true state
        #this is evaluated with a control
    #true_model_est_null: gradient variable using state estimate, not true state
        #this is evaluated at null control
    
    #make EKF filter in tensor flow
    #the jacobian A(x) = dF/dx
    dF1dx = tf.gradients(true_model_est_null[0],X_est)
    dF2dx = tf.gradients(true_model_est_null[1],X_est)
    A = tf.stack([dF1dx[0],dF2dx[0]])
    A_ekf = tf.constant(np.eye(xdim),dtype=tf.float32)+A #convert continuous form to discrete

    #linear model params, assumed known for now
    G = tf.constant(dt**(0.5)*np.eye(xdim),dtype=tf.float32) #system noise matrix
    C = tf.constant(np.eye(xdim),dtype=tf.float32) #observation matrix
    I_x = tf.constant(np.eye(xdim),dtype=tf.float32) 

    #estimate propagate
    X_minus = X_est + true_model_est
    #PI_minus = A*PI + PI*A' + G*Q*G'
    PI_minus = tf.matmul(A_ekf,PI_est) + tf.matmul(PI_est,tf.transpose(A_ekf)) + tf.matmul(
        tf.matmul(G,Q),tf.transpose(G))

    #kalman gain
    #K = PI_minus*C'* inv(C*Pi_minus*C' + R)
    #inverse form
    #K = tf.matmul(tf.matmul(PI_minus,tf.transpose(C)),
    #          tf.matrix_inverse(tf.matmul(tf.matmul(C,PI_minus),tf.transpose(C))+R))
    
    #matrix solver form
    #K (C*PI_minus*C'+R) = PI_minus*C'
    #(C*PI_minus*C'+R)' K' = C * PI_minus '
    K = tf.transpose(
        tf.matrix_solve(tf.matmul(tf.matmul(C,PI_minus),tf.transpose(C))+R,
        tf.transpose(tf.matmul(PI_minus,tf.transpose(C)))) ) 
    
    

    #estimate update.
    #X_plus = X_minus +  K*(Y-C*X_minus)
    X_plus = X_minus + mvMul(K,(Y_tp1-mvMul(C,X_minus)))
    #PI_plus = (I_M - K*C)*Pi_minus
    PI_plus = tf.matmul(I_x-tf.matmul(K,C),PI_minus) 
    
    return X_plus, PI_plus


def myopicController(X_est,PI_est,Control,gamma,true_model_est,true_model_est_null,target_model_est,xdim,udim):
    #graphs for updating state and observation
    #true_model_est: state est. gradient, controlled dyamics, must depend upon X_plus, Control
    #true_model_est_null: state est. gradient null control, must depend upon X_plus
    #target_model_est: state. est. target dynamics, must depend upon X_plus

    #control coupling matrix, evaluated at state estimate, NOT TRUE STATE
    B = grad_elemwise(true_model_est,Control)

    #myopic control
    ##d^2f[0]/ dx^2 hessian
    fp = grad_elemwise(true_model_est,X_est)
    fdp = hess_elemwise(true_model_est,X_est)

    ##d^2g[0]/ dx^2 hessian
    ##d^2g[1]/ dx^2 hessian
    gp = grad_elemwise(target_model_est,X_est)
    gdp = hess_elemwise(target_model_est,X_est)

    #dB/dx and d^2B/dx^2, and transposes
    Bvec = tf.reshape(B,[xdim*udim,1])
    Bp = grad_elemwise(Bvec,X_est)
    Bdp = hess_elemwise(Bvec,X_est)
    #reshape them back
    Bp = tf.reshape(Bp,[xdim,udim,xdim])
    Bdp = tf.reshape(Bdp,[xdim,udim,xdim,xdim])
    #transpose operations
    Bpt = tf.transpose(Bp,perm=[1,0,2])
    Bdpt = tf.transpose(Bdp,perm=[1,0,2,3])

    #first expected term E(B^T B) + gamma I
    #gamma = 1e-4 #regularization term
    #(B^T B) + gamma I
    exp1_1 = tf.matmul(tf.transpose(B),B)+gamma*np.eye(xdim,xdim)

    #stacked version of covariance for partial traces
    Pistack4 = tf.stack([tf.stack([PI_est,PI_est]),tf.stack([PI_est,PI_est])])
    Pistack3 = tf.stack([PI_est,PI_est])

    #0.5 B^T Tr_{3,4}[B'' \Sigma]
    exp1_2 = 0.5*tf.matmul(tf.transpose(B),tf.trace(tf.matmul(Bdp,Pistack4))) 
    #0.5 B^T Tr_{3,4}[B''^T \Sigma]
    exp1_2_t = 0.5*tf.matmul(tf.transpose(B),tf.trace(tf.matmul(Bdpt,Pistack4))) 
    exp1_3 = tf.trace(tf.matmul(tf.matmul(Bpt,Bp),Pistack3))
    exp1_approx = exp1_1 + exp1_2 + exp1_2_t + exp1_3
    

    #B'^T* (f-g)
    exp2_1 = mvMul(tf.transpose(B),tf.squeeze(true_model_est_null-target_model_est))
    #0.25* B^T* Tr_{2,3}([f''-g'']Sigma)
    exp2_2 = 0.25*mvMul(tf.transpose(B),tf.trace(tf.matmul((fdp-gdp),Pistack3)))
    #B^T * Tr_{2,3}(B'^T (f'-g') Sigma)
    exp2_3 = mvMul(tf.transpose(B),tf.trace(
        tf.matmul(tf.matmul(Bpt,tf.stack([fp-gp,fp-gp])),Pistack3)))
    #0.5*Tr_{3,4}(B''^T Sigma)*(f-g)
    exp2_4 = 0.5*mvMul(tf.trace(tf.matmul(Bdpt,Pistack4)),
                  tf.squeeze(true_model_est_null-target_model_est))
    exp2_approx = exp2_1 + exp2_2 + exp2_3 +exp2_4
    #Control_new = -1.0*mvMul(tf.matrix_inverse(exp1_approx),exp2_approx)   
    #exp1_approx C = exp2_approx. avoid inversion
    Control_new = tf.squeeze(
        tf.matrix_solve(exp1_approx,-1.0*tf.expand_dims(exp2_approx,1)))
    #Control_new = tf.squeeze(
    #    tf.cholesky_solve(tf.cholesky(exp1_approx),-1.0*tf.expand_dims(exp2_approx,1)))
    
    
    return Control_new


def myopicController_meanonly(X_est,PI_est,Control,gamma,true_model_est,
                              true_model_est_null,target_model_est,xdim,udim):
    #graphs for updating state and observation
    #true_model_est: state est. gradient, controlled dyamics, must depend upon X_plus, Control
    #true_model_est_null: state est. gradient null control, must depend upon X_plus
    #target_model_est: state. est. target dynamics, must depend upon X_plus

    #control coupling matrix, evaluated at state estimate, NOT TRUE STATE
    B = grad_elemwise(true_model_est,Control)

    #first expected term E(B^T B) + gamma I
    #gamma = 1e-4 #regularization term
    #(B^T B) + gamma I
    exp1_1 = tf.matmul(tf.transpose(B),B)+gamma*np.eye(xdim,xdim)

    #B'^T* (f-g)
    exp2_1 = mvMul(tf.transpose(B),tf.squeeze(true_model_est_null-target_model_est))
    
    exp1_approx_meanonly = exp1_1
    exp2_approx_meanonly = exp2_1
    #Control_new = -1.0*mvMul(tf.matrix_inverse(exp1_approx_meanonly),exp2_approx_meanonly)    
    #avoid matrix inversion
    #Control_new = tf.squeeze(
    #    tf.matrix_solve(exp1_approx_meanonly,-1.0*tf.expand_dims(exp2_approx_meanonly,1)))
    Control_new = tf.squeeze(
        tf.cholesky_solve(tf.cholesky(exp1_approx_meanonly),-1.0*tf.expand_dims(exp2_approx_meanonly,1)))

    
    return Control_new

def myopicController_noBdiff(X_est,PI_est,Control,gamma,true_model_est,
                              true_model_est_null,target_model_est,xdim,udim):
    #graphs for updating state and observation, but B is not differentiable with respect to state
    
    #true_model_est: state est. gradient, controlled dyamics, must depend upon X_plus, Control
    #true_model_est_null: state est. gradient null control, must depend upon X_plus
    #target_model_est: state. est. target dynamics, must depend upon X_plus

    #control coupling matrix, evaluated at state estimate, NOT TRUE STATE
    B = grad_elemwise(true_model_est,Control)

    #first expected term E(B^T B) + gamma I
    #gamma = 1e-4 #regularization term
    #(B^T B) + gamma I
    exp1_1 = tf.matmul(tf.transpose(B),B)+gamma*np.eye(xdim,xdim)

    #B^T* (f-g)
    exp2_1 = mvMul(tf.transpose(B),tf.squeeze(true_model_est_null-target_model_est))
    #0.25* B^T* Tr_{2,3}([f''-g'']Sigma)
    Pistack3 = tf.stack([PI_est,PI_est])
    fdp = hess_elemwise(true_model_est,X_est)
    gdp = hess_elemwise(target_model_est,X_est)
    exp2_2 = 0.25*mvMul(tf.transpose(B),tf.trace(tf.matmul((fdp-gdp),Pistack3)))
    
    exp1_approx_meanonly = exp1_1
    exp2_approx_meanonly = exp2_1+exp2_2
    #Control_new = -1.0*mvMul(tf.matrix_inverse(exp1_approx_meanonly),exp2_approx_meanonly)    
    #avoid matrix inversion
    #Control_new = tf.squeeze(
    #    tf.matrix_solve(exp1_approx_meanonly,-1.0*tf.expand_dims(exp2_approx_meanonly,1)))
    Control_new = tf.squeeze(
        tf.cholesky_solve(tf.cholesky(exp1_approx_meanonly),-1.0*tf.expand_dims(exp2_approx_meanonly,1)))

    
    return Control_new

def loss_full(X_est,PI_est,true_model_est,target_model_est):
    #graphs for calculating the fulle expected loss function up to second order
    #E_x [ (f-g)^T(f-g)] , expanded up to second moment
    
    #true_model_est: state est. gradient, controlled dyamics, must depend upon X_plus, Control
    #target_model_est: state. est. target dynamics, must depend upon X_plus
    
    #helpful terms, the gradients and hessians
    Pistack3 = tf.stack([PI_est,PI_est])
    fp = grad_elemwise(true_model_est,X_est)
    gp = grad_elemwise(target_model_est,X_est)
    fdp = hess_elemwise(true_model_est,X_est)
    gdp = hess_elemwise(target_model_est,X_est)

    #mean term
    l0 = tf.norm(true_model_est-target_model_est)**2
    
    #hessian term: (f-g)^T Tr_{2,3}[(f''-g'')\Sigma]
    l2_1 = tf.trace(tf.matmul((fdp-gdp),Pistack3))
    l2 = tf.tensordot(tf.transpose(true_model_est-target_model_est),l2_1,1)
    
    #gradient term: Tr[\Sigma(f'-g')^T(f'-g')]
    l1 = tf.trace(tf.matmul(PI_est,
                            tf.matmul(tf.transpose(fp-gp),fp-gp)))
    
    return [l0 + l1 + l2,l0,l1,l2]

def controller_check(X_est,PI_est,Control,gamma,true_model_est,true_model_est_null,target_model_est,xdim,udim):
    #graphs for checkign out the individual terms int the second order controller
    #true_model_est: state est. gradient, controlled dyamics, must depend upon X_plus, Control
    #true_model_est_null: state est. gradient null control, must depend upon X_plus
    #target_model_est: state. est. target dynamics, must depend upon X_plus

    #control coupling matrix, evaluated at state estimate, NOT TRUE STATE
    B = grad_elemwise(true_model_est,Control)

    #myopic control
    ##d^2f[0]/ dx^2 hessian
    fp = grad_elemwise(true_model_est,X_est)
    fdp = hess_elemwise(true_model_est,X_est)

    ##d^2g[0]/ dx^2 hessian
    ##d^2g[1]/ dx^2 hessian
    gp = grad_elemwise(target_model_est,X_est)
    gdp = hess_elemwise(target_model_est,X_est)

    #dB/dx and d^2B/dx^2, and transposes
    Bvec = tf.reshape(B,[xdim*udim,1])
    Bp = grad_elemwise(Bvec,X_est)
    Bdp = hess_elemwise(Bvec,X_est)
    #reshape them back
    Bp = tf.reshape(Bp,[xdim,udim,xdim])
    Bdp = tf.reshape(Bdp,[xdim,udim,xdim,xdim])
    #transpose operations
    Bpt = tf.transpose(Bp,perm=[1,0,2])
    Bdpt = tf.transpose(Bdp,perm=[1,0,2,3])

    #first expected term E(B^T B) + gamma I
    #gamma = 1e-4 #regularization term
    #(B^T B) + gamma I
    exp1_1 = tf.matmul(tf.transpose(B),B)+gamma*np.eye(xdim,xdim)

    #stacked version of covariance for partial traces
    Pistack4 = tf.stack([tf.stack([PI_est,PI_est]),tf.stack([PI_est,PI_est])])
    Pistack3 = tf.stack([PI_est,PI_est])

    #0.5 B^T Tr_{3,4}[B'' \Sigma]
    exp1_2 = 0.5*tf.matmul(tf.transpose(B),tf.trace(tf.matmul(Bdp,Pistack4))) 
    #0.5 B^T Tr_{3,4}[B''^T \Sigma]
    exp1_2_t = 0.5*tf.matmul(tf.transpose(B),tf.trace(tf.matmul(Bdpt,Pistack4))) 
    exp1_3 = tf.trace(tf.matmul(tf.matmul(Bpt,Bp),Pistack3))
    exp1_approx = exp1_1 + exp1_2 + exp1_2_t + exp1_3
    

    #B'^T* (f-g)
    exp2_1 = mvMul(tf.transpose(B),tf.squeeze(true_model_est_null-target_model_est))
    #0.25* B^T* Tr_{2,3}([f''-g'']Sigma)
    exp2_2 = 0.25*mvMul(tf.transpose(B),tf.trace(tf.matmul((fdp-gdp),Pistack3)))
    #B^T * Tr_{2,3}(B'^T (f'-g') Sigma)
    exp2_3 = mvMul(tf.transpose(B),tf.trace(
        tf.matmul(tf.matmul(Bpt,tf.stack([fp-gp,fp-gp])),Pistack3)))
    #0.5*Tr_{3,4}(B''^T Sigma)*(f-g)
    exp2_4 = 0.5*mvMul(tf.trace(tf.matmul(Bdpt,Pistack4)),
                  tf.squeeze(true_model_est_null-target_model_est))
    exp2_approx = exp2_1 + exp2_2 + exp2_3 +exp2_4
    #Control_new = -1.0*mvMul(tf.matrix_inverse(exp1_approx),exp2_approx)   
    #exp1_approx C = exp2_approx. avoid inversion
    Control_new = tf.squeeze(
        tf.matrix_solve(exp1_approx,-1.0*tf.expand_dims(exp2_approx,1)))
    #Control_new = tf.squeeze(
    #    tf.cholesky_solve(tf.cholesky(exp1_approx),-1.0*tf.expand_dims(exp2_approx,1)))
    
    
    return [Control_new, exp1_1, exp1_2, exp1_2_t, exp1_3, exp2_1, exp2_2, exp2_3 , exp2_4]
