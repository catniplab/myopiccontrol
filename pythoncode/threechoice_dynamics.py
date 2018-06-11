#diseased and healthy dynamics gradients
import tensorflow as tf
import numpy as np
import math
pi = math.pi

def gabor_lobes(x,m,L):
    #function to make stable points using gabor functions, non-tensorflow graph version
    #input:
    #s: n x 1 state for which value is being evaluated
    #m: nx 1 stable point location
    #L: basin of attraction, i.e., width of the gaussian envelope around stable point
    n = len(x)
    ic = 1/(L/2.)**2
    gaussfun = np.exp(-0.5*ic*np.inner((x-m),(x-m)))
    
    K = -np.sin(pi/L*(x-m))*gaussfun
    return K

def gabor_lobes_tf(x,m,L,n):  
    #function to make stable points using gabor functions, tensorflow version
    #input:
    #s: n x 1 state for which value is being evaluated
    #m: nx 1 stable point location
    #L: basin of attraction, i.e., width of the gaussian envelope around stable point
    
    ic = 1/(L/2.)**2
    gaussfun = tf.exp(-0.5*ic*tf.norm((tf.squeeze(x)-m))**2)
                     
    K = -tf.sin(pi/L*(tf.squeeze(x)-m))*gaussfun
    return K


def grad_threechoice(V,n,dt,Control):
    import numpy as np
    #gradient [dV/dt, dn/dt] for the diseased state dynamics, with a limit cycle attractor
    #and three stable points
    #based on the Na-K model spiking neuron model from Dynamical systems in Neuroscience, Ch4, Izhikevich
    
    #V should range from 0 to 1
    #transform from -100 to 20
    Vold = V
    V = 180.*V-80.

    #constants
    C = 1.
    I = 10.
    ENa = 60. #mV
    Ek = -90. #mV
    gNa = 20. 
    gK = 10. 
    gL = 8.
    V_minf = -20. 
    k_minf = 15.
    tau = 1.

    #high threshold params
    El = -80. #mV
    V_ninf = -25.
    k_ninf = 5.

    #low threshold params
    #El = -78. #mV
    #V_ninf = -45.
    #k_ninf = 5.
 
    #mInf = 1./(1+np.exp((V_minf-V)/k_minf))
    #nInf = 1./(1+np.exp((V_ninf-V)/k_ninf))
    #try another function 
    emaxarg = 10.
    exargm = min(emaxarg,(V_minf-V)/k_minf)
    mInf = 1./(1+np.exp(exargm))
    exargn = min(emaxarg,(V_ninf-V)/k_ninf)
    nInf = 1./(1+np.exp(exargn))
    #if exargm==10.:
    #    print('it happened m')
    #    print((V_ninf-V)/k_ninf)
    #elif exargn==10.:
    #    print('it happened n')
    #    print((V_ninf-V)/k_ninf)
    #print((V_ninf-V)/k_ninf)
      
    Vprime = 1/C*(I-gL*(V-El)-gNa*mInf*(V-ENa) - gK*n*(V-Ek))
    nprime = (nInf-n)/tau
    
    #rescale each dimension
    rfactn = 160.
    vf = 180.
    rfactV = 160./vf
    rfact = np.array([rfactV,rfactn])
    Kold = np.array([Vprime,nprime])*rfact
    
    #add stable points
    chX = 0.9
    #position of stable points
    meanvec = np.array([[chX,0.25],
                [chX,0.5],
                [chX,0.75]])
    L = 0.2 #width of the stable points
    
    x = np.array([Vold,n])
    
    K1 = gabor_lobes(x,meanvec[0,:],L)
    K2 = gabor_lobes(x,meanvec[1,:],L)
    K3 = gabor_lobes(x,meanvec[2,:],L)
    
    #scale factors for stable points
    rfactn = 100.
    rfactV = 200.
    rfactnew = np.array([rfactV,rfactn])

    #create muliplicative filters centered at each stable point
    #to smoothly remove original limit cycle behavior at those points
    ic = 1/((1.5*L)/2)**2 #inverse covariance of gaussian
    #gaussian filters
    gm = lambda x,m: 1-np.exp(-0.5*ic*np.inner((x-m),(x-m)))
    gm1 = gm(x,meanvec[0,:])
    gm2 = gm(x,meanvec[1,:])
    gm3 = gm(x,meanvec[2,:])

    #multiplicative gaussian mask centered around stable point position
    #and addition of new stable points
    K = dt*(Kold*gm1*gm2*gm3+ rfactnew*(K1 + K2 + K3))+Control

    return K

def grad_threechoice_tf(V,n,dt,Control):
    #gradient [dV/dt, dn/dt] for the diseased state dynamics, with a limit cycle attractor
    #and three stable points
    #based on the Na-K model spiking neuron model from Dynamical systems in Neuroscience, Ch4, Izhikevich
    
    #V should range from 0 to 1
    #transform from -100 to 20
    Vold = V
    V = 180.*V-80.

    #constants
    C = 1.
    I = 10.
    ENa = 60. #mV
    Ek = -90. #mV
    gNa = 20. 
    gK = 10. 
    gL = 8.
    V_minf = -20. 
    k_minf = 15.
    tau = 1.

    #high threshold params
    El = -80. #mV
    V_ninf = -25.
    k_ninf = 5.

    #low threshold params
    #El = -78. #mV
    #V_ninf = -45.
    #k_ninf = 5.

    #nInf = @(V12,k,V) 1./(1+exp((V12-V)/k));   
    #mInf = 1./(1+tf.exp((V_minf-V)/k_minf))
    #nInf = 1./(1+tf.exp((V_ninf-V)/k_ninf))
    emaxarg = 10.
    exargm = tf.minimum(emaxarg,(V_minf-V)/k_minf)
    mInf = 1./(1+tf.exp(exargm))
    exargn = tf.minimum(emaxarg,(V_ninf-V)/k_ninf)
    nInf = 1./(1+tf.exp(exargn))        
      
    Vprime = 1/C*(I-gL*(V-El)-gNa*mInf*(V-ENa) - gK*n*(V-Ek))
    nprime = (nInf-n)/tau
    
    #rescale each dimension
    rfactn = 160.
    vf = 180.
    rfactV = 160./vf
    rfact = np.array([rfactV,rfactn])
    #Kold = tf.concat([Vprime,nprime],axis=0)*rfact #old way
    Kold = tf.stack([Vprime,nprime])*rfact #new way
    
    #add stable points
    chX = 0.9
    #position of stable points
    meanvec = np.array([[chX,0.25],
                [chX,0.5],
                [chX,0.75]])
    L = 0.2 #width of the stable points
    
    x = tf.stack([Vold,n])
    
    K1 = gabor_lobes_tf(x,meanvec[0,:],L,2)
    K2 = gabor_lobes_tf(x,meanvec[1,:],L,2)
    K3 = gabor_lobes_tf(x,meanvec[2,:],L,2)
    
    #scale factors for stable points
    rfactn = 100.
    rfactV = 200.
    rfactnew = np.array([rfactV,rfactn])

    #create muliplicative filters centered at each stable point
    #to smoothly remove origial limit cycle behavior at those points
    ic = 1/((1.5*L)/2)**2 #inverse covariance of gaussian
    #gaussian mask
    gm = lambda x,m: 1-tf.exp(-0.5*ic*tf.norm((tf.squeeze(x)-m))**2)
    gm1 = gm(x,meanvec[0,:])
    gm2 = gm(x,meanvec[1,:])
    gm3 = gm(x,meanvec[2,:])

    #multiplicative gaussian mask centered around stable point position
    #and addition of new stable points
    K = dt*(Kold*gm1*gm2*gm3+ rfactnew*(K1 + K2 + K3))+Control
    return K

def grad_threechoice_healthy(V,n,dt):
    #The ficticious dynamical system [dV/dt, dn/dt] that is the candidate for 
    #a healthy version of the dyanamics, which blocks the region
    #originally containing the limit cycle attractor,
    #and keeps the three stable points
    
    #primary design strategy. place multiplicative filter to keep wanted parts of original dynamics
    #and add a barrier function to prevent movement into an undesired region. form
    # new_dynamics = filter*orignal_dynamics + barrier

    #original dynamics
    Kold = grad_threechoice(V,n,dt,[0.,0.])

    #state array
    x = np.array([V,n])
    #x at which position to place barrier and filter. will hold for any y position. 
    #i.e., a vertical barrier in a 2d state sapce
    x0 = 0.9
    
    #scaling and parameters for the hyperbolic tangent barrier
    tfact = 0.05 #steepness of hyperbolic tangent barrier
    rfactn = 10*dt #doesn't actually matter I don't think...
    rfactV = 20*dt
    
    rfactnew = np.array([rfactV,rfactn]) #scaling factor for each dimension of dynamics
       
    tm_pos = 0.5*(math.tanh((x[0]-x0)*pi/tfact)+1) #1 on postivie side of xi
    tm_neg = 0.5*(math.tanh(-(x[0]-x0)*pi/tfact)+1) #1 on negative side of xi
    #the additive barrier to prevent limit cycle. a hyperbolic tangent
    barrier = rfactnew*[tm_neg,0]

    #the multiplicative filter to preserve wanted dynamics
    gfilt = np.array([tm_pos,tm_pos])

    K = gfilt*Kold + barrier
    
    return K

def grad_threechoice_healthy_tf(V,n,dt):
    #The ficticious dynamical system [dV/dt, dn/dt] that is the candidate for 
    #a healthy version of the dyanamics, which blocks the region
    #originally containing the limit cycle attractor,
    #and keeps the three stable points
    
    #primary design strategy. place multiplicative filter to keep wanted parts of original dynamics
    #and add a barrier function to prevent movement into an undesired region. form
    # new_dynamics = filter*orignal_dynamics + barrier

    #original dynamics
    Kold = grad_threechoice_tf(V,n,dt,np.array([0.,0.]))

    #state array
    #x = tf.concat([V,n],axis=0) #old way
    x = tf.stack([V,n])#new way
    #x at which position to place barrier and filter. will hold for any y position. 
    #i.e., a vertical barrier in a 2d state sapce
    x0 = 0.9
    
    #scaling and parameters for the hyperbolic tangent barrier
    rtfact = 0.05; #steepness of hyperbolic tangent barrier
    rfactn = 10*dt; #doesn't actually matter I don't think...
    rfactV = 20*dt;
    tfact = 0.05
    
    rfactnew = np.array([rfactV,rfactn]) #scaling factor for each dimension of dynamics
       
    tm_pos = 0.5*(tf.tanh((x[0]-x0)*pi/tfact)+1) #1 on postivie side of xi
    tm_neg = 0.5*(tf.tanh(-(x[0]-x0)*pi/tfact)+1) #1 on negative side of xi
    #the additive barrier to prevent limit cycle. a hyperbolic tangent
    barrier = rfactnew*tf.stack([tm_neg,0.])

    #the multiplicative filter to preserve wanted dynamics
    gfilt = [tm_pos,tm_pos]

    K = gfilt*Kold + barrier
    
    return K