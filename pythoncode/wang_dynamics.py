#gradients for wong-wang dynamics and koulakov dynamics
#there are both numpy and tensorflow versions in here

import math
import tensorflow as tf
import numpy as np
from tf_funs import *

def koulakov_line(S,I):
    #function to calculate the dynamics dS/dt for a koulakov line, 
    #which acts as a robust neural integrator
    S1 = S[0]
    S2 = S[1]
    dt = 1.0e-3 #multiplicative factor for gradient
    pi = math.pi
    
    #parameters for number of stable points, length of line, etc
    L = 0.7 #length of stable line in state space
    n = 7.0 #number of stable points
    theta = pi/4.0 #the rotation angle of line from horizontal in state space (make it diagonal
    a = 0.2; #amplitude of oscillations in state space         
    
    #the rotation of the line
    A = tf.constant(np.array([[math.cos(theta),-math.sin(theta)],
                              [math.sin(theta),math.cos(theta)]]),
                   dtype=tf.float32) #the rotation matrix
    R0 = [L/2.0,1.0] #point around which to roate
    F = mvMul(A,(S-R0))+R0
    S1r = F[0] #rotated S1
    S2r = F[1] #rotated S2    
    
    #the main functions of the koulakov line, in rotated coordinates
    u = a*tf.sin((n-1)*pi/L*S1r+pi)
    f = tf.tanh( (pi/a)*(u-2.0*(S2r-0.5)))
    g = tf.tanh( (pi/a)*(-u-2.0*(S2r-0.5)))
    
    #boundary conditions. don't use rotated coordinates
    steep = 3.0
    ramprange = 4.0*(n-1); #support of which the tanh will ramp up on bc in x direction
    offset = (steep-1)/(ramprange)

    bcS1 = (10.0*tf.tanh(-pi/steep*ramprange*(S1+offset)) - 
            10.0*tf.tanh(pi/steep*ramprange*(S1-0.65-offset)))                       
    bcS2 = (10.0*tf.tanh(- 2.0*pi/(steep*a)*(S2-L-offset))+
                        10.0*tf.tanh( -2.0*pi/(steep*a)*(S2+offset)))
    
    #final gradients
    K1 = dt*(f+bcS1) + I[0]
    K2 = dt*(g+bcS2) + I[1]   
          
    return tf.stack([K1,K2])


def koulakov_line_nontf(S,I):
    #calculate the koulakov line dynamics dS/dt, outside of tensorflow
    #which acts as a robust neural integrator

    #function to calculate the dynamics for a koulakov line
    S1 = S[0]
    S2 = S[1]
    dt = 1.0e-3 #multiplicative factor for gradient
    pi = math.pi
    
    #parameters for number of stable points, length of line, etc
    L = 0.7 #length of stable line in state space
    n = 7.0 #number of stable points
    theta = pi/4.0 #the rotation angle of line from horizontal in state space (make it diagonal
    a = 0.2; #amplitude of oscillations in state space         
    
    #the rotation of the line
    A = np.array([[math.cos(theta),-math.sin(theta)],
                              [math.sin(theta),math.cos(theta)]]) #the rotation matrix
    R0 = np.array([L/2.0,1.0]) #point around which to roate
    F = A.dot(S-R0)+R0
    S1r = F[0] #rotated S1
    S2r = F[1] #rotated S2    
    
    #the main functions of the koulakov line, in rotated coordinates
    u = a*math.sin((n-1)*pi/L*S1r+pi)
    f = math.tanh( (pi/a)*(u-2.0*(S2r-0.5)))
    g = math.tanh( (pi/a)*(-u-2.0*(S2r-0.5)))
    
    #boundary conditions. don't use rotated coordinates
    steep = 3.0
    ramprange = 4.0*(n-1); #support of which the tanh will ramp up on bc in x direction
    offset = (steep-1)/(ramprange)

    bcS1 = (10.0*math.tanh(-pi/steep*ramprange*(S1+offset)) - 
            10.0*math.tanh(pi/steep*ramprange*(S1-0.65-offset)))                       
    bcS2 = (10.0*math.tanh(- 2.0*pi/(steep*a)*(S2-L-offset))+
                        10.0*math.tanh( -2.0*pi/(steep*a)*(S2+offset)))
    
    #final gradients
    K1 = dt*(f+bcS1) + I[0]
    K2 = dt*(g+bcS2) + I[1]   
          
    return np.array([K1,K2])
    

def wonggrad(S,control,mu0,coh):
    #function to calculate wong dynamics dS/dt
    
    #input: S: synapse activations
    #control: the control terms
    #mu0: mean stimulus parameter
    #coh: coherence
    
    s1 = S[0] #synapse activation of population 1
    s2 = S[1] #synapse activation of population 2
   
    Tnmda = 100. #decay constant for NMDA synapses
    gamma = 0.641 #Parameter that relates presynaptic input firing rate to synaptic gating variable
    #Parameters for excitatory cells; see JNS2006  
    a = 270
    b = 108
    d = 0.1540
    I0E1 = 0.3255
    I0E2 = 0.3255 #background current
    JAext = 5.2e-4  
    I_stim_1 = JAext*mu0*(1+coh/100)
    I_stim_2 = JAext*mu0*(1-coh/100)
    #background current and control
    I_eta1 = I0E1+control[0]
    I_eta2 = I0E2+control[1]
    
    #coupling constants connecting populations 1 and 2
    JN11 = 0.2609
    JN22 = 0.2609
    JN12 = 0.0497 
    JN21 = 0.0497
    
    #Total synaptic input to population 1
    Isyn1 = JN11*s1 - JN12*s2 + I_stim_1 + I_eta1;
    efun1 = tf.exp(tf.minimum(10.,-d*(a*Isyn1-b)));
    phi1  = (a*Isyn1-b)/(1-efun1);

    #Total synaptic input to population 2
    Isyn2 = JN22*s2 - JN21*s1 + I_stim_2 + I_eta2;
    efun2 = tf.exp(tf.minimum(10.,-d*(a*Isyn2-b)));
    phi2  = (a*Isyn2-b)/(1-efun2);
    
    #final 
    g1 = -s1/Tnmda + gamma*(1-s1)*phi1/1000;
    g2 = -s2/Tnmda + gamma*(1-s2)*phi2/1000;
    
    return tf.stack([g1,g2])    
    
def wonggrad_nontf(S,control,mu0,coh):
    #function to calculate wong gradient, outside of tensorflow
    
    #input: S: synapse activations
    # control: the control terms
    #mu0: mean stimulus parameter
    #coh: coherence
    
    s1 = S[0] #synapse activation of population 1
    s2 = S[1] #synapse activation of population 2
    
    Tnmda = 100. #decay constant for NMDA synapses
    gamma = 0.641 #Parameter that relates presynaptic input firing rate to synaptic gating variable
    #Parameters for excitatory cells; see JNS2006  
    a = 270
    b = 108
    d = 0.1540
    I0E1 = 0.3255
    I0E2 = 0.3255 #background current
    JAext = 5.2e-4  
    I_stim_1 = JAext*mu0*(1+coh/100)
    I_stim_2 = JAext*mu0*(1-coh/100)
    #background current and control
    I_eta1 = I0E1+control[0]
    I_eta2 = I0E2+control[1]
    
    #coupling constants connecting populations 1 and 2
    JN11 = 0.2609
    JN22 = 0.2609
    JN12 = 0.0497 
    JN21 = 0.0497
    
    #Total synaptic input to population 1
    Isyn1 = JN11*s1 - JN12*s2 + I_stim_1 + I_eta1;
    efun1 = math.exp(min(10,-d*(a*Isyn1-b)));
    #if min(10,-d*(a*Isyn1-b)) == 10:
    #    print('hit upper limit on Isyn1')
    phi1  = (a*Isyn1-b)/(1-efun1);

    #Total synaptic input to population 2
    Isyn2 = JN22*s2 - JN21*s1 + I_stim_2 + I_eta2;
    #if min(10,-d*(a*Isyn2-b)):
    #    print('hit upper limit on Isyn2')
    efun2 = math.exp(min(10,-d*(a*Isyn2-b)));
    phi2  = (a*Isyn2-b)/(1-efun2);
    
    #final 
    g1 = -s1/Tnmda + gamma*(1-s1)*phi1/1000;
    g2 = -s2/Tnmda + gamma*(1-s2)*phi2/1000;
    
    return np.array([g1,g2])
