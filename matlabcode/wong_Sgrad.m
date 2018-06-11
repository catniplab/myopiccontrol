function [g,A,B,test] = wong_Sgrad(type,svec,u,mu0,coh,varargin)
%WONG_SGRAD: a function to calculate the time derivative of the Wong
%and Wang 2006 state space model for perceptual decision making

%input
%type: 'linear' for the locally linearized form, and fullly nonlinear for anything else
%svec: (2 x ns) vector of state values, svec(:,k) [S_1(k),S_2(k)], for which g is calculated
%u: (2x1) control term [u_1(t), u_2(t)]^T used to modulate the state
%mu0: strength of input visual stimulus
%coh: coherence of input visuaul stimulus (0 to 100 please!)

%output
%g: (2xns) tensor of [dS1(s1,s2)/dt, dS2(s1,s2)/dt]^T
%A: jacobian w.r.t. state, dg/ds
%B: jacobian w.r.t. control, dg/du

%% do some checks on the s vector sizes to determine output size
ns = size(svec,2);
g = zeros(2,ns);
s1 = svec(1,:);
s2 = svec(2,:);
    



%% parameters from the model
Tnmda = 100;   % NMDA decay time constant; JNS 2006
%Tampa = 2;      % AMPA decay time constant
gamma = 0.641;  % Parameter that relates presynaptic input firing rate to synaptic gating variable
%%%% FI curve parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
a = 270; b = 108; d = 0.1540;  % Parameters for excitatory cells; see JNS2006      
I0E1 = 0.3255; I0E2 = 0.3255; %background current
JAext = 5.2e-4; 
%JAext = 2.243e-4; %original in-text value

%---- Random dot stimulus------------------------------------------------------
%unbiased stimulus. original values commented out. set mu0=0 for no stim
%mu0 = 30; %hz
%coh = 5;
I_stim_1 = JAext*mu0*(1+coh/100);
I_stim_2 = JAext*mu0*(1-coh/100);
%background current and control
I_eta1 = I0E1+u(1);
I_eta2 = I0E2+u(2);
%---- Recurrent synaptic coupling constants-------------------------------------
if ~isempty(varargin)
    JN11 = varargin{1}; 
    JN12 = varargin{2}; 
    JN21 = varargin{3}; 
    JN22 = varargin{4}; 
else
    JN11 = 0.2609; JN22 = 0.2609; 
    JN12 = 0.0497; JN21 = 0.0497; 
end
%---- Resonse function of competiting excitatory population 1 ------
% Total synaptic input to population 1

Isyn1 = JN11.*s1 - JN12.*s2 + I_stim_1 + I_eta1;
efun1 = exp(-d.*(a.*Isyn1-b));
phi1  = (a.*Isyn1-b)./(1-efun1);

Isyn2 = JN22.*s2 - JN21.*s1 + I_stim_2 + I_eta2;
efun2 = exp(-d.*(a.*Isyn2-b));
phi2  = (a.*Isyn2-b)./(1-efun2);

test = [1/(1-efun1),1/(1-efun2)];

%% ---- Dynamical equations ---------------------------------------
%calculate jacobians dS/dx and dS/du, called A and B, respectively
%jacobian w.r.t. state
dS1pdS1 = (-1/Tnmda-gamma*phi1/1000)+gamma*(1-s1).*...
(a*JN11./(1-efun1)-(a*Isyn1-b).*(d*a*JN11).*efun1./(1-efun1).^2)/1000;

dS2pdS2 = (-1/Tnmda-gamma*phi2/1000)+gamma*(1-s2).*...
(a*JN22./(1-efun2)-(a*Isyn2-b).*(d*a*JN22).*efun2./(1-efun2).^2)/1000;

dS1pdS2 = gamma*(1-s1).*...
(-1*a*JN12./(1-efun1)-(a*Isyn1-b).*(d*a*-1*JN12).*efun1./(1-efun1).^2)/1000;

dS2pdS1 = gamma*(1-s2).*...
(-1*a*JN21./(1-efun2)-(a*Isyn2-b).*(d*a*-1*JN21).*efun2./(1-efun2).^2)/1000;

A = zeros(2,2,ns);
A(1,1,:) = dS1pdS1;
A(1,2,:) = dS1pdS2;
A(2,1,:) = dS2pdS1;
A(2,2,:) = dS2pdS2;

%jacobian w.r.t. control
%dS_1'/du_1(t)
dS1pdu1 = gamma*(1-s1).*...
(a./(1-efun1)-(a*Isyn1-b).*(d*a).*efun1./(1-efun1).^2)/1000;
%dS_2'/du_1(t)
dS2pdu2 = gamma*(1-s2).*...
(a./(1-efun2)-(a*Isyn2-b).*(d*a).*efun2./(1-efun2).^2)/1000;

B = zeros(2,2,ns);
B(1,1,:) = dS1pdu1;
B(2,2,:) = dS2pdu2;


%decide if this should be full derivative, or locally linearized one
if ~strcmp(type,'linear')
    %normal form
    g(1,:) = -s1/Tnmda + gamma*(1-s1).*phi1/1000;
    g(2,:) = -s2/Tnmda + gamma*(1-s2).*phi2/1000;
else
    %linearized form  
    for j = 1:ns,
        g(:,j) = A(:,:,j)*svec(:,j)+B(:,:,j)*u;
    end
    
end
    
    
