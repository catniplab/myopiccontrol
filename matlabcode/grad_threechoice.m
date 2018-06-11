function [K,dK] = grad_threechoice(V,n,dt)
%my ficticious dynamical system containing a limit cycle attractor
%and three stable points

%V shoudl range from 0 to 0.5
%transform to -100 to 20
Vold = V;
V = 180*V-80;

C = 1;
I = 10;
ENa = 60; %mV
Ek = -90; %mV
gNa = 20; gK = 10; gL = 8;
V_minf = -20; k_minf = 15;
tau = 1;

%high threshold params
El = -80; %mV
V_ninf = -25; k_ninf = 5;

%low threshold params
%El = -78; %mV
%V_ninf = -45; k_ninf = 5;

nInf = @(V12,k,V) 1./(1+exp((V12-V)/k));

Vp = 1/C*(I-gL*(V-El)-gNa*nInf(V_minf,k_minf,V).*(V-ENa) - gK*n.*(V-Ek));
np = (nInf(V_ninf,k_ninf,V)-n)/tau;

%rescale each dimension
rfactn = 160;
vf = 180;
rfactV = 160/vf;
rfact = [rfactV;rfactn];
Kold = [Vp;np].*rfact;

if nargout > 1
    dInfdV = @(V12,k,V) 1/k*exp((V12-V)/k).*nInf(V12,k,V).^2;
    dVtdV =  1/C*(-gL-gNa*dInfdV(V_minf,k_minf,V)*(V-ENa)-gNa*nInf(V_minf,k_minf,V)-gK*n);
    dVtdn = -gK*(V-Ek);
    dntdV = dInfdV(V_ninf,k_ninf,V)/tau;
    dntdn = -1/tau;
    dKold = [rfact*vf,rfact].*[dVtdV,dVtdn;dntdV,dntdn];
end


%add stable points
chX = 0.9;
%chXnew = chX*180-80; %placement of lobes in the transformed V range
meanvec = [chX,0.25;
            chX,0.5;
            chX,0.75]';
L = 0.2;
[K1,dK1] = gabor_lobes([Vold;n],meanvec(:,1),L);
[K2,dK2] = gabor_lobes([Vold;n],meanvec(:,2),L);
[K3,dK3] = gabor_lobes([Vold;n],meanvec(:,3),L);

%scale factors for stable points
rfactn = 100;
rfactV = 200;
rfactnew = [rfactV;rfactn];

x = [Vold;n];
ic = 1/((1.5*L)/2)^2*eye(size(x,1));
gm = @(x,m) 1-exp(-0.5*(x-m)'*ic*(x-m));
gm1 = gm(x,meanvec(:,1));
gm2 = gm(x,meanvec(:,2));
gm3 = gm(x,meanvec(:,3));

K = dt*(Kold*gm1*gm2*gm3+ rfactnew.*(K1 + K2 + K3));

if nargout > 1
    dgm = @(x,m) (x-m)'*ic*exp(-0.5*(x-m)'*ic*(x-m));
    dgm1 = dgm(x,meanvec(:,1));
    dgm2 = dgm(x,meanvec(:,2));
    dgm3 = dgm(x,meanvec(:,3));
    dK = (dKold)*(gm1*gm2*gm3)+ (Kold*dgm1)*gm2*gm3 + gm1*(Kold*dgm2)*gm3 + gm1*gm2*(Kold*dgm3)+[rfactnew,rfactnew].*(dK1 + dK2 + dK3);
    dK = dt*dK;
end
        
    

