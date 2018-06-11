function [K,dK] = gabor_lobes(x,m,L)
%function to make stable points using gabor functions
%input:
%s: n x 1 state
%mvec: nx 1 stable point
%s: 2x 1 state space to be calculated
%w: weights for the gaussians
%L: basin of attraction

%only working for s as a 2x1 element
assert(logical(prod(size(x)==size(m))),'x and m need to be same dimensions');
n = size(x,1);

ic = 1/(L/2)^2*eye(size(x,1));
%gaussfun = 1/(sqrt(2*pi*1/det(ic)))*exp(-0.5*(x-m)'*ic*(x-m));
gaussfun = exp(-0.5*(x-m)'*ic*(x-m));
K = -sin(pi/L*(x-m))*gaussfun;
%dK = -pi/L*cos(pi/L*(x-m))*ones(1,n)*gaussfun + sin(pi/L*(x-m))*(x-m)'*ic*gaussfun;
dK = -diag(pi/L*cos(pi/L*(x-m)))*gaussfun + sin(pi/L*(x-m))*(x-m)'*ic*gaussfun;