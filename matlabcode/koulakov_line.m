function [K1,K2,A] = koulakov_line(X,Y,L,n,m,bias,a,tmin,tmax,dt,I_K)
%function koulakov_line. makes koulakov-like alternating stable-unstable
%fixed points
%s:2 x 1 state
%L: domain of state space covered by oscillations
%n: number of stable points (beginning and end will be stable if you choose
%odd number
%m: slope in state space that you want to the points
%b: bias for the line in state space
%st: steepness of basin of attraction
%a: amplitude of oscillations in state space. i.e., where peaks of
%nullclines hit


%% 2D

%memming's test form
%f = @(x,y) tanh(3 * (sin(15*x) - y*3 + 1.5));
%g = @(x,y) -tanh(3 * (cos(15*x) - y*3 + 1.5));

%more general form. place tanh centered at xi
u = @(x,L,n,a) a*sin((n-1)*pi/L*x+pi);
dudx = @(x,L,n,a) a*pi*(n-1)/L*cos((n-1)*pi/L*x+pi);
f = @(x,y,L,n,a) tanh( (pi/a).*(u(x,L,n,a)-2*(y-0.5)));
g = @(x,y,L,n,a) tanh( (pi/a).*(-u(x,L,n,a)-2*(y-0.5)));

%boundary conditions
%bcS1 = @(x) 10*tanh(-pi*(n-1)*x) - 10*tanh(pi*(n-1)*(x-L));
%bcS2 = @(x,y) (tanh(-pi*(n-1)*x) - tanh(-pi*(n-1)*(x-L))+2).*...
%        tanh( -2*pi/a*(y-0.5))*5;

%bcS1 = @(x) 10*tanh(-pi*(n-1)*(x)) - 10*tanh(pi/steep*(n-1)*(x-0.65));
%bcS2 = @(x,y) 10*tanh(- 2*pi/a*(y-L))+10*tanh( -2*pi/a*(y));

steep = 3;
ramprange = 4*(n-1); %support of whih the tanh will ramp up on bc in x direction
offset = (steep-1)/(ramprange);
bcS1 = @(x) 10*tanh(-pi/steep*ramprange*(x+offset)) - 10*tanh(pi/steep*ramprange*(x-0.65-offset));
bcS2 = @(x,y) 10*tanh(- 2*pi/(steep*a)*(y-L-offset))+10*tanh( -2*pi/(steep*a)*(y+offset));
    
%boundary condition derivatives
arg = (n-1)*pi;
dbcS1dx = @(x) -arg*20*(1-tanh(-pi*(n-1)*x).^2)-arg*10*(1-tanh(pi*(n-1)*(x-L)).^2);
dbcS2dx = @(x,y) (-arg*(1-tanh(-pi*(n-1)*x).^2)+arg*(1-tanh(pi*(n-1)*(x-L)).^2)).*...
                tanh( (-2*pi/a*(y-0.5)))*10;
dbcS2dy = @(x,y) (tanh(-pi*(n-1)*x) - tanh(-pi*(n-1)*(x-L))+2).*...
                -20*pi/a*(1-tanh(-2*pi/a*(y-0.5)).^2);
%%
%new way. proper rotations

theta = pi/4;
A = [cos(theta),-sin(theta);sin(theta),cos(theta)];
%B = [0.2;-0.1];
B = [0;0];
R0 = [L/2;1]; %point around which to roate
F = A*([X;Y]-R0)-B+R0;
Xt = F(1);
Yt = F(2);
dxtdx = A(1,1) ;dytdy = A(2,2) ;dxtdy = A(1,2); dytdx = A(2,1);


%Xt = X;
%Yt = Y;
%dxtdx = 1;dytdy = 1;dxtdy = 0; dytdx = 0;

K1 = f(Xt,Yt,L,n,a)+bcS1(X);
K2 = g(Xt,Yt,L,n,a)+bcS2(X,Y);

%K1 = bcS1(X);
%K2 = bcS2(X,Yt);

K1 = dt*K1+I_K(1);
K2 = dt*K2+I_K(2);

%general form
if nargout > 2,
    
    dgdxt = pi/a*dudx(Xt,L,n,a);
    dhdyt = (-2*pi/a);

    dK1dx = (1-f(Xt,Yt,L,n,a).^2).*(dgdxt.*dxtdx+dhdyt.*dytdx);
    dK1dx = dK1dx + dbcS1dx(X);

    dK1dy = (1-f(Xt,Yt,L,n,a).^2).*(dgdxt.*dxtdy+dhdyt.*dytdy);
    dK1dy = dK1dy + dbcS1dx(X);

    dK2dx = (1-g(Xt,Yt,L,n,a).^2).*(-dgdxt.*dxtdx+dhdyt.*dytdx);
    dK2dx = dK2dx + dbcS2dx(X,Yt).*dxtdx + dbcS2dy(X,Yt).*dytdx ;

    dK2dy = (1-g(Xt,Yt,L,n,a).^2).*(-dgdxt.*dxtdy+dhdyt.*dytdy);
    dK2dy = dK2dy + dbcS2dx(X,Yt).*dxtdy+ dbcS2dy(X,Yt).*dytdy;

    %eventually generalize this to vecotirzed operations 
    A = [dK1dx,dK1dy;dK2dx,dK2dy];
    A = dt*A;
end

