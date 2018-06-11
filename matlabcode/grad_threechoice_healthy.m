function [K,dK] = grad_threechoice_healthy(V,n,dt)
%my ficticious dynamical system that is the candidate for 
%a healthy version of the dyanamics, which blocks the region
%originally containing the limit cycle attractor
%and three stable points

%original dynamics
[Kold,dKold] = grad_threechoice(V,n,dt); %old dyanamics

x = [V;n];
%OLD
%rfactn = 10*dt; %doesn't actually matter I don't think...
%rfactV = 20*dt;
%NEW
rfactn = 10*dt; %doesn't actually matter I don't think...
rfactV = 20*dt;

rfactnew = [rfactV;rfactn];


%tfact = 0.05; %steepness to the regime. OLD
tfact = 0.05;

x0 = 0.9; %x position to place filter. will hold for any y position
%y0 = 0.5;
tm_pos = @(x,xi) 0.5*(tanh((x-xi)*pi/tfact)+1); %1 on postivie side of xi
tm_neg = @(x,xi) 0.5*(tanh(-(x-xi)*pi/tfact)+1); %1 on negative side of xi
%tm_0 = @(x,xi) tanh(-(x-xi)*pi/tfact); %+1 on neg side, -1 on pos side of xi
dtm_pos = @(x,xi)  0.5*pi/tfact*(1-tanh( (x-xi)*pi/tfact).^2);
dtm_neg = @(x,xi) -0.5*pi/tfact*(1-tanh(-(x-xi)*pi/tfact).^2);

%K = Kold;
%K = Kold.*[tm_pos(x(1),xi);1];
%barrier = rfactnew.*[tm_neg(x(1),x0);tm_neg(x(1),x0)*tm_0(x(2),y0)];

%the new dynamics being added in
barrier = rfactnew.*[tm_neg(x(1),x0);0];
dbarrier = [rfactnew,rfactnew].*[dtm_neg(x(1),x0),0;0,0];

%the filter to preserve wanted dynamics
gfilt = [tm_pos(x(1),x0);tm_pos(x(1),x0)];
dgfilt = [dtm_pos(x(1),x0),0;dtm_pos(x(1),x0),0];


K = diag(gfilt)*Kold + barrier;

dK = diag(gfilt)*dKold + diag(Kold)*dgfilt + dbarrier;
