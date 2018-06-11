function [x1,nuc1,x2,nuc2,spoints] = koulakov_nullcline(x,n,theta,L,B,a)

ns = numel(x);
u = @(x) -a*sin((n-1)*pi/L*x+pi); %nullcline function 1
fx =  (u(x)+1)/2;
fx2 = (-u(x)+1)/2;

x1 = zeros(1,ns);
x2 = zeros(1,ns);
nuc1 = zeros(1,ns);
nuc2 = zeros(1,ns);

A = [cos(theta),-sin(theta);sin(theta),cos(theta)];
%B = [0.2;-0.1];
%B = [0;0];
R0 = [L/2;1]; %point around which to roate

for j = 1:ns,
    f = A*([x(j);fx(j)]-R0)+R0-B;
    f2 = A*([x(j);fx2(j)]-R0)+R0-B;
    
    %f2 = [x(j);fx2(j)];
    x1(j) = f(1); nuc1(j) = f(2);
    x2(j) = f2(1); nuc2(j) = f2(2);
    
end

nucx = linspace(0,2*L,2*n-1);
nucy = 0.5*ones(size(nucx));
spoints = zeros(2,numel(nucx));

for j = 1:numel(nucx),
    ntemp = A*([nucx(j);nucy(j)]-R0)+R0-B;
    spoints(:,j) = ntemp;
end

%plot(xnew,fxnew,xnew,fxnew2,'k')
plot(x1,nuc1,'k')
hold on
plot(x2,nuc2,'k')

%only grab ones in the figure limits
spoints = spoints(:,spoints(1,:) < 1.01 & spoints(1,:) > 0.01 & spoints(2,:) < 1.01 & spoints(2,:) > 0.01);
%now plot these mother fucking nullclines
scatter(spoints(1,1:2:end),spoints(2,1:2:end),30,[0,0,0],'filled');
%text(spoints(1,2:2:end)-0.02,spoints(2,2:2:end),'X','fontsize',15);
plot(spoints(1,2:2:end),spoints(2,2:2:end),'kd','markersize',10);
xlim([0,1]);ylim([0;1])
%0.05 added for spacing

