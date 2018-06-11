function Jvec = loss_gaussian(N,xest,PIest,F,G,c)
%funciton to calculate expected loss E_x( (F[x,c]-G[x])^(F[x,c]-G[x]) ) 



ns = size(xest,2);
Jvec = zeros(ns,1);

for j = 1:ns,
    %draw N samples from distribution N(x,sqrt(PI))
    %just a assume a 2x2 for now, due to anger
    Sigma = PIest(:,:,j);
    Sigma(1,2) = Sigma(2,1);
    x_samp = mvnrnd(xest(:,j),Sigma,N);
    for k = 1:N,
        Jvec(j) = Jvec(j)+...
            norm(F(x_samp(k,:),c(:,j))-G(x_samp(k,:)))^2;
    end
    Jvec(j) = Jvec(j)/N;
    
end
    