%Bayesian Model Machine Learning
%Homework 3

load data3.mat
%Initialize all variables
[N, d] = size(X);
%500 iterations
t = 500;
mu = zeros(d, 1);
sigma = zeros(d, d);
a = zeros(d, 1);
b = zeros(d, 1);
akt = zeros(d, 1);
b0t = zeros(d, 1);
L = zeros(t, 1);
e0 = 1;
e = e0;
f0 = 1;
f = f0;

a0 = 10^(-16);
b0 = 10^(-16);
for i = 1:d
    b0t(i) = b0;
    a(i) = a0;
    b(i) = b0;
end
%All variables are initiated
%Initialize matrices
x1 = zeros(d, d);
x2 = zeros(d, 1);
for i = 1:N
    x1 = x1 + X(i,:)'*X(i,:);
    x2 = x2 + X(i,:)'*y(i);
end

Eqalpha = zeros(d,1);
%Time to iterate through
%Need to define p(alpha) p(lambda)p(w) and the qs as well
for i = 1:t
    sigma = pinv(diag(a./b) + e/f*x1);
    mu = sigma*(e/f)*x2;
    
    
    yxitmu = 0;
    for l = 1:N
        yxitmu = yxitmu + (y(l) - X(l,:)*mu)^2 + X(l,:)*sigma*X(l,:)';
    end
    for j = 1:d
        a(j) = a0+0.5;
    end
    e = e0 + N/2;
    f = f0 + 0.5*yxitmu;
    mumusigma = mu*mu' + sigma;
    
    Elnpw = 0;
    Elnpalpha = 0;
    Elnqalpha = 0; 

    for k = 1:d
        b(k) = b0t(k) + 0.5*mumusigma(k,k);
        Elnpw = Elnpw + 0.5*(psi(a(k))-log(b(k))) - 0.5*a(k)./b(k)*mumusigma(k,k);
        Elnpalpha = Elnpalpha + (a0 - 1)*(psi(a(k))-log(b(k))) - b0*a(k)/b(k);
        Elnqalpha = Elnqalpha + log(gamma(a(k))) + (1 - a(k))*psi(a(k)) + a(k) - log(b(k));
        Eqalpha(k) = a(k)./b(k);
    end
    
    
    Elnpy = N/2*(psi(e)-log(f)) - 0.5*e/f*yxitmu;
    Elnplambda = (e0 - 1)*(psi(e)-log(f)) - f0*e/f;
    Elnqlambda = e - log(f) + (1 - e)*psi(e) +  gammaln(e);
    Elnqw = 0.5*log(det(sigma));
    
    
    
    L(i) = Elnpy + Elnpw + Elnpalpha + Elnplambda + Elnqw + Elnqalpha + Elnqlambda;
    
end
figure;
plot(L);
figure;
stem(1./Eqalpha);
Eqlambda = e/f;
disp(1/Eqlambda);
y_hat = X*mu;
figure;
plot(z, y_hat, z,y,'-', z, 10*sinc(z));

