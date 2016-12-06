clear;
close all

load data.mat
[d, N] = size(X);
K = 25;
T = 100;
t1 = zeros(K,1);
t2 = zeros(K,1);
t3 = zeros(K,1);
t4 = zeros(K,1);
alpha_0 = ones(K,1);
alpha = ones(K,1);
c = 10;
a = d.*ones(K,1);
a_0 = d.*ones(K,1);
A = cov(X(1,:),X(2,:));%d*d
B_0 = d/10.*A;%d*d
B = zeros(d, d, K);%d*d*K
sigma = zeros(d, d, K);
n = zeros(K, 1);
fi = zeros(N, K);
fi_mul_x = zeros(d, 1);

[labels, mu] = kmeans(X', K);% mu:K*d
m = mu';%d*K
%m = zeros(d, K);
Y = X';

for j = 1:K
    B(:,:,j) = B_0;
    sigma(:,:,j) = cov(Y(labels == j,:));
    %sigma(:,:,j) = [0.9, 0.4;0.4, 0.3];
end

L = zeros(T,1);
for t = 1:T
    for j = 1:K
        for i = 1:N
            psia = 0;
            for k = 1:d
                psia = psia + psi((1 - k + a(j))/2);
            end
            t1(j) = psia - log(det(B(:,:,j)));
            t2(j) = (X(:,i) - m(:,j))'*(a(j).*pinv(B(:,:,j)))*(X(:,i) - m(:,j));
            t3(j) = trace(a(j).*pinv(B(:,:,j))*sigma(:,:,j));
            t4(j) = psi(alpha(j)) - psi(sum(alpha));
            fi(i,j) = exp(0.5*t1(j) - 0.5*t2(j) - 0.5*t3(j) + t4(j));
        end  
    end
    fi = fi./repmat(sum(fi,2),1,K);  
    for j = 1:K       
        sumfi = sum(fi, 1);
        n(j) = sumfi(j);
        alpha(j) = alpha_0(j) + n(j);%.*ones(K,1);
        sigma(:,:,j) = pinv(1/c.*eye(d) + n(j)*a(j).*pinv(B(:,:,j)));
        summ = zeros(d ,1);
        for i = 1:N
            fi_mul_x = fi(i,j).* X(:,i);%d*1
            summ = fi_mul_x + summ;
        end
        m(:,j) = sigma(:,:,j) * (a(j)*pinv(B(:,:,j))*summ);
        a(j) = a_0(j) + n(j);
        
        sumfixx = zeros(d ,d);
        for i = 1:N
            mul = fi(i,j).* ((X(:,i)-m(:,j))*(X(:,i)-m(:,j))'+sigma(:,:,j));
            sumfixx = sumfixx + mul;
        end      
        B(:,:,j) = B_0 + sumfixx;
    end
    
    ga = 1;
    for j = 1:K
        ga = ga + gammaln(alpha(j));
    end
    Elnxcml = 0;
    Elnc = 0;
    Elnpm = 0;
    Elnpt = 0;
    Elnqc = 0;
    Elnqm = 0;
    Elnql = 0;
    Elnqpi = 0;
    for j = 1:K
        for i = 1:N   
            Elnxcml = Elnxcml + fi(i,j)*(0.5*t1(j) - 0.5*t2(j) - 0.5*t3(j) + t4(j));
            Elnc = Elnc + fi(i,j)*t4(j);
            Elnqc = Elnqc + fi(i,j)*log(fi(i,j));
        end
        Elnpm = Elnpm - c/2*trace(sigma(:,:,j) + m(:,j)*m(:,j)');
        Elnpt = Elnpt + (a_0(j)-d-1)/2*t1(j) + 0.5*trace(B_0*a(j).*pinv(B(:,:,j)));
        Elnqm = Elnqm + 0.5*log(det(pinv(sigma(:,:,j))));
        Elnql = Elnql - 0.5*log(2)*a(j)*d - log(gamma(0.5*(a(j)+1))) - log(gamma(0.5*a(j))) + 0.5*a(j)*log(det(B(:,:,j))) + 0.5*(a(j)-d+1)*t1(j) - 0.5*trace(B(:,:,j)*a(j)*pinv(B(:,:,j)));
        
        Elnqpi = Elnqpi + (alpha(j) - 1)*t4(j);

    end
    
    L(t) = Elnxcml + Elnc + Elnpm + Elnpt - Elnqc - Elnqm - Elnql - Elnqpi + log(exp(gammaln(sum(alpha))-ga));
end

plot(L);

fi_t = fi';
[maxnum,I] = max(fi_t);
color = ['r','g','b','c','m','r','g','b','c','m','r','g','b','c','m','r','g','b','c','m','r','g','b','c','m'];
dot = ['.','.','.','.','.','*','*','*','*','*','x','x','x','x','x','s','s','s','s','s','s','>','>','>','>','>'];

figure;
for i=1:N
    plot(X(1,i),X(2,i),[color(I(i)) dot(I(i))])
    hold on
end
plot(m(1,:),m(2,:),['k','x'],'LineWidth',4)