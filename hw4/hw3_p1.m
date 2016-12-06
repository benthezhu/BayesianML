load data.mat
[d, N] = size(X);
T=100; %100 iterations
K = 2;
fimux = zeros(d,10);
fi = zeros(N,K);
nn = zeros(K,1);
sumfi = zeros(K,1);
pi_m = zeros(K,1);
sigma_m =zeros(d,d,K);
ft=zeros(T,1);


[labels, mu] = kmeans(X', K); %mu given K d
mu_m = mu';
Y = X';


for j=1:K
    sigma_m(:,:,j) = cov(Y(labels== j,:));
    pi_m(j)=sum(labels==j)/N;
end

for t= 1:T
    for j=1:K
        for i = 1:N
            nor = mvnpdf(X(:,i), mu_m(:,j),sigma_m(:,:,j));
            fi(i,j) = pi_m(j) .* nor;
        end
    end
    ft(t) = sum(log(sum(fi,2)));
    fi=fi./repmat(sum(fi,2),1,K);
    
    for j=1:K
        sumfi = sum(fi,1);
        nn(j) = sumfi(j);
        summ=zeros(d,1);
        for i = 1:N
            fimux = fi(i,j).* X(:,i);
            summ = fimux+ summ;
        end
        mu_m(:,j) = summ./nn(j);
        sumfixx = zeros(d,d);
        for i = 1:N
            mul = fi(i,j).*((X(:,i)-mu_m(:,j))*(X(:,i)-mu_m(:,j))');
            sumfixx= sumfixx+mul;
        end
        sigma_m(:,:,j) = sumfixx./nn(j);
        pi_m(j) = nn(j)/N;
    end
end

