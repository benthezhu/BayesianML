clear;
close all

load data.mat
[d, N] = size(X);
Y = X';
labels = ones(N,1);

m_0 = mean(Y)';%2*1
c_0 = 0.1;
a_0 = d;
A_0 = cov(X(1,:),X(2,:));
B_0 = c_0*d.*A_0;
alpha = 1;

K = 1;

T = 1;
for t = 1:T
s = zeros(N,1);%size of cluster j
sumx = zeros(d,1);
m = zeros(d,1);
c = zeros(N,1);
a = zeros(N,1);
B = zeros(d, d, N);
sigma = zeros(d, d, N);
fi = zeros(N,1);
fi_new = zeros(N,1);
mu = zeros(d,1);
for j = 1:K
    n = zeros(N,1);
    for ii = 1:N%calculate n(i) & s(j)
        if(labels(ii) == j)
            s(j) = s(j) + 1;%how many numbers in cluster j
        end
        n(j) = s(j) - 1;
        if (labels(ii) == j)
            sumx = sumx + X(:,ii);
        end
    end%calculate n(i)

    x_b = sumx./s(j);%d*1
    m(:,j) = c_0/(c_0 + s(j)).*m_0 + 1/(c_0 + s(j)).*sumx;
    c(j) = s(j) + c_0;
    a(j) = s(j) + a_0;
    sumxsxb = zeros(d, d, N);

    for ii = 1:N
        if (labels(ii) == j)
            sumxsxb(:,:,j) = sumxsxb(:,:,j)  + (X(:,ii)-x_b)*(X(:,ii)-x_b)';
        end
    end
    B(:,:,j) = B_0 + sumxsxb(:,:,j)  + s(j)/(a(j)*s(j)+1).*(x_b - m(:,j))*(x_b - m(:,j))';

    sigma(:,:,j) = wishrnd(B(:,:,j),a(j));
    mu(:,j) = mvnrnd(m(:,j), pinv(c(j).*sigma(:,:,j)));
end


for i = 1:3
for j = 1:K
    %calculate phi for current cluster
    fi(j) = mvnpdf(X(:,i), mu(:,j), pinv(sigma(:,:,j))) .* (n(j)/(alpha + N -1));
    %calculate phi for a new cluster
    part1 = c_0/(pi*(1+c_0))^(d/2);
    part2 = det(B_0 + (c_0/(1+c_0).*(X(:,i)-m_0)*(X(:,i)-m_0)'))^(-0.5*(a_0+1));
    part3 = det(B_0)^(-0.5*a_0);
    part4 = exp(gammaln((a_0+1)/2) + gammaln(a_0/2) - gammaln(a_0/2) - gammaln((a_0-1)/2));
    fi_new(j) = alpha/(alpha + N - 1)*(part1 * part2/part3 * part4);

    fi_oldd = fi(j)/(fi(j) + fi_new(j));
    fi_neww = fi_new(j)/(fi(j) + fi_new(j));
end
    class = discretesample([fi_oldd, fi_neww], 1);
    if (class == 2)
        K = K + 1;
        %[labels, mu] = kmeans(X', K);%random sample
        labels(i) = K;%put i in the new cluster
        
        x_b = X(:,i);%d*1
        sumx = X(:,i);%d*1
        m(:,K) = c_0/(c_0 + 1).*m_0 + 1/(c_0 + 1).*sumx;
        c(K) = 1 + c_0;
        a(K) = 1 + a_0; 
        sumxsxb(:,:,K) = (X(:,i)-x_b)*(X(:,i)-x_b)';
        
        B(:,:,K) = B_0 + sumxsxb(:,:,K)  + 1/(a(K)+1).*(x_b - m(:,K))*(x_b - m(:,K))';

        sigma(:,:,K) = wishrnd(B(:,:,K),a(K));
        mu(:,K) = mvnrnd(m(:,K), pinv(c(K).*sigma(:,:,K)));
    end  
    %end
end
end


