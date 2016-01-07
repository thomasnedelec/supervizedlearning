function K = generateKernelMatrix (X,sigma)
    nData = size(X,1);
    K = zeros(nData, nData);
    for i=1:nData
        for j=1:nData
             K(i,j)=exp(-norm(X(i,:)-X(j,:))^2/(2*sigma^2));
        end
    end
            