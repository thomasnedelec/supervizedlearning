function K = generateKernelMatrix (trainX,testX,sigma)
    [n1,d1]=size(trainX);
    [n2,d2]=size(testX);
    sizeDataSet=n1+n2;
    X=zeros(sizeDataSet,d1);
    X(1:n1,:)=trainX(:,:);
    X(n1+1:sizeDataSet,:)=testX(:,:);
    K=zeros(sizeDataSet,sizeDataSet);
    for i=1:sizeDataSet
        for j=1:sizeDataSet-i
            %1st approach
            norme=0;
            for l=1:d1
               norme=norme+(X(i,l)-X(j,l))^2;
            end
            K(i,j)=exp(-norme/(2*sigma^2));
            K(j,i)=K(i,j);
            
            %2nd approach (why it gives different reauslts?)
%             K(i,j)=exp(-norm(X(i,l)-X(j,l))/(2*sigma^2));
%             K(j,i)=K(i,j);
        end
    end
            