function problem10;

clear all;
clc;
close all;

nbdimensions=30;

indexSamples=zeros(1,nbdimensions);
for n=10:nbdimensions+9
    m=5;
    generalizationError=1;
    while generalizationError > 0.1 && m < 50
        p=rand(1,m);
        X=zeros(m,n);
        for j=1:m
            for k=1:n
                X(j,k)=binornd(1,p(1,j));
                if X(j,k)==0
                    X(j,k)=-1;
                end
            end
        end
        Y=X(:,1);
        %generalizationError=perceptron(X,Y,m,n); 
        %generalizationError=NNeigh(X,Y,m,n);
        generalizationError=leastSquare(X,Y,m,n);
        generalizationError=winnow(X,Y,m,n);
        m=m+1;
    end
    indexSamples(1,n-9)=m;
end
size(indexSamples)
size(linspace(1,nbdimensions,nbdimensions))
plot(linspace(10,nbdimensions+10,nbdimensions),indexSamples)


function generalizationError=perceptron(X,Y,m,n)
    weight=zeros(1,n);
    nbiter=10;
    for i=1:nbiter
        for j=1:size(X,1)
            if (Y(j)*(weight*transpose(X(j,:)))<=0)
                weight=weight+Y(j)*X(j,:);
            end
        end
    end
    
    nbDataSets=10;
    generalizationError=0;
    for k=1:nbDataSets
        p=rand(1,m);
        Xnew=zeros(m,n);
        for j=1:m
            for l=1:n
                Xnew(j,l)=binornd(1,p(1,j));
                if Xnew(j,l)==0
                    Xnew(j,l)=-1;
                end
                Ynew=Xnew(:,1);
            end
        end
        for j=1:m
            if (Ynew(j)*(weight*transpose(Xnew(j,:)))<=0)
                generalizationError=generalizationError+1;
            end
        end
    end
    generalizationError=generalizationError/(nbDataSets*m);          

function generalizationError=leastSquare(X,Y,m,n)
    weight = pinv(X) * Y;
    nbDataSets=10;
    generalizationError=0;
    for k=1:nbDataSets
        p=rand(1,m);
        Xnew=zeros(m,n);
        for j=1:m
            for l=1:n
                Xnew(j,l)=binornd(1,p(1,j));
                if Xnew(j,l)==0
                    Xnew(j,l)=-1;
                end
                Ynew=Xnew(:,1);
            end
        end
        for j=1:m
            if (Ynew(j)*(transpose(weight)*transpose(Xnew(j,:)))<=0)
                generalizationError=generalizationError+1;
            end
        end
    end
    generalizationError=generalizationError/(nbDataSets*m);          

    
function generalizationError=NNeigh(X,Y,m,n)
    nbDataSets=10;
    generalizationError=0;
    for k=1:nbDataSets
        p=rand(1,m);
        Xnew=zeros(m,n);
        for j=1:m
            for k=1:n
                Xnew(j,k)=binornd(1,p(1,j));
                if Xnew(j,k)==0
                    Xnew(j,k)=-1;
                end
                Ynew=Xnew(:,1);
            end
        end
        for j=1:m
            indexminDistance=1;
            minDistance=(Xnew(j,:)-X(1,:))*transpose(Xnew(j,:)-X(1,:));
            for l=2:m
                 if (Xnew(j,:)-X(l,:))*transpose(Xnew(j,:)-X(l,:))<minDistance
                     indexminDistance=l;
                 end
            end
            if (Ynew(j)~=Y(indexminDistance))
                generalizationError=generalizationError+1;
            end
        end
    end
    generalizationError=generalizationError/(nbDataSets*m);
    
function generalizationError=winnow(X,Y,m,n)
    for i=1:m
        for j=1:n
            if X(i,j)==-1
                X(i,j)=0;
            end
        end
    end
    Y=X(:,1);
    weight=ones(1,n);
    niter=2;
    for k=1:niter
        for i=1:m
            if (weight*transpose(X(i,:))>n/2)&&(Y(i)==0)
                for j=1:n
                    if(X(i,j)==1)
                        weight(j)=weight(j)/2;
                    end
                end
            if (weight*transpose(X(i,:))<n/2)&&(Y(i)==1)
                 for j=1:n
                    if(X(i,j)==1)
                        weight(j)=weight(j)*2;
                    end
                 end
            end
            end
        end
    end
    nbDataSets=10;
    generalizationError=0;
    for k=1:nbDataSets
        p=rand(1,m);
        Xnew=zeros(m,n);
        for j=1:m
            for l=1:n
                Xnew(j,l)=binornd(1,p(1,j));
            end
        end
        Ynew=Xnew(:,1);
        for i=1:m
            if (weight*transpose(Xnew(i,:))>n/2)&&(Ynew(i)==0)||(weight*transpose(Xnew(i,:))<n/2)&&(Ynew(i)==1)
                generalizationError=generalizationError+1;
            end
        end
    end
    generalizationError=generalizationError/(nbDataSets*m);
