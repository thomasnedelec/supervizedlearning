clear all;
clc;
close all;

nbValueGamma=15;
vectorGamma=zeros(1,nbValueGamma);
for i=1:nbValueGamma
    vectorGamma(1,i) = 2^(i-41);
end

nbValueSigma=13;
vectorSigma=zeros(1,nbValueSigma);
for i=1:nbValueSigma
    6.5+0.5*i
    vectorSigma(1,i) = 2^(6.5+0.5*i);
end

load('boston.mat');
X=boston(:,1:13);
Y=boston(:,14);

sizeOfData = size(X,1);
nTrainPoints = round(2/3 * sizeOfData);
trainX = X(1:nTrainPoints,:);
trainY = Y(1:nTrainPoints,:);
testX = X(nTrainPoints+1:sizeOfData,:);
testY = Y(nTrainPoints+1:sizeOfData,:);

%implementation of cross-validation
k = 5;
[xSetsTrain, ySetsTrain, xSetsValidation, ySetsValidation] = kFoldCrossValidation(trainX,trainY,k);
%generate the Kernel Matrix
for i=1:nbValueSigma
    K=generateKernelMatrix(trainX,testX,vectorSigma(1,i));
    for j=1:nbValueGamma
        for fold = 1 : k
            smallerTrainX = xSetsTrain(:,:,fold);
            smallerTrainY = ySetsTrain(:,fold);
            validationX = xSetsValidation(:,:,fold);
            validationY = ySetsValidation(:,fold);
                
            alpha=kridgereg(K,vectorGamma(1,j),smallerTrainY,nTrainPoints,k,fold);
            %mseValidationAverage=mseValidationAverage+dualcost(K,alpha,validationY,k,fold);
        end;
    end
end


                