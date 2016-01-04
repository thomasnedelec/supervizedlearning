clear all;
clc;
close all;

nbValueGamma=14;
vectorGamma=zeros(1,nbValueGamma);
for i=1:nbValueGamma
    vectorGamma=2^(i-41);
end
nbValueSigma=12;
vectorSigma=zeros(1,nbValueSigma);
for i=1:nbValueSigma
    vectorSigma=2^(6.5+0.5*i);
end
dimension = 10;

nSplits=20;
k=5;

MSETrainAverage=0;
MSETestAverage=0;
for j=1:nSplits
    seed = 20
    gamma=10^-6;
    s = RandStream('mt19937ar','Seed',seed);
    RandStream.setGlobalStream(s);

    %variables
    nData = 20;
    nTrainPoints = 8;
    
    X = randn(nData,dimension);
    N = randn(nData,1);
    w = randn(dimension,1);
    Y = X*w + N; 
  
    trainX = X(1:nTrainPoints(1,j),:);
    trainY = Y(1:nTrainPoints(1,j),:);

    testX = X(nTrainPoints(1,j)+1:nData,:);
    testY = Y(nTrainPoints(1,j)+1:nData,:);

    %implementation of cross-validation
    [xSetsTrain, ySetsTrain, xSetsValidation, ySetsValidation] = kFoldCrossValidation(trainX,trainY,k);
    %generate the Kernel Matrix
    MSEGammaSigma=zeros(nbValueSigma,nbValueGamma);
    for i=1:nbValueSigma
        K=generateKernelMatrix(trainX,trainY,testX,testY,vectorSigma(1,i));
        for j=1:nbValueGamma
            mseValidationAverage = 0;
            for fold = 1 : k
                smallerTrainX = xSetsTrain(:,:,fold);
                smallerTrainY = ySetsTrain(:,fold);
                validationX = xSetsValidation(:,:,fold);
                validationY = ySetsValidation(:,fold);
                
                alpha=kridgereg(K,vectorGamma(1,j),smallerTrainY,k,fold);
                mseValidationAverage=mseValidationAverage+dualcost(K,alpha,validationY,k,fold);
            end;
            MSEGammaSigma(i,j)=mseValidationAverage/fold;
        end
    end
    [a,b]=min(MSEGammaSigma);
    K=generateKernelMatrix(trainX,trainY,testX,testY,vectorSigma(1,b(1,1)));

    alpha=kridgereg(K,vectorGamma(1,b(1,2)),trainX,1,1);
    
    MSETest=dualcost(K,alpha,testY,3,3);
    MSETrain1=dualcost(K,alpha,trainY,3,1);
    MSETrain2=dualcost(K,alpha,trainY,3,2);
    MSETrain=(MSETrain1+MSETrain2)/2;
    MSETestAverage=MSETestAverage+MSETest;
    MSETrainAverage=MSETrainAverage+MSETrain
end
MSETrainAverage=MSETrainAverage/nSplits;    
MSETestAverage=MSETestAverage/nSplits;

                
                