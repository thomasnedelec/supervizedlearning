clear all;
clc;
close all;
toPlot = 0;
averageTestError = 0;
averageTrainError = 0;
nLoops = 200;
nTableTrainPoints=[10,100];
resultsMatrix=zeros(2);
load('boston.mat');
X=boston(:,1:13);
Y=boston(:,14);
nLoop=1
MSEresults=zeros(nLoop,15);

for i=1:nLoop
    
    %generate training set and test set
    [n1,d1]=size(X)
    randomIndexes=randperm(n1);
    sizeTraining=int64(2*n1/3);
    trainX=X(randomIndexes(1:sizeTraining),:);
    trainY=Y(randomIndexes(1:sizeTraining),:);
    testX=X(randomIndexes(sizeTraining+1:n1),:);
    testY=Y(randomIndexes(sizeTraining+1:n1),:);
    
    % implementation of naive regression 
    
    naiveOnesTrain=ones(sizeTraining,1);
    naiveOnesTest=ones(n1-sizeTraining,1);
    MSEresults(i,1)=linearRegression(naiveOnesTrain,naiveOnesTest,trainY,testY);
    
    % implementation of linear regression withonefeature
    
    for j =1:13
        featureTrain=trainX(:,j)
        featureTest=testX(:,j)
        MSEresults(i,j+1)=linearRegression(featureTrain,featureTest,trainY,testY);
    end
    
    % implementation of linear regression using all atributes
    
    MSEresults(i,15)=linearRegression(trainX,testX,trainY,testY);

end
MSEresults
