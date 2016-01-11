clear all;
clc;
close all;
toPlot = 0;
averageTestError = 0;
averageTrainError = 0;
nTableTrainPoints=[10,100];
resultsMatrix=zeros(2);
load('boston.mat');
X=boston(:,1:13);
Y=boston(:,14);
nLoop=20
MSEresultsTrain=zeros(nLoop,15);
MSEresultsTest=zeros(nLoop,15);
for i=1:nLoop
    seed = i;
    s = RandStream('mt19937ar','Seed',seed);
    RandStream.setGlobalStream(s);
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
    [MSEresultsTrain(i,1), MSEresultsTest(i,1)] =linearRegression(naiveOnesTrain,naiveOnesTest,trainY,testY);
    
    % implementation of linear regression withonefeature
    
    for j =1:13
        featureTrain=trainX(:,j);
        bias = ones(size(featureTrain));
        featureTrain = [featureTrain, bias];
        
        featureTest=testX(:,j);
        bias2 = ones(size(featureTest));
        featureTest = [featureTest, bias2];
        
        [MSEresultsTrain(i,j+1), MSEresultsTest(i,j+1)]=linearRegression(featureTrain,featureTest,trainY,testY);
    end
    
    % implementation of linear regression using all atributes
    
    [MSEresultsTrain(i,15), MSEresultsTest(i,15)]=linearRegression(trainX,testX,trainY,testY);

end
meanTrainError = mean(MSEresultsTrain);
stdTrainError = std(MSEresultsTrain);
meanTestError = mean(MSEresultsTest);
stdTestError = std(MSEresultsTest);
