clear all;
clc;
close all;
toPlot = 0;
nbValueGamma=10;
vectorGamma=zeros(1,nbValueGamma);
nLoops = 200;
MSETrain=zeros(1,nbValueGamma);
MSETest=zeros(1,nbValueGamma);
averageTestError = zeros(1,nbValueGamma);
averageTrainError = zeros(1,nbValueGamma);

for seed = 1:nLoops
    gamma=10^-6;
    for i =1:nbValueGamma
        s = RandStream('mt19937ar','Seed',seed);
        RandStream.setGlobalStream(s);

        %variables
        nData = 600;
        dimension = 10;
        nTrainPoints = 100;
    
        X = randn(nData,dimension);
        N = randn(nData,1);
        w = randn(dimension,1);
        Y = X*w + N; 
  
        trainX = X(1:nTrainPoints,:);
        trainY = Y(1:nTrainPoints,:);

        testX = X(100:nData,:);
        testY = Y(100:nData,:);
    
        wTrained=(transpose(X)*X+gamma*nTrainPoints*eye(dimension))\(transpose(X)*Y);
    
        MSETrain(1,i) = computeMSE(wTrained,trainX,trainY);
        MSETest(1,i) = computeMSE(wTrained,testX,testY);
        
        vectorGamma(1,i)=gamma;
        gamma=10*gamma;
        averageTestError(1,i) = averageTestError(1,i) + MSETest(1,i);
        averageTrainError(1,i) = averageTrainError(1,i) + MSETrain(1,i);
    end;
    if(toPlot)

        figure
        semilogx(vectorGamma,MSETrain, '-b')
        hold on;
        semilogx(vectorGamma, MSETest, '-g')
    end;
    
end;
averageTestError=averageTestError./nLoops;
averageTrainError=averageTrainError./nLoops;
figure(2)
semilogx(vectorGamma,averageTestError, '-b')
hold on;
semilogx(vectorGamma, averageTrainError, '-g')


