clear all;
clc;
close all;
toPlot = 0;
averageTestError = 0;
averageTrainError = 0;
nLoops = 200;
nTableTrainPoints=[10,100];
resultsMatrix=zeros(2);
for i=1:2
    for seed = 1:nLoops
        s = RandStream('mt19937ar','Seed',seed);
        RandStream.setGlobalStream(s);

        %variables
        nData = 600;
        dimension = 10;
        nTrainPoints = nTableTrainPoints(1,i);

        X = randn(nData,dimension);
        N = randn(nData,1);
        w = randn(dimension,1);
        Y = X*w + N; 

        trainX = X(1:nTrainPoints,:);
        trainY = Y(1:nTrainPoints,:);

        testX = X(100:nData,:);
        testY = Y(100:nData,:);

        wTrained = trainX\trainY;
        if(toPlot)
            labelX = -5:5;
            labelY = -6:6;
            figure
            plot(trainX,trainY, 'or')
            hold on;
            plot(labelX, wTrained*labelX, '-g')

            figure
            plot(testX,testY, 'or')
            hold on;
            plot(labelX, wTrained*labelX, '-g')
        end

        MSETrain = computeMSE(wTrained,trainX,trainY);
        MSETest = computeMSE(wTrained,testX,testY);
        averageTestError = averageTestError + MSETest;
        averageTrainError = averageTrainError + MSETrain;
    end

    averageTestError = averageTestError / nLoops;
    averageTrainError = averageTrainError / nLoops;
    
    resultsMatrix(i,1)=averageTrainError;
    resultsMatrix(i,2)=averageTestError;

end
resultsMatrix