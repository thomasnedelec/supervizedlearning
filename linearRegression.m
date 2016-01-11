function [MSEtrain, MSEtest] = linearRegression (trainX, testX, trainY, testY)

toPlot = 0;

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

%MSETrain = computeMSE(wTrained,trainX,trainY);
MSEtest = computeMSE(wTrained,testX,testY);
MSEtrain = computeMSE(wTrained,trainX,trainY);
        
