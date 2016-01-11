clear all;
clc;
close all;

toPlot = 0;
vectorGamma = 2.^(-40:-26);
vectorSigma = 2.^(7:0.5:13)

load('boston.mat');
X=boston(:,1:13);
Y=boston(:,14);

sizeOfData = size(boston, 1);
nTrainPoints = round(2/3 * sizeOfData);

Iter = 20;
trainError = zeros(1,Iter);
testError = zeros(1,Iter);

%%repeat proccess 20 times
for step = 1 : Iter
    seed = step;
    s = RandStream('mt19937ar','Seed',seed);
    RandStream.setGlobalStream(s);
    %shuffle indices
    indices = randperm(sizeOfData);
    trainInd = indices(1:nTrainPoints);
    testInd = indices(nTrainPoints+1:end);

    trainX = X(trainInd, :);
    trainY = Y(trainInd, :);

    testX = X(testInd, :);
    testY = Y(testInd, :);

    k = 5; % 5 cross validation
    MSEerror = zeros(size(vectorGamma, 2), size(vectorSigma, 2));

    for i=1:size(vectorSigma,2)
        KAll = generateKernelMatrix(X,vectorSigma(i));
        KTrainAll = KAll(trainInd, trainInd);
        for j=1:size(vectorGamma, 2)
            crossValidIndices = crossvalind('Kfold', size(trainX,1), k);
            for fold = 1 : 5
                validIndeces = (crossValidIndices == fold);
                trainIndeces = ~validIndeces;

                smallerTrainY = trainY(trainIndeces,:);
                validationY = trainY(validIndeces,:);

                KTrain = KTrainAll(trainIndeces, trainIndeces);
                KValidation = KTrainAll(validIndeces, trainIndeces);

                alpha=kridgereg(KTrain,vectorGamma(1,j),smallerTrainY);
                MSEerror(j,i) = MSEerror(j,i) + dualcost(KValidation, alpha,validationY );
            end;
        end
    end
    MSEerror = MSEerror ./ k;
   
    [~, index] = min(MSEerror(:));
    [gammaIndex, sigmaIndex] = ind2sub(size(MSEerror), index);

    KAll = generateKernelMatrix(X,vectorSigma(sigmaIndex));
    KTrain = KAll(trainInd, trainInd);
    alpha = kridgereg(KTrain,vectorGamma(gammaIndex), trainY );
    trainError(1,step) = dualcost(KTrain,alpha, trainY ); 

    Ktest = KAll(testInd, trainInd);
    testError(1,step) = dualcost(Ktest,alpha, testY);
    if (toPlot)
        mesh(log(vectorSigma), log(vectorGamma), MSEerror);
        zlim([0 50])
        ylabel('log(gamma)');
        xlabel('log(sigma)');
        zlabel('MSE');
    end
end

meanTrainError = mean(trainError)
stdTrainError = std(trainError)
meanTestError = mean(testError)
stdTestError = std(testError)
