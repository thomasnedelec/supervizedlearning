clear all;
clc;
close all;

nLoops = 200;
nbValueGamma=10;
vectorGamma=zeros(1,nbValueGamma);
dimension = 10;
nTableTrainPoints=[10,100];

MSETrain=zeros(2,nbValueGamma);
MSEValidation=zeros(2,nbValueGamma);
MSECrossValidation = zeros(2,nbValueGamma);

testErrorTrain = zeros(nLoops,2);
testErrorValidation = zeros(nLoops,2);
testErrorCrossValidation = zeros(nLoops,2);
for j=1:2
    for seed = 1:nLoops
        gamma=10^-6;
        for i =1:nbValueGamma
            s = RandStream('mt19937ar','Seed',seed);
            RandStream.setGlobalStream(s);

            %variables
            nData = 600;
            nTrainPoints = 100;

            X = randn(nData,dimension);
            N = randn(nData,1);
            w = randn(dimension,1);
            Y = X*w + N; 

            trainX = X(1:nTableTrainPoints(1,j),:);
            trainY = Y(1:nTableTrainPoints(1,j),:);

            testX = X(nTableTrainPoints(1,j)+1:nData,:);
            testY = Y(nTableTrainPoints(1,j)+1:nData,:);

            % minimizing the training error
            wTrained=(transpose(trainX)*trainX+gamma*nTrainPoints*eye(dimension))\(transpose(trainX)*trainY);
            MSETrain(1,i) = computeMSE(wTrained,trainX,trainY);

            %minimizing validation error
            percentage=0.8;
            smallerTrainX = X(1:percentage*nTableTrainPoints(1,j),:);
            smallerTrainY = Y(1:percentage*nTableTrainPoints(1,j),:);

            validationX= X(percentage*nTableTrainPoints(1,j)+1:nTableTrainPoints(1,j),:);
            validationY= Y(percentage*nTableTrainPoints(1,j)+1:nTableTrainPoints(1,j),:);

            wTrained(:,i)=(transpose(smallerTrainX)*smallerTrainX+gamma*percentage*nTrainPoints*eye(dimension))\(transpose(smallerTrainX)*smallerTrainY);
            MSEValidation(j,i) = computeMSE(wTrained(:,i),validationX,validationY);


            %minimizing the 5-fold cross validation error 
            k = 5;
            [xSetsTrain, ySetsTrain, xSetsValidation, ySetsValidation] = kFoldCrossValidation(trainX,trainY,k);

            mseValidationerror = 0;
            for fold = 1 : k
                smallerTrainX = xSetsTrain(:,:,fold);
                smallerTrainY = ySetsTrain(:,fold);
                validationX = xSetsValidation(:,:,fold);
                validationY = ySetsValidation(:,fold);
                percentage = (k-1) / k;
                wTrained=(transpose(smallerTrainX)*smallerTrainX+gamma*percentage*nTrainPoints*eye(dimension))\(transpose(smallerTrainX)*smallerTrainY);

                mseValidationerror = mseValidationerror + computeMSE(wTrained,validationX,validationY);
            end
            mseValidationerror = mseValidationerror / k;

            MSECrossValidation(j,i) = mseValidationerror;

            vectorGamma(1,i)=gamma;
            gamma=10*gamma;           
        end;
        %%
        [minimumTrain, indexTrain]=min(MSETrain(j,:));
        wTrained=(transpose(trainX)*trainX+vectorGamma(indexTrain)*nTrainPoints*eye(dimension))\(transpose(trainX)*trainY);
        testErrorTrain(seed,j) = computeMSE(wTrained, testX, testY)
            
        %%
        [minimumValidation, indexValidation]=min(MSEValidation(j,:));
        wTrained=(transpose(trainX)*trainX+vectorGamma(indexValidation)*nTrainPoints*eye(dimension))\(transpose(trainX)*trainY);
        testErrorValidation(seed,j) = computeMSE(wTrained, testX, testY)
        
        %%
        [minimumCrossValidation, indexCrossValidation]=min(MSECrossValidation(j,:));
        wTrained=(transpose(trainX)*trainX+vectorGamma(indexCrossValidation)*nTrainPoints*eye(dimension))\(transpose(trainX)*trainY);
        testErrorCrossValidation(seed,j) = computeMSE(wTrained, testX, testY)
    end;
end;


train_mean_10 = mean(testErrorTrain(:,1))
valid_mean_10 = mean(testErrorValidation(:,1))
cross_mean_10 = mean(testErrorCrossValidation(:,1))
train_std_10 = std(testErrorTrain(:,1))
valid_std_10 = std(testErrorValidation(:,1))
cross_std_10 = std(testErrorCrossValidation(:,1))


train_mean_100 = mean(testErrorTrain(:,2))
valid_mean_100 = mean(testErrorValidation(:,2))
cross_mean_100 = mean(testErrorCrossValidation(:,2))
train_std_100 = std(testErrorTrain(:,2))
valid_std_100 = std(testErrorValidation(:,2))
cross_std_100 = std(testErrorCrossValidation(:,2))
