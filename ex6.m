clear all;
clc;
close all;

nbValueGamma=10;
vectorGamma=zeros(1,nbValueGamma);
dimension = 10;
nTableTrainPoints=[10,100];

MSETrain=zeros(2,nbValueGamma);
MSETest=zeros(2,nbValueGamma);
MSEValidation=zeros(2,nbValueGamma);

for j=1:2
    seed = 20
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

        %implementation of cross-validation
        k = 5;
        [xSetsTrain, ySetsTrain, xSetsValidation, ySetsValidation] = kFoldCrossValidation(trainX,trainY,k);

        mseTrainAverage = 0;
        mseTestAverage = 0;
        mseValidation = 0;
        for fold = 1 : k
            smallerTrainX = xSetsTrain(:,:,fold);
            smallerTrainY = ySetsTrain(:,fold);
            validationX = xSetsValidation(:,:,fold);
            validationY = ySetsValidation(:,fold);
            percentage = (k-1) / k;
            wTrained=(transpose(smallerTrainX)*smallerTrainX+gamma*percentage*nTrainPoints*eye(dimension))\(transpose(smallerTrainX)*smallerTrainY);
   
            mseTrainAverage = mseTrainAverage + computeMSE(wTrained,smallerTrainX,smallerTrainY);
            mseTestAverage = mseTestAverage + computeMSE(wTrained,testX,testY);
            mseValidation = mseValidation + computeMSE(wTrained,validationX,validationY);
        end

         mseTrainAverage = mseTrainAverage / k;
         mseTestAverage = mseTestAverage / k;
         mseValidation = mseValidation / k;
            
         MSETrain(j,i) = mseTrainAverage;
         MSETest(j,i) = mseTestAverage;
            
         MSEValidation(j,i) = mseValidation;
            
         vectorGamma(1,i)=gamma;
         gamma=10*gamma;           
    end;
end;

figure()
semilogx(vectorGamma,MSEValidation(1,:), '-b')
hold on
semilogx(vectorGamma,MSETrain(1,:), '-g')
semilogx(vectorGamma,MSETest(1,:), '-r')
legend('validation set', 'train set', 'test set')
title('Evolution of the error for 10-sample train set in function of gamma');
xlabel('regularization parameter')
ylabel('MSE')

figure()
semilogx(vectorGamma,MSEValidation(2,:), '-b')
hold on
semilogx(vectorGamma,MSETrain(2,:), '-g')
semilogx(vectorGamma,MSETest(2,:), '-r')
legend('validation set', 'train set', 'test set')
title('Evolution of the error for 100-sample train set in function of gamma');
xlabel('regularization parameter')
ylabel('MSE')


