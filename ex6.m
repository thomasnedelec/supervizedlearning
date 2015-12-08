clear all;
clc;
close all;
toPlot = 0;
nbValueGamma=10;
vectorGamma=zeros(1,nbValueGamma);
nLoops = 200;
MSETrain=zeros(1,nbValueGamma);
MSETest=zeros(1,nbValueGamma);
MSEValidation=zeros(2,nbValueGamma);
averageTestError = zeros(1,nbValueGamma);
averageTrainError = zeros(1,nbValueGamma);
averageValidationError=zeros(1,nbValueGamma);
dimension = 10;
wTrained=zeros(dimension,nbValueGamma);
nTableTrainPoints=[10,100];
resultsMatrix=zeros(2);
indexaverage=zeros(1,2);
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
                wTrained(:,i)=(transpose(smallerTrainX)*smallerTrainX+gamma*percentage*nTrainPoints*eye(dimension))\(transpose(smallerTrainX)*smallerTrainY);
   
                mseTrainAverage = mseTrainAverage + computeMSE(wTrained(:,i),trainX,trainY);
                mseTestAverage = mseTestAverage + computeMSE(wTrained(:,i),testX,testY);
                mseValidation = mseValidation + computeMSE(wTrained(:,i),validationX,validationY);
            end

            mseTrainAverage = mseTrainAverage / k;
            mseTestAverage = mseTestAverage / k;
            mseValidation = mseValidation / k;
            
            MSETrain(j,i) = mseTrainAverage;
            MSETest(j,i) = mseTestAverage;
            
            MSEValidation(j,i) = mseValidation;
            
            vectorGamma(1,i)=gamma;
            gamma=10*gamma;
            averageValidationError(1,i) = averageValidationError(1,i) + MSEValidation(1,i);
        end;
        if(toPlot)
            figure
            semilogx(vectorGamma,MSEValidation(j,:), '-b')
            title('Evolution of the error on the development set in function of gamma');
        end;
        [minimum,index]=min(MSEValidation(j,:));
        indexaverage(1,j)=((i-1)*indexaverage(1,j)+index)./i;
        resultsMatrix(1,j)=minimum;
        resultsMatrix(2,j) = computeMSE(wTrained(:,index),testX,testY);
     end;
end;
indexaverage(1,1)=10^(-6)*10^indexaverage(1,1);
indexaverage(1,2)=10^(-6)*10^indexaverage(1,2);
indexaverage
resultsMatrix
MSEValidation;


