function [xSetsTrain, ySetsTrain, xSetsValidation, ySetsValidation] = kFoldCrossValidation(X,Y, k)
dimension = size(X,2);
nTrainPoints = size(X,1);
xSetsTrain = zeros((k-1)/k * nTrainPoints, dimension, k);
ySetsTrain = zeros((k-1)/k * nTrainPoints, k);
xSetsValidation = zeros(1/k * nTrainPoints, dimension, k);
ySetsValidation = zeros(1/k * nTrainPoints, k);

iter = 1;
for i = 0 : 1/k : (k-1)/k
    xCopy = X;
    yCopy = Y;
    xSetsValidation(:,:,iter) = X(i * nTrainPoints + 1: (i + 1/k) * nTrainPoints,:);
    xCopy(i * nTrainPoints + 1: (i + 1/k) * nTrainPoints,:) = [];
    xSetsTrain(:,:,iter) = xCopy;
    
    ySetsValidation(:,iter) = Y(i * nTrainPoints + 1: (i + 1/k) * nTrainPoints,:);
    yCopy(i * nTrainPoints + 1: (i + 1/k) * nTrainPoints,:) = [];
    ySetsTrain(:,iter) = yCopy;
    iter = iter + 1;
end