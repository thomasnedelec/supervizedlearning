function MSE = dualcost(K,alpha,testY,k,nCrossValidation)
[nTrainPoints,d]=size(testY);
[nPoints,nPoints]=size(K);
KtestFirstPart=K(nCrossValidation * nTrainPoints + 1: (nCrossValidation + 1/k) * nTrainPoints,1:nCrossValidation * nTrainPoints);
KtestSecondPart=K(nCrossValidation * nTrainPoints + 1: (nCrossValidation + 1/k) * nTrainPoints,(nCrossValidation + 1/k) * nTrainPoints+1:nPoints);
Ktest=[KtestFirstPart,KtestSecondPart];
size(Ktest)
size(alpha)
MSE=1/nTrainPoints*transpose((Ktest*alpha-testY))*(Ktest*alpha-testY);
