function alpha = kridgereg(K,gamma,trainY,k,nCrossValidation)
[nTrainPoints,d]=size(trainY);
KfirstPart=K(1:nCrossValidation * nTrainPoints,1:nCrossValidation * nTrainPoints);
KsecondPart=K((nCrossValidation+1 * nTrainPoints) + 1:nTrainPoints, 1:nCrossValidation * nTrainPoints);
KthirdPart=K(1:nCrossValidation * nTrainPoints,(nCrossValidation+1 * nTrainPoints) + 1:nTrainPoints);
KfourthPart=K((nCrossValidation+1 * nTrainPoints) + 1:nTrainPoints,(nCrossValidation+1 * nTrainPoints)+1:nTrainPoints);
Ktrain=[KfirstPart,KthirdPart;KsecondPart,KfourthPart];
[l,m]=size(Ktrain);
alpha=(Ktrain+gamma*l*eye(l))\trainY;
end
