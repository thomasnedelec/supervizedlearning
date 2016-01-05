function alpha = kridgereg(K,gamma,trainY,nTrainPoints, k,nCrossValidation)
KfirstPart=K(1:nCrossValidation * nTrainPoints/k,1:nCrossValidation * nTrainPoints/k);
KsecondPart=K(((nCrossValidation+1) * nTrainPoints/k) + 1:nTrainPoints, 1:nCrossValidation * nTrainPoints/k);
KthirdPart=K(1:nCrossValidation * nTrainPoints/k,((nCrossValidation+1) * nTrainPoints/k) + 1:nTrainPoints);
KfourthPart=K(((nCrossValidation+1) * nTrainPoints/k) + 1:nTrainPoints,((nCrossValidation+1) * nTrainPoints/k)+1:nTrainPoints);
Ktrain=[KfirstPart,KthirdPart;KsecondPart,KfourthPart];
[l,m]=size(Ktrain);
size(Ktrain)
l
size(trainY)
alpha=(Ktrain+gamma*l*eye(l))\trainY;
end
