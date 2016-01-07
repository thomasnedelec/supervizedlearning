function MSE = dualcost(K,alpha,y)
l = size(K,1);
MSE = (K*alpha - y)'*(K*alpha - y) / l;
