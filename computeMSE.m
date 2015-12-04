function [er] = computeMSE(w,X,Y)
nData = size(X,1);
er = 1/nData * (X*w - Y)'*(X*w - Y);