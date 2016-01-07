function alpha = kridgereg(K,gamma,y)
l = size(K,1);
alpha=(K+gamma*l*eye(l))\y;
end
