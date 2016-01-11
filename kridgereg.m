function alpha = kridgereg(K,gamma,y)
l = size(K,1);
alpha = (K + gamma*eye(l))\y; 
end
