function q=spicy(x)
q=x;
q(x>0)=1./(1+exp(-x(x>0)));
q(x<0)=(exp(x(x<0))./(1+exp(x(x<0))));
end