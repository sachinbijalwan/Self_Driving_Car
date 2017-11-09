function error=meansquarederr(X,Y)
% error=zeros(length(X(:,1)),1);
% for i=1:length(X(:,1))
%     error(i,1)=X(i,1)-Y(i,1);
% end
% error=0;
% for i=1:length(X(:,1))
% error=error+(X(i,1)-Y(i,1))*(X(i,1)-Y(i,1));
% end
error=(X-Y)'*(X-Y);
% error=error/(2*length(X(:,1)));
end