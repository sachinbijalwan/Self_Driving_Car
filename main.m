file=importdata('data.txt');
mini_size=64;  %batch size
size1=1024;    %size of the first layer
size2=512;     %size of the second layer
size3=64;      %size of third layer
dropout=0;     %dropout percentage
alpha=0.01;    %learning rate for first layer
alpha1=0.1;    %learning rate for second layer 
alpha2=1;       %learning rate for third layer
%values of weight matrix
w1=-0.01+rand(size1+1,size2)*0.02;  
w2=-0.01+rand(size2+1,size3)*0.02;
w3=-0.01+rand(size3+1,1)*0.02;

error=0;
%Buffer of file
L=ones(size1,size(file.textdata,1));

%normalizing image and converting it from 32*32 to 1024
for j=1:size(file.textdata,1)
      C=im2double(imread(file.textdata{j}(3:end)));
      C=rgb2gray(C);
      p=Twotoone(C);        
      L(:,j)=p(:,1);
end  

%preprocessing splitting and randomizing data
ids=randperm(size(file.data,1));
file.data=file.data(ids);
L=L(:,ids);
all_angles=file.data;
angles=ones(20);
training_size=80/100*(size(file.textdata,1)-mini_size);
nEpochs=5000;
training_size=floor(training_size);
train_error=zeros(nEpochs,1);
test_error=train_error;
final_training=zeros(1,training_size);

T=L(:,training_size+1:end);
L=L(:,1:training_size);
train_angles=all_angles(1:training_size,1);
test_angles=all_angles(training_size+1:end,1);

tic
for j=1:nEpochs

 final_training=zeros(1,training_size);
final_angles=zeros(training_size,1);

for i=1:mini_size:training_size-mini_size
   clear X

X=L(:,i:i+mini_size-1); %train data
angles=train_angles(i:i+mini_size-1,1);

A=[ones(size(X,2),1) X']';

%forward pass
X1=w1'*A;
X1=spicy(X1);
A1=[ones(size(X1,2),1) X1']';
X2=w2'*A1;
X2=spicy(X2);
A2=[ones(size(X2,2),1) X2']';
X3=w3'*A2;

%transformation matrices
t3=A2*(X3-angles')';
t2=A1*(((X3-angles')'*w3(2:end)').*X2'.*(1-X2)');
t1=A*((((((X3-angles')'*w3(2:end)').*X2'.*(1-X2)'))*w2(2:end,:)').*X1'.*(1-X1)');

drop1=randperm(size(t1,1),floor(dropout*size(t1,1)));
drop2=randperm(size(t2,1),floor(dropout*size(t2,1)));
drop3=randperm(size(t3,1),floor(dropout*size(t3,1)));

%performing dropout
t1(drop1,:)=0;
t2(drop2,:)=0;
t3(drop3,:)=0;
%updating weights
w3=w3-alpha*t3/size(X,2);
w2=w2-alpha1*t2/size(X,2);
w1=w1-alpha2*t1/size(X,2);
end

%calculating train error
X=L;
angles=train_angles;
A=[ones(size(X,2),1) X']';
X1=w1'*A;
X1=spicy(X1);
A1=[ones(size(X1,2),1) X1']';
X2=w2'*A1;
X2=spicy(X2);
A2=[ones(size(X2,2),1) X2']';
X3=w3'*A2;
train_error(j)=meansquarederr(X3',angles)/(size(X,2));

%calculating test error
X=T;
angles=test_angles;
A=[ones(size(X,2),1) X']';
X1=w1'*A;
X1=spicy(X1);
A1=[ones(size(X1,2),1) X1']';
X2=w2'*A1;
X2=spicy(X2);
A2=[ones(size(X2,2),1) X2']';
X3=w3'*A2;
test_error(j)=meansquarederr(X3',angles)/size(X,2);

fprintf("iteration : %d , test error %d train error %d\n",j,test_error(j),train_error(j));
toc
end
figure
plot(1:size(train_error),train_error,'b',1:size(test_error),test_error,'r');
ylabel('error');
xlabel('epoch');
savefig('first.fig');
close;