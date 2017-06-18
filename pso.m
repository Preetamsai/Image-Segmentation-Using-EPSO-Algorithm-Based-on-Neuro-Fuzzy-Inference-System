clc;
clear all;
close all;
%reading image
I=imread('coins.png');
figure, imshow(I); title('Original image');
[m n]=size(I);
%PSO implementation
P=30;       %swarm size
iter=10;    %number of iterations
c1=2.5;
c2=1.5;     %usually c1=c2=2
r1=rand(1,1);
r2=rand(1,1);
fit_val=[]; %matrix for storing fitness values
P_best=[];  %matrix for storing pbest values
pbest=0;
gbest=0;
%particle initialization
for i=1:P
%updating particle position
      a(i)=(1.5).*rand(1,1);
      b(i)=(0.5).*rand(1,1);
      c(i)=rand(1,1);
      k(i)=0.5+1.*rand(1,1);
      %updating particle velocity
      v(i)=rand(1,1);
  end
%iterations
for it=1:iter
    for i=1:P
        g={I,a(i),b(i),c(i),k(i)}; %generating enhanced image
          fitness={g,m,n}; 
          fit_val=[fit_val,fitness];   	    %calculating fitness value
      end
      gbest=max(P_best(:));             %calculating gbest
      %updating particle position and velocity
      for i=1:P
          v(i)=v(i) + c1.*r1.*(pbest-i) + c2.*r2.*(gbest-i);
          x(i)=x(i) + v(i);
      end
  end
 
%function to transform image to enhanced version
D=mean2(double(I));                     %global mean
s=stdfilt(I);                           %local std dev
lm=conv2(double(I),ones(3)/9,'same');   %local mean
x=s+b;
w=k.*D;
K=w./x;
g=K.*(double(I)-(c*lm))+(lm.^a);

%function to calculate fitness value
      Is=edge(I,'sobel');
      E=nnz(Is);      %sum of pixel intensities
      n_edgels=E;     %number of edge pixels
      H=entropy(I);   %entropy of enhanced image
      f=log(log(E)).*(n_edgels./(m.*n)).*H;
