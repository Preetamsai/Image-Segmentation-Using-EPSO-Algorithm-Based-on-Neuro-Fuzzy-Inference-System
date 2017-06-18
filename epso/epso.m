
popsize=20;
MAXITER=1000;
dimension=100;
irange_l=50;
irange_r=100;
xmax=100;
vmax=xmax;

sum1=0;
sum2=0;
mean=0;
st=0;
runno=5;
c1=2;
c2=2;
z1=zeros(1,runno);%�������õ��Ĳ�����

data1=zeros(runno,MAXITER);

for run=1:runno
T=cputime;
x=(irange_r- irange_l)*rand(popsize,dimension)+ irange_l;
vstep=2*vmax*rand(popsize,dimension)-vmax;

pbest=x;
gbest=zeros(1,dimension);

td=vmax*ones(1,dimension);
z=zeros(1,dimension);
for i=1:popsize
    f_x(i)=f1(x(i,:));
    f_pbest(i)=f_x(i);
end
  
    g=min(find(f_pbest==min(f_pbest(1:popsize))));
    gbest=pbest(g,:);
   
    f_gbest=f_pbest(g);

    MINIUM=f_pbest(g);
for t=1:MAXITER
    
     w_now=0.4;
     
 for i=1:popsize  
       r1=rand(1,dimension);
       r2=rand(1,dimension);
       y=w_now.*vstep(i,:)+c1.*r1.*(pbest(i,:)-x(i,:))+c2.*r2.*(gbest-x(i,:));
       vstep(i,:)=y;
       vstep(i,:)=sign(y).*min(abs(y),vmax);    
       
      for j=1:dimension
         if(abs(vstep(i,j))<td(j))
             %a1=(-0.1)*(vmax);%�˴���gaussian���졣�˴� �ɿ������ⲿʹ�ý������Ա��졣
            %b1=a1.*ones(1,1);
            %vstep(i,j)=vmax+(1/sqrt(2*pi))*exp((vmax.*vmax)./b1);
              vstep(i,j)=rand*vmax;
              z(j)=z(j)+1;
              pbest(i,j)=x(i,j)+w_now.*vstep(i,j)+c1.*rand(1,1)*(pbest(i,j)-x(i,j))+c2.*rand(1,1)*(gbest(1,j)-x(i,j));
              f_pbest(i)=f_x(i);
              if z(j)>2
                  z(j)=0;
                  td(j)=td(j)/10;
              end
          end
        end
          
       x(i,:)=x(i,:)+vstep(i,:);
       x(i,:)=sign(x(i,:)).*min(abs(x(i,:)),xmax);
          
            f_x(i)=f1(x(i,:));

           if f_x(i)<f_pbest(i)
                pbest(i,:)=x(i,:);
                f_pbest(i)=f_x(i);
           end          
            if f_pbest(i)<f_gbest
                gbest=pbest(i,:);
                f_gbest=f_pbest(i);
            end

           MINIUM=f_gbest;   
 end%end popsize
 %�˴���Ӻ�������
   [f_2,b]=sort(f_x);
   for i=1:3
      for j=1:dimension
                z=gbest;
                z(j)=x(b(i),j);
          if f1(z(1,:))<f_gbest
             gbest(j)=x(b(i),j);
             f_gbest=f1(z(1,:));
           end
       end        
   end
     f_gbest=f1(gbest(1,:));
    
 data1(run,t)=MINIUM;
 
%�˴�Ϊ�̶���������Ŀ��ֵ�����۴ﵽĿ����Ҫ�ĵ���������
%if MINIUM>10
    %z1(runno)=z1(runno)+1;
%else
    %break;
%end
end  %end interation
sum1=sum1+mean;  
sum2=sum2+MINIUM;
 %MINIUM
time=cputime-T;
st=st+time;

end  %end runno
av1=sum1/5;  %���ƽ���������
av2=sum2/5;  %���ƽ�����Ž�
st/5  %����㷨��ƽ��ʱ��
    
 