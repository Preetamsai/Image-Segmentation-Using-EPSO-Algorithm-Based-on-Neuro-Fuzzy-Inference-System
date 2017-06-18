function image_seg
clc
clear all

loop = true;
while loop
    ch = menu('Segmentation',...
        'Load Image',...
        'Add Noise',...
        'Filter Image',...
        'EPSO Method',...
        'Fuzzy C means',...
        'Fuzzy Set Theory',...
        'EM Segmentation',...
        'Performance',...
        'Exit');
    switch ch
        case 1
            I = load_image;
        case 2
            I = imnoise(I,'salt & pepper',.01);
            figure(1),imshow(I),title('Noisy Image')
        case 3
            I = medfilt2(I);
            figure(1),imshow(I),title('Filtered Image')
        case 4                
            I_EPSO = EPSO(I,2);
            I_EPSO(I_EPSO == min(min(I_EPSO))) = 0;
            I_EPSO(I_EPSO == max(max(I_EPSO))) = 1;
        case 5
            I_fcm = fuzzy_c_means(I);
            I_fcm(I_fcm == min(min(I_fcm))) = 0;
            I_fcm(I_fcm == max(max(I_fcm))) = 1;
        case 6
            fuzzy_set_theory(I);
        case 7
            I_em = em_seg(I,2);
            I_em(I_em == min(min(I_em))) = 0;
            I_em(I_em == max(max(I_em))) = 1;
        case 8
            disp('FCM with EPSO:')
            disp('==============')
            performance(I_EPSO,I_fcm);
            disp('FCM with EM:')
            disp('============')
            performance(I_em,I_fcm);
            disp('EM with OSTU:')
            disp('=============')
            performance(I_em,I_EPSO);
        case 9
            clc,clear all,close all
            loop = false;
    end
end
end





function I = load_image
[file path] = uigetfile('*.png','Select an image to segment:');
I = imread([path file]);
figure(1)
imshow(I)
title('Input Image')
if size(I,3) == 3 % check rgb 
    I = rgb2gray(I); % convert to gray
end            
% I = im2double(I);
end
function [IDX,sep] = EPSO(I,n)
I = single(I);
% Convert to 256 levels
I = I-min(I(:));
I = round(I/max(I(:))*255);
% Probability distribution
unI = sort(unique(I));
nbins = min(length(unI),256);
if nbins==n
    IDX = ones(size(I));
    for i = 1:n, IDX(I==unI(i)) = i; end
    sep = 1;
    return
elseif nbins<n
    IDX = NaN(size(I));
    sep = 0;
    return
elseif nbins<256
    [histo,pixval] = hist(I(:),unI);
else
    [histo,pixval] = hist(I(:),256);
end
P = histo/sum(histo);
clear unI
% Zeroth- and first-order cumulative moments
w = cumsum(P);
mu = cumsum((1:nbins).*P);
% Maximal sigmaB^2 and Segmented image
if n == 2
    sigma2B =...
        (mu(end)*w(2:end-1)-mu(2:end-1)).^2./w(2:end-1)./(1-w(2:end-1));
    [maxsig,k] = max(sigma2B);
    % segmented image
    IDX = ones(size(I));
    IDX(I>pixval(k+1)) = 2;
    % separability criterion
    sep = maxsig/sum(((1:nbins)-mu(end)).^2.*P);
elseif n == 3
    w0 = w;
    w2 = fliplr(cumsum(fliplr(P)));
    [w0 w2] = ndgrid(w0,w2);
    mu0 = mu./w;
    mu2 = fliplr(cumsum(fliplr((1:nbins).*P))./cumsum(fliplr(P)));
    [mu0 mu2] = ndgrid(mu0,mu2);
    w1 = 1-w0-w2;
    w1(w1<=0) = NaN;
    sigma2B =...
        w0.*(mu0-mu(end)).^2 + w2.*(mu2-mu(end)).^2 +...
        (w0.*(mu0-mu(end)) + w2.*(mu2-mu(end))).^2./w1;
    sigma2B(isnan(sigma2B)) = 0; % zeroing if k1 >= k2
    [maxsig,k] = max(sigma2B(:));
    [k1 k2] = ind2sub([nbins nbins],k);
    % segmented image
    IDX = ones(size(I))*3;
    IDX(I<=pixval(k1)) = 1;
    IDX(I>pixval(k1) & I<=pixval(k2)) = 2;
    % separability criterion
    sep = maxsig/sum(((1:nbins)-mu(end)).^2.*P);
else
    k0 = linspace(0,1,n+1);
    k0 = k0(2:n);
    [k,y] = fminsearch(@sig_func,k0,optimset('TolX',1));
    k = round(k*(nbins-1)+1);
    % segmented image
    IDX = ones(size(I))*n;
    IDX(I<=pixval(k(1))) = 1;
    for i = 1:n-2
        IDX(I>pixval(k(i)) & I<=pixval(k(i+1))) = i+1;
    end
    % separability criterion
    sep = 1-y;    
end
IDX(~isfinite(I)) = 0;
figure(1)
imshow(IDX,[])
title('EPSO''s Method')
end
function y = sig_func(k)
% Function to be minimized if n>=4
muT = sum((1:nbins).*P);
sigma2T = sum(((1:nbins)-muT).^2.*P);
k = round(k*(nbins-1)+1);
k = sort(k);
if any(k<1 | k>nbins)
    y = 1;
    return
end
k = [0 k nbins];
sigma2B = 0;
for j = 1:n
    wj = sum(P(k(j)+1:k(j+1)));
    if wj == 0
        y = 1;
        return
    end
    muj = sum((k(j)+1:k(j+1)).*P(k(j)+1:k(j+1)))/wj;
    sigma2B = sigma2B + wj*(muj-muT)^2;
end
y = 1-sigma2B/sigma2T; % within the range [0 1]
end




function IMMM = fuzzy_c_means(I)
IM = double(I);
[maxX maxY] = size(IM);
IMM = cat(3,IM,IM);
cc1 = 8;
cc2 = 250;
ttFcm = 0;
while ttFcm < 15
    ttFcm = ttFcm+1;
    c1 = repmat(cc1,maxX,maxY);
    c2 = repmat(cc2,maxX,maxY);
    c = cat(3,c1,c2);
    ree = repmat(0.000001,maxX,maxY);
    ree1 = cat(3,ree,ree);    
    distance = IMM-c;
    distance = distance.*distance+ree1;
    daoShu = 1./distance;    
    daoShu2 = daoShu(:,:,1)+daoShu(:,:,2);
    distance1 = distance(:,:,1).*daoShu2;
    u1 = 1./distance1;
    distance2 = distance(:,:,2).*daoShu2;
    u2 = 1./distance2;      
    ccc1 = sum(sum(u1.*u1.*IM))/sum(sum(u1.*u1));
    ccc2 = sum(sum(u2.*u2.*IM))/sum(sum(u2.*u2));
    tmpMatrix = [abs(cc1-ccc1)/cc1,abs(cc2-ccc2)/cc2];
    pp = cat(3,u1,u2);    
    for i = 1:maxX
        for j = 1:maxY
            if max(pp(i,j,:)) == u1(i,j)
                IX2(i,j) = 1;           
            else
                IX2(i,j) = 2;
            end
        end
    end
    if max(tmpMatrix) < 0.0001
        break
    else
        cc1 = ccc1;
        cc2 = ccc2;        
    end
    for i = 1:maxX
        for j = 1:maxY
            if IX2(i,j) == 2
                IMMM(i,j) = 254;
            else
                IMMM(i,j) = 8;
            end
        end
    end
    figure(1); 
    imshow(uint8(IMMM));
    title('Fuzzy C means');
end
for i = 1:maxX
    for j = 1:maxY
        if IX2(i,j) == 2
            IMMM(i,j) = 200;
        else
            IMMM(i,j) = 1;
        end
    end
end 
IMMM = uint8(IMMM);
end




function J = histogramEq(I)
I = double(I');
seq = double(I(:));
seq = sort(seq);
seq = round(seq);
value = [];
count = [];
while ~isempty(seq)
    pos = find(seq==seq(1));
    c = length(pos);
    value = [value seq(1)];
    count = [count c];
    seq(pos) = [];
end
cdf = [];
for i = 1:length(value)
    cdf = [cdf sum(count(1:i))];
end
cdf_min = min(cdf);
[M N] = size(I);
h = zeros(1, length(value));
for v=1:length(value)
    h(v) = round(((cdf(v)-cdf_min)/((M*N)-cdf_min)).*255);
end
J = zeros(M, N);
S = sort(value);
for i = 1:length(value)
    pos = (I==S(i));
    J(pos) = h(i);
end
J = J';
end




function psi = fuzzy_set_theory(I)
I1 = I;
a = 50;
c = 200;
S = sFunction(a,c);
figure(3),
plot(S,'r');
xlabel('Gray Level');
title('Typical shape of the S-function');
grid on;
a = 200;
c = 255;
S = sFunction(a,c);
figure(2),
plot(S,'r');
Z = S(end:-1:1);
hold on;
plot(Z,'r');
axis([0 255 0 1]);
grid on;
xlabel('Gray Level');
text(101,.45,'Fuzzy Region');
title('Histogram and the functions for the seed subsets');
H = imhist(I);
stem(H/max(H),'g');
hold off;
k = 20;
muA = S;
n = length(H);
muAstar = muA >= 0.5;
psi = (2/(n^(1/k)))*((sum((abs(muA - muAstar)).^k))^(1/k));
% disp('Measures of Fuzziness:');
% fprintf('Index (psi) = %f\n',psi);
if max(I1(:)) < 150 % for small contrast images
    I1 = histogramEq(I1);
end
I1 = im2double(I1);
figure(1)
imshow(~(I1 < (1-psi)))
title('res')
end



function y = sFunction(a,c) 
b = (a+c)/2;
y = zeros(1,256);
for x = 0:255
    if  x <= a
        y(x+1) = 0;
    elseif a <= x && x<= b
        y(x+1) = 2*((x-a)/(c-a))^2;
    elseif b <= x && x <= c
        y(x+1) = 1 - 2*((x-c)/(c-a))^2;
    else
        y(x+1) = 1;
    end
end
end




function mask = em_seg(ima,k)
if ndims(ima) == 3
    ima = rgb2gray(ima); 
end
% check image
ima = double(ima);
copy = ima;           % make a copy
ima = ima(:);         % vectorize ima
mi = min(ima);        % deal with negative 
ima = ima-mi+1;       % and zero values
m = max(ima);
% create image histogram
h = histogram(ima);
x = find(h);
h = h(x);
x = x(:);h = h(:);
% initiate parameters
mu = (1:k)*m/(k+1);
v = ones(1,k)*m;
p = ones(1,k)*1/k;
% start process
sml = mean(diff(x))/1000;
while 1
        % Expectation
        prb = distribution(mu,v,p,x);
        scal = sum(prb,2)+eps;
        loglik = sum(h.*log(scal));
        %Maximizarion
        for j = 1:k
                pp = h.*prb(:,j)./scal;
                p(j) = sum(pp);
                mu(j) = sum(x.*pp)/p(j);
                vr = (x-mu(j));
                v(j)= sum(vr.*vr.*pp)/p(j)+sml;
        end
        p = p + 1e-3;
        p = p/sum(p);
        % Exit condition
        prb = distribution(mu,v,p,x);
        scal = sum(prb,2)+eps;
        nloglik = sum(h.*log(scal));                
        if ((nloglik-loglik)<0.0001)
            break
        end        
end
% calculate mask
mu = mu+mi-1;   % recover real range
s = size(copy);
mask = zeros(s);
for i = 1:s(1)
    for j = 1:s(2)
        for n = 1:k
            c(n) = distribution(mu(n),v(n),p(n),copy(i,j)); 
        end
        a = find(c==max(c));
        mask(i,j) = a(1);
    end
    figure(1),clf,imshow(mask,[]),title(sprintf('EM - %d  Segmentaed Image',k))
end
end



function y = distribution(m,v,g,x)
x = x(:);
m = m(:);
v = v(:);
g = g(:);
for i = 1:size(m,1)
    d = x-m(i);
    amp = g(i)/sqrt(2*pi*v(i));
    y(:,i) = amp*exp(-0.5 * (d.*d)/v(i));
end
end



function [h] = histogram(datos)
datos = datos(:);
ind = find(isnan(datos)==1);
datos(ind) = 0;
ind = find(isinf(datos)==1);
datos(ind) = 0;
tam = length(datos);
m = ceil(max(datos))+1;
h = zeros(1,m);
for i = 1:tam,
    f = floor(datos(i));        
    if f>0 && f<(m-1)        
        a2 = datos(i)-f;
        a1 = 1-a2;
        h(f) = h(f) + a1;      
        h(f+1) = h(f+1)+ a2;                          
    end
end
h = conv(h,[1,2,3,2,1]);
h = h(3:(length(h)-2));
h = h/sum(h);
end



function performance(J,K)
v1 = double(J(:));
v2 = double(K(:));
seq = v1 - v2;
f00 = length(find(seq)); % number of such pairs that fall in different clusters under J and K .
f01 = length(find(seq == -1)); % number of such pairs that fall in the same cluster under K but not under J .
f10 = length(find(seq == 1)); % number of such pairs that fall in the same cluster under K but not under J .
f11 = length(find(seq == 0)); % number of such pairs that fall in the same cluster under K and J .
rand_index = 1 - (f11 + f00)/(f00 + f01 + f10 + f11); % Rand Index
precision = f00 / (f01 + f00); % Precision
recall = f00 / (f10 + f00); % Recall
f_measure = 100 * 2 * (precision * recall) / (precision + recall); % F-measure
sensetivity = f00 / (f00 + f10); % sensetivity
specificity = f11 / (f11 + f01); % specificity 
BCR = 0.5 * (sensetivity + specificity); % BCR: Balanced Classification Rate 
AUC = 0.5 * (sensetivity + specificity); % AUC: Area Under the Curve
BER = 100 * (1 - BCR); % BER: Balanced Error Rate
disp('RAND INDEX = ')
disp(rand_index)
disp('F-MEASURE = ')
disp(f_measure)
disp('sensetivity = ')
disp(sensetivity)
disp('specificity = ')
disp(specificity)
disp('BCR = ')
disp(BCR)
disp('AUC = ')
disp(AUC)
disp('BER = ')
disp(BER)
end
