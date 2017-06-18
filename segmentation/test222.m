
[file path] = uigetfile('*.png','Select an image to segment:');
I = imread([path file]);
figure(1)
imshow(I)
title('Input Image')
if size(I,3) == 3 % check rgb 
    I = rgb2gray(I); % convert to gray
end            
% I = im2double(I);

[Iout,intensity,fitness,time]=segmentation(I,2,'dpso');

figure(2)
imshow(Iout);
