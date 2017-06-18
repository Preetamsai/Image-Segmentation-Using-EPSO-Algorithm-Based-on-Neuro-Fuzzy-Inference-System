close all; 
clear all; 
workspace;
%------ GET DEMO IMAGES ----------------------------------------------------------
% Read in a standard MATLAB gray scale demo image.
grayImage = imread('coins.png');
[rows columns] = size(grayImage);
% Display the first image.
subplot(2, 2, 1);
imshow(grayImage, []);
title('Original Gray Scale Image');
set(gcf, 'Position', get(0,'Screensize')); % Maximize figure.
% Get a second image by adding noise to the first image.
noisyImage = imnoise(grayImage, 'gaussian', 0, 0.003);
% Display the second image.
subplot(2, 2, 2);
imshow(noisyImage, []);
title('Noisy Image');
%------ PSNR CALCULATION ----------------------------------------------------------
% Now we have our two images and we can calculate the PSNR.
% First, calculate the "square error" image.
% Make sure they're cast to floating point so that we can get negative differences.
% Otherwise two uint8's that should subtract to give a negative number
% would get clipped to zero and not be negative.
squaredErrorImage = (double(grayImage) - double(noisyImage)) .^ 2;
% Display the squared error image.
subplot(2, 2, 3);
imshow(squaredErrorImage, []);
title('Squared Error Image');
% Sum the Squared Image and divide by the number of elements
% to get the Mean Squared Error.  It will be a scalar (a single number).
mse = sum(sum(squaredErrorImage)) / (rows * columns);
% Calculate PSNR (Peak Signal to Noise Ratio) from the MSE according to the formula.
PSNR = 10 * log10( 256^2 / mse);
% Alert user of the answer.
message = sprintf('The mean square error is %.2f.\nThe PSNR = %.2f', mse, PSNR);
msgbox(message);