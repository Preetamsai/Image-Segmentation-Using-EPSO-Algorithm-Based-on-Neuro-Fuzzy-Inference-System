%
% Copyright (c) 2015, Yarpiz (www.yarpiz.com)
% All rights reserved. Please read the "license.txt" for license terms.
%
% Project Code: YPFZ104
% Project Title: Evolutionary ANFIS Traing in MATLAB
% Publisher: Yarpiz (www.yarpiz.com)
% 
% Developer: S. Mostapha Kalami Heris (Member of Yarpiz Team)
% 
% Contact Info: sm.kalami@gmail.com, info@yarpiz.com
%

clc;
clear;
close all;

%% Load Data

data=LoadData();

%% Generate Basic FIS

fis=CreateInitialFIS(data,10);

%% Tarin Using PSO

Options = {'Genetic Algorithm', 'Particl Swarm Optimization'};

[Selection, Ok] = listdlg('PromptString', 'Select training method for ANFIS:', ...
                          'SelectionMode', 'single', ...
                          'ListString', Options);

pause(0.01);
          
if Ok==0
    return;
end

switch Selection
    case 1, fis=TrainAnfisUsingGA(fis,data);        
    case 2, fis=TrainAnfisUsingPSO(fis,data);        
end

%% Results

% Train Data
TrainOutputs=evalfis(data.TrainInputs,fis);
PlotResults(data.TrainTargets,TrainOutputs,'Train Data');

% Test Data
TestOutputs=evalfis(data.TestInputs,fis);
PlotResults(data.TestTargets,TestOutputs,'Test Data');
