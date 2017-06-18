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

function fis=CreateInitialFIS(data,nCluster)

    if ~exist('nCluster','var')
        nCluster='auto';
    end

    x=data.TrainInputs;
    t=data.TrainTargets;
    
    fcm_U=2;
    fcm_MaxIter=100;
    fcm_MinImp=1e-5;
    fcm_Display=false;
    fcm_options=[fcm_U fcm_MaxIter fcm_MinImp fcm_Display];
    fis=genfis3(x,t,'sugeno',nCluster,fcm_options);

end