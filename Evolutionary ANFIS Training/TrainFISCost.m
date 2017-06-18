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

function [z, out]=TrainFISCost(x,fis,data)

    MinAbs=1e-5;
    if any(abs(x)<MinAbs)
        S=(abs(x)<MinAbs);
        x(S)=MinAbs.*sign(x(S));
    end

    p0=GetFISParams(fis);

    p=x.*p0;
    
    fis=SetFISParams(fis,p);
    
    x=data.TrainInputs;
    t=data.TrainTargets;
    y=evalfis(x,fis);
    
    e=t-y;
    
    MSE=mean(e(:).^2);
    RMSE=sqrt(MSE);
    
    z=RMSE;
    
    out.fis=fis;
    %out.y=y;
    %out.e=e;
    out.MSE=MSE;
    out.RMSE=RMSE;
    
end