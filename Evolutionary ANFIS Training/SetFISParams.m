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

function fis=SetFISParams(fis,p)

    nInput=numel(fis.input);
    for i=1:nInput
        nMF=numel(fis.input(i).mf);
        for j=1:nMF
            k=numel(fis.input(i).mf(j).params);
            fis.input(i).mf(j).params=p(1:k);
            p(1:k)=[];
        end
    end

    nOutput=numel(fis.output);
    for i=1:nOutput
        nMF=numel(fis.output(i).mf);
        for j=1:nMF
            k=numel(fis.output(i).mf(j).params);
            fis.output(i).mf(j).params=p(1:k);
            p(1:k)=[];
        end
    end

end