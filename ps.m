fcn = @dejong5fcn;
       nvars = 2;
       lb = [-64 -64];
       ub = [64 64];
       [x,fval] = particleswarm(fcn,nvars,lb,ub)