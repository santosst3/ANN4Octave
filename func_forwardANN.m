function [yfinal,i,y] = func_forwardANN(ann,x)
  y_orig = x;
  for k = 1 : ann.neuron_layers
    switch ann.activfunc{k}
      case 1
        y{k} = [-ones(1,size(y_orig,2));y_orig];
        i{k} = ann.w{k}*y{k};
        y_orig = func_logistic(i{k},ann.beta_log);
      case 2
        y{k} = [-ones(1,size(y_orig,2));y_orig];
        i{k} = ann.w{k}*y{k};
        y_orig = func_tanh(i{k},ann.beta_log);
      case 3
        y{k} = y_orig;
        i{k} = (y{k} - ann.w{k}).^2;
        i{k} = sum(i{k});
        y_orig = func_gaussian(i{k},ann.vargauss{k})';
      case 4
        y{k} = [-ones(1,size(y_orig,2));y_orig];
        i{k} = ann.w{k}*y{k};
        y_orig = i{k}; % Linear
    endswitch
  endfor
  yfinal = y_orig;
endfunction
