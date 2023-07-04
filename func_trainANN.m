function [y,err,epoch] = func_trainANN(ann,x,d)
  % Metafunction for calling other training functions
  switch ann.anntype
    case 1
      [y,err,epoch] = func_trainMLP(ann,x,d);
    case 2
      [y,err,epoch] = func_trainRBF(ann,x,d);
  endswitch
  y.trained = 1; % Training completed
endfunction
