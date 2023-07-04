function [y,w_ant] = func_forbackRBF(ann,x,d,w_ant)
  % Output layer's RBF training
  % Forward
  y_orig = x;
  for k = ann.neuron_layers
    switch ann.activfunc{k}
      case 1
        y{k} = [y_orig];
        i{k} = ann.w{k}'*y{k};
        y_orig = func_logistic(i{k},ann.beta_log);
      case 2
        y{k} = [y_orig];
        i{k} = ann.w{k}'*y{k};
        y_orig = func_tanh(i{k},ann.beta_log);
      case 3
        y{k} = y_orig;
        i{k} = (y{k} - ann.w{k}).^2;
        i{k} = sum(i{k});
        y_orig = func_gaussian(i{k},ann.vargauss{k})';
      case 4
        y{k} = [y_orig];
        i{k} = ann.w{k}*y{k};
        y_orig = i{k}; % Linear
    endswitch
  endfor
  
  % Backward
  k = ann.neuron_layers;
  delta = (d-y_orig);
  switch ann.activfunc{k}
    case 1
      delta = delta.*func_dlogistic(i{k},ann.beta_log);
    case 2
      delta = delta.*func_dtanh(i{k},ann.beta_log);
    case 3
      delta = delta.*func_dgauss(i{k},ann.beta_log);
    case 4
      delta = delta; % Linear
  endswitch
  varaux = ann.w{k} + ann.alfa*(ann.w{k} - w_ant{k}(:,:)) + ann.eta*delta*x';
  w_ant{k} = ann.w{k};
  ann.w{k} = varaux;
  y = ann;
endfunction
