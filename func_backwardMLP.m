function [y,w_ant] = func_backwardMLP(rna,y,d,i,yant,w_ant)
  % Output layer
  delta = (d-y);
  k = rna.neuron_layers;
  switch rna.activfunc{k}
    case 1
      delta = delta.*func_dlogistic(i{k},rna.beta_log);
    case 2
      delta = delta.*func_dtanh(i{k},rna.beta_log);
    case 3
      delta = delta.*func_dgauss(i{k},rna.beta_log);
    case 4
      delta = delta; % Linear
  endswitch
  varaux = rna.w{k} + rna.alfa*(rna.w{k} - w_ant{k}(:,:)) + rna.eta*delta*yant{k}';
  w_ant{k} = rna.w{k};
  rna.w{k} = varaux;
  
  % Hidden layers
  for k = (rna.neuron_layers-1) : -1 : 1
    delta = sum(delta'*rna.w{k+1}(:,2:end),2)';
    switch rna.activfunc{k}
      case 1
        delta = delta.*func_dlogistic(i{k},rna.beta_log);
      case 2
        delta = delta.*func_dtanh(i{k},rna.beta_log);
    endswitch
    % Momentum
    varaux = rna.w{k} + rna.alfa*(rna.w{k} - w_ant{k}(:,:)) + rna.eta*delta*yant{k}';
    w_ant{k} = rna.w{k};
    rna.w{k} = varaux;
  endfor
  y = rna;
endfunction
