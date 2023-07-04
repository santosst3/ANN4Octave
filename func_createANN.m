function y = func_createANN(anntype,anndimensions,activfunc,eta,epsilon,b,a,max1,min1)
  switch anntype
    case {"MLP" "mlp"}
      y.anntype = 1;
      printf("\nANN type: MLP");
    case {"RBF" "rbf"}
      y.anntype = 2;
      printf("\ANN type: RBF");
    otherwise
      y.anntype = 1;
      printf("\ANN type: MLP");
  endswitch
  y.eta = eta; % learning rate
  y.epsilon = epsilon; % learning error
  y.alfa = a; % momentum
  y.beta_log = b; % slope (logistic)
  y.neuron_layers = length(anndimensions)-1; % no of layers
  for k = 1 : y.neuron_layers
    switch activfunc{k}
      case {"Logistic" "logistic" "LOGISTIC" "Log" "log" "LOG"}
        y.activfunc{k} = 1;
        y.w{k} = (2*rand(anndimensions(k)+1,anndimensions(k+1))'-1);
      case {"Tanh" "tanh" "TANH" "Tangent" "tangent" "TANGENT"}
        y.activfunc{k} = 2;
        y.w{k} = (2*rand(anndimensions(k)+1,anndimensions(k+1))'-1);
      case {"Gaussian" "gaussian" "GAUSSIAN" "Gauss" "gauss" "GAUSS"}
        y.activfunc{k} = 3;
        y.w{k} = rand(anndimensions(k),anndimensions(k+1))';
      case {"Linear" "linear" "LINEAR" "Lin" "lin" "LIN"}
        y.activfunc{k} = 4;
        y.w{k} = 2*rand(anndimensions(k)+1,anndimensions(k+1))'-1;
      otherwise
        switch y.anntype
          case 1
            y.activfunc{k} = 1; % ogistic
            y.w{k} = (2*rand(anndimensions(k)+1,anndimensions(k+1))-1);
          case 2
            if k == y.neuron_layers
              y.activfunc{k} = 4; % Linear
              y.w{k} = (2*rand(anndimensions(k)+1,anndimensions(k+1))-1);
            else
              y.activfunc{k} = 3; % Gaussian
              y.w{k} = rand(anndimensions(k),anndimensions(k+1));
            endif
        endswitch
    endswitch
  endfor
  y.maxima = max1;
  y.minima = min1;
  y.trained = 0; % Not trained yet
  printf("\nANN created!\n\n");
endfunction
