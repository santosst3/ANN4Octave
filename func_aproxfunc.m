function [y, mean_error, mean_var] = func_aproxfunc(rna,x,d)
  % Function approximator/predictor with ANN
  mean_error = 0;
  mean_var = 0;
  for k = 1 : length(x(1,:))
    y(:,k) = func_forwardANN(rna,x(:,k));
    if exist('d','var')
      % Compute the error if the real data is provided
      mean_error = mean_error + abs(y(:,k)-d(k));
      mean_var = mean_var + (y(:,k)-d(k))^2;
    end
  endfor
  if exist('d','var')
    % Compute the error if the real data is provided
    mean_error = mean_error*100/length(d);
    mean_var = mean_var*100/length(d);
  else
    mean_error = [];
    mean_var = [];
  end
endfunction
