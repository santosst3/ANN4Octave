function [y, y_pp, hitrate] = func_patternclass(rna,x,d)
  % Pattern classifier with ANNs
  hit = 0;
  for k = 1 : length(x(1,:))
    y(:,k) = func_forwardANN(rna,x(:,k));
    y_pp(k) = round(y(k));
    if exist('d','var')
      % Compute the error if the real data is provided
      if sum(y_pp(k) - d(k)) == 0
        hit = hit + 1;
      endif
    end
  endfor
  if exist('d','var')
    % Compute the error if the real data is provided
    hitrate = hit*100/length(d);
  else
    hitrate = [];
  end
endfunction
