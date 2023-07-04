function [y,err,epoch] = func_trainMLP(ann,x,d,refreshrate)
  epoch = 0;
  conttrain = 1;
  err = 1;
  preverror = 0;
  w_previous = ann.w;
  if ~exist('refreshrate','var')
    refreshrate = 20;
  endif
  while conttrain == 1
    if epoch == 0
      preverror = err;
    else
      preverror = err(epoch);
    endif
    for k = 1 : length(d)
      [y_saida, i_intermed, y_intermed] = func_forwardANN(ann,x(:,k));
      [ann, w_previous] = func_backwardMLP(ann,y_saida,d(:,k), i_intermed, y_intermed,w_previous);
    endfor
    epoch = epoch + 1;
    err(epoch) = func_mse(ann,x,d);
    if abs(err(epoch) - preverror) < ann.epsilon || epoch > 1000
      conttrain = 0;
    endif
    if refreshrate ~= 0
      if mod(epoch,refreshrate) == 0
        printf("\nEpoch: %d", epoch);
        printf("\nMSE: %f", err(epoch));
        printf("\nMSE variation: %e\n", abs(err(epoch) - preverror));
      endif
    endif
  endwhile
  y = ann;
endfunction
