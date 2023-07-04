function [y,err,epoch] = func_trainRBF(ann,x,d)
  % Treinamento da rede RBF tem dois estágios
  % Step 1: Non-supervised training of hidden layer
  [ann.w{1}, ann.vargauss{1}] = func_kmeans(x,size(ann.w{1},1));
  
  % Step 2: Supervised Training of output layer
  epoch = 0;
  cont_train = 1;
  err = 1;
  errant = 0;
  w_anterior = ann.w;
  for k = 1 : length(d)
    [y_saida, i_intermed, y_intermed] = func_forwardANN(ann,x(:,k));
    newinput(:,k) = y_intermed{ann.neuron_layers};
  endfor
  while cont_train == 1
    if epoch == 0
      errant = err;
    else
      errant = err(epoch);
    endif
    for k = 1 : length(d)
      [ann, w_anterior] = func_forbackRBF(ann,newinput(:,k),d(:,k),w_anterior);
    endfor
    epoch = epoch + 1;
    err(epoch) = func_mse(ann,x,d);
    if abs(err(epoch) - errant) < ann.epsilon
      cont_train = 0;
    endif
  endwhile
  y = ann;
endfunction
