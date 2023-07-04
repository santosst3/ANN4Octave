function y = func_inputnorm(inputs,max1,min1)
  y = inputs*0;
  for k = 1 : size(inputs,2)
    y(:,k) = 2*(inputs(:,k)-min1)./(max1-min1) - 1;
  endfor
endfunction
