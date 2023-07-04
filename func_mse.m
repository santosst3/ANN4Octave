function y = func_mse(rna,x,d)
  p = length(d);
  yfinal = 0;
  for k = 1 : p
    y = func_forwardANN(rna,x(:,k));
    yfinal = yfinal + 0.5*sum((d(:,k) - y).^2);
  endfor
  y = yfinal/p;
endfunction
