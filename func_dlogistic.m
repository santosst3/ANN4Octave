function y = func_dlogistic(u,b)
  % Logistic function derivative
  y = (b*exp(-b*u))./(1+exp(-b*u)).^2;
endfunction
