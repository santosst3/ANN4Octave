function [theta1,vargauss] = func_kmeans(data_inp,k)

    % Start means
    [dimens,n_data_inp] = size(data_inp);
    theta_n = data_inp(:,1:k);

    % Start clusters
    cur_cluster = zeros(n_data_inp,1);
    prev_cluster = ones(n_data_inp,1);

    % Covnergence loop
    while sum(cur_cluster == prev_cluster) ~= n_data_inp
        prev_cluster = cur_cluster;
        for a = 1 : n_data_inp
            % Euclidean distance
            for l = 1 : k
              d1(l) = vecnorm(data_inp(:,a)-theta_n(:,l),2);
            endfor
            % Nearest mean
            [~, var1] = min(d1);
            cur_cluster(a) = var1;
        endfor
        % Mean computation
        for l = 1 : k
            theta_n(:,l) = sum(data_inp(:,(cur_cluster == l)),2)/length(find(cur_cluster == l));
        endfor
    endwhile
    
    theta1 = theta_n;
    
    for l = 1 : k
      soma = 0;
      for a = find(cur_cluster == l)'
        soma = soma + vecnorm(data_inp(:,a)-theta_n(:,l),2)^2;
      end
      vargauss(:,l) = soma/length(find(cur_cluster == l));
    endfor
endfunction
