function [dimseq_without_i,dimseq_first_i] = dimseq(n_dim,i)
if i == 1
    dimseq_without_i = [i+1:n_dim];
    dimseq_first_i = [1:n_dim];
else if i == n_dim
        dimseq_without_i = [1:i-1];
        dimseq_first_i = [i,1:i-1];
    else
        
        dimseq_without_i = [1:i-1,i+1:n_dim];
        dimseq_first_i = [i,1:i-1,i+1:n_dim];
    end
end
end