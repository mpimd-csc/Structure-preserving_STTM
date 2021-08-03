function flatten_matrix = my_matricization(X,md)
dims = size(X);
n_dim = length(dims);
[dimseq_without_i,dimseq_first_i] = dimseq(n_dim,md);
flatten_matrix = permute(X,dimseq_first_i);
flatten_matrix = reshape(flatten_matrix, dims(md),prod(dims(dimseq_without_i)));
end
