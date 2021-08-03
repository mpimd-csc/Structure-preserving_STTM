function [ data_TT2 ] = TTmat( X, data_TT);
%Detailed explanation goes here
%% Input
% X: a n*1 cell array for storing tensor data
% data_TT: the TT decomposed data, here we use TT-Toolbox to get it.
%% Output
% data_TT2 : matricization of TT cores
%%
n = length(X);
dimn=size(X{1});
m = length(dimn);
data_TT2 = cell(n,1);

%% Matricization of data_TT 
for i = 1:m
    for j = 1:n
        data_TT2{j,1}{i,1} = my_matricization(data_TT{j,1}{i,1},2);
    end
end

end