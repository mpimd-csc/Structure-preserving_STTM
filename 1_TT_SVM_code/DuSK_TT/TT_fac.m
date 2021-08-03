function [ data_TT , R] = TT_fac(X);
%% Input
% X: a N*1 cell array for storing third-order tensor data
% Task 1 : Getting compact representation of input data in TT form
N = length(X);
eps = 1e-14;
%% Decompose the input data with TT decomposition
addpath('.\TT-Toolbox-master');
data_TT=cell(N,1);                                                            % Save TT decomposition results
fprintf('Decomposing the input data with TT decomposition, please wait!\n');
for i=1:N
    tt = tt_tensor(X{i}, eps);
     R{i} = tt.r;
     G = core2cell(tt); 
    data_TT{i}=cell(3,1);
    data_TT{i}{1}=G{1};
    data_TT{i}{2}=G{2};
    data_TT{i}{3}=G{3};
end
clear G
%display(R);
end
