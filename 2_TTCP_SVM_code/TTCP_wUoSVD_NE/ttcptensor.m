function [TT_CP_data] = ttcptensor(data_TT,R1,R2,dimn,trunc);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Inputs:   data_TT: TT-core(G) of each tensor X
%           N      : Numer of input samples X
%           Rank   : [R1,R2]: Rank from tt -decomposition of each X
%           dimn = size(X{1});

% Note: This is the TT-CP expansion part of the paper, corresponding Matlab
%       commands have been explained there.

% Output:   TT-CP expansion for TT decomposition of each input tensor X
%%
N = length(data_TT);
if trunc == 1; % when ranks are different for TT-decomposition
  for i = 1:N
    TT_CP_data{i,1}{1,1} = kron(ones(1,R2(i,1)),reshape(permute(data_TT{i}{1},[2,1,3]),dimn(1),R1(i,1)));
    TT_CP_data{i,1}{2,1} = reshape(permute(data_TT{i}{2},[2,1,3]),dimn(2),R1(i,1)*R2(i,1));
    TT_CP_data{i,1}{3,1} = kron(reshape(permute(data_TT{i}{3},[2,1]),dimn(3),R2(i,1)),ones(1,R1(i,1)));
    normA{i,1}=sqrt(sum(TT_CP_data{i,1}{1,1}.*conj(TT_CP_data{i,1}{1,1})));  % Frobenius norm of each column of A
    normB{i,1}=sqrt(sum(TT_CP_data{i,1}{2,1}.*conj(TT_CP_data{i,1}{2,1})));
    normC{i,1}=sqrt(sum(TT_CP_data{i,1}{3,1}.*conj(TT_CP_data{i,1}{3,1})));
    prod_norm{i,1}=normA{i,1}.*normB{i,1}.*normC{i,1};
    Scale_mat{i,1}=diag(prod_norm{i,1}.^(1/3)); 
    % equal repartition of power of each rank-1 tensor over the 3 vectors:
    TT_CP_data{i,1}{1,1}=TT_CP_data{i,1}{1,1}*diag(1./normA{i,1})*Scale_mat{i,1};
    TT_CP_data{i,1}{2,1}=TT_CP_data{i,1}{2,1}*diag(1./normB{i,1})*Scale_mat{i,1};
    TT_CP_data{i,1}{3,1}=TT_CP_data{i,1}{3,1}*diag(1./normC{i,1})*Scale_mat{i,1};
  end
else
    for i =1:N
    l = R1;
    TT_CP_data{i,1}{1,1} = kron(ones(1,l),reshape(permute(data_TT{i}{1},[2,1,3]),dimn(1),l));
    TT_CP_data{i,1}{2,1} = reshape(permute(data_TT{i}{2},[2,1,3]),dimn(2),l*l);
    TT_CP_data{i,1}{3,1} = kron(reshape(permute(data_TT{i}{3},[2,1]),dimn(3),l),ones(1,l));
    normA{i,1}=sqrt(sum(TT_CP_data{i,1}{1,1}.*conj(TT_CP_data{i,1}{1,1})));  % Frobenius norm of each column of A
    normB{i,1}=sqrt(sum(TT_CP_data{i,1}{2,1}.*conj(TT_CP_data{i,1}{2,1})));
    normC{i,1}=sqrt(sum(TT_CP_data{i,1}{3,1}.*conj(TT_CP_data{i,1}{3,1})));
    prod_norm{i,1}=normA{i,1}.*normB{i,1}.*normC{i,1};
    Scale_mat{i,1}=diag(prod_norm{i,1}.^(1/3)); 
    
    % equal repartition of power of each rank-1 tensor over the 3 vectors:
    TT_CP_data{i,1}{1,1}=TT_CP_data{i,1}{1,1}*diag(1./normA{i,1})*Scale_mat{i,1};
    TT_CP_data{i,1}{2,1}=TT_CP_data{i,1}{2,1}*diag(1./normB{i,1})*Scale_mat{i,1};
    TT_CP_data{i,1}{3,1}=TT_CP_data{i,1}{3,1}*diag(1./normC{i,1})*Scale_mat{i,1};
    end
end
for i = 1:N
    for j = 1:3 % 3 is order of tensor
       TT_CP_data{i,1}{j,1} = TT_CP_data{i,1}{j,1};
    end
end
end