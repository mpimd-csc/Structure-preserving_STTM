function [Kernel_Value] = Ker_fTTCP(X1,X2,Order,g,l);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% It gives second part of algrithm 
% kernel of each tensor pair in KTTCP form 
%% input: 
% X1,X2 : the data_KTTCP of input tensors X, and X = N*1 cell
% Order : global variable and order/way of input tensor X
% g     : sigma of Gaussian kernel function

%% Output:
% kernel approximation value over each pair of input tensor
% This is the last step of the Algorithm 1 mentioned in paper
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
       Kernel_Value =0;
       R =  size(X1{1}, 2); % r = r1*r2
       Q =  size(X2{1}, 2); % q = q1*q2
        for i=1:R
            for j=1:Q
                S=1;
                for k=1:Order
                    S=S*exp(-norm(X1{k,1}(:,i)-X2{k,1}(:,j))^2/(2*g^2));
                end
                Kernel_Value = Kernel_Value+S;
            end
        end
end 