function Kernel_value = Ker_fTT(X1,X2,Y1,Y2,Order,g)
% Details goes here
% it gives second part of algrithm 
% kernel of each tensor pair in KTT form 
% X1 X2 are data_KTT2 of input tensors X, and X = n*1 cell 
% Y1 Y2 are data_TT , to get the corresponding dimensions of TT-cores

% Kernel_value = exp(-norm(cell2core(tt_tensor, Y1) - cell2core(tt_tensor, Y2))^2/(2*g^2));

g(3) = g(1);
for k=1:Order
%    Kernel_value = Kernel_value*exp(-norm(X1{k}(:)-X2{k}(:))^2/(2*g^2));
    
    R =  size(X1{k}, 2); % r = r1*r2
    Q =  size(X2{k}, 2); % q = q1*q2
    r1 = size(Y1{k}, 1);
    r2 = size(Y1{k}, 3);
    q1 = size(Y2{k}, 1);
    q2 = size(Y2{k}, 3);
    for s = 1:R
        for t = 1:Q  
           % format long
           % Kcore{k}(s,t)= exp(-norm(X1{k}(:,s)-X2{k}(:,t)).^2/(2*g.^2));
           Kcore{k}(s,t)= exp(-norm(X1{k}(:,s)-X2{k}(:,t))^2/(2*g(k)^2));
        end
    end
     K_tt{k} = reshape(Kcore{k}, r1,r2,q1,q2);
     K_tt{k} = permute(K_tt{k}, [1, 3, 2, 4]);
     K_tt{k} = reshape(K_tt{k},r1*q1,1,r2*q2);   
end
% this output goes to training of input data set into TrainAvgAcu
  K_ttcore =  cell2core(tt_tensor, K_tt);
  Kernel_value = full(K_ttcore);
end
% 
%     
%     
%     