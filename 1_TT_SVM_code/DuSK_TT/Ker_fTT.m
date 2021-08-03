function ValueOfKernel = Ker_fTT(X1,X2,Order,g,R,flagker);
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
% Task 2: defining structure preserving mapping or getting dual structure
% preserving kernel
% i.e. \phi :X -> \phi(X) \in tensor product space
%%  DuSK kernel computing
if nargin==3
 ValueOfKernel=X1'*X2;
else
    if flagker==0
        S=ones(R,R); 
        for k=1:Order
            S=S.*(X1{k,1}'*X2{k,1});           % X1 and X2 is Dk*R size
        end
        ValueOfKernel=sum(sum(S));
    else
        ValueOfKernel=0;
        for i=1:R
            for j=1:R
                S=1;
                for k=1:Order
                    S=S*exp(-norm(X1{k,1}(:,i)-X2{k,1}(:,j))^2/(2*g^2));
                end
                ValueOfKernel=ValueOfKernel+S;
            end
        end
    end
end
end

