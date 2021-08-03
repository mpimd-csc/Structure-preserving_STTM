% demo 4 to test cp3_alsls algorithm.
% The purpose of this demo is to show how to exploit several initializations.
% The function cp3_alsls has 2 kinds of optional input arguments to deal with initialization:
% - Ninit : the algorithm will generate Ninit different starting points and for
%   each starting point, it will stop when the stop criterion is satisfied 
%  (input parameters Tol1 and MaxIt1 to select tolerance and max number of iterations).
%  Output parameters are the ones associated to the best starting point.
% - A,B,C: if one or more matrices are provided, the latter are used as a starting point
%   and this is the only starting point that will be tried, no matter the value of Ninit
%  (it is for instance useful when one already knows estimates close to the solution).

% When one uses e.g. Ninit=10, it can take time before the algorithm stops because
% each starting point will be tried until the convergence criterion is satisfied.
% Thus, it can be useful in practice to try Ninit=10 starting points for a limited
% number of iterations, say MaxItInit (less than MaxIt1) and then go on until convergence
% with only the best initialization obtained among the Ninit tried.
% This demo show how to to this.


clear all
close all
clc

%**********************************************
%--- Choose PARAMETERS of the DEMO
%**********************************************
%--- data parameters
    data_type='complex';       % choose 'real' or 'complex' to select the kind of data to generate
    I=9;                      % Dimensions  of the tensor
    J=10;
    K=40;           
    R=15;                     % Rank of the tensor
    SNR=inf;                   % SNR [dB], choose SNR=inf for a noise-free model
    condA=10;                 % Impose a condition number on the loading matrices:
    condB=10;                 % the highest these values, the more likely swamps will appear 
    condC=10;
   
%***************************************************
%---- Build Loading matrices and observed tensor-----
%**************************************************** 
    if strcmp(data_type,'real')==1
        A=randn(I,R);B=randn(J,R);C=randn(K,R);
    elseif strcmp(data_type,'complex')==1
        A=randn(I,R)+j*randn(I,R);B=randn(J,R)+j*randn(J,R);C=randn(K,R)+j*randn(K,R);
    end
   
    % Now impose a condition number on the matrices
    [Ua,Sa,Va]=svd(A,0);
    [Ub,Sb,Vb]=svd(B,0);
    [Uc,Sc,Vc]=svd(C,0);
    Sa(1:min(I,R),1:min(I,R))=diag(linspace(condA,1,min(I,R)));   % change singular values (linearly spaced new values)
    Sb(1:min(J,R),1:min(J,R))=diag(linspace(condB,1,min(J,R)));   % change singular values (linearly spaced new values)
    Sc(1:min(K,R),1:min(K,R))=diag(linspace(condC,1,min(K,R)));   % change singular values (linearly spaced new values)
    A=Ua*Sa*Va';
    B=Ub*Sb*Vb';
    C=Uc*Sc*Vc';
    
    % Create observed tensor that follows PARAFAC model
    X=zeros(I,J,K);
    for k=1:K
        X(:,:,k)=A*diag(C(k,:))*B.';
    end

    % Add noise 
    if strcmp(data_type,'real')==1
        Noise_tens=randn(I,J,K);
    elseif strcmp(data_type,'complex')==1
        Noise_tens=randn(I,J,K)+j*randn(I,J,K);
    end
    sigma=(10^(-SNR/20))*(norm(reshape(X,J*I,K),'fro')/norm(reshape(Noise_tens,J*I,K),'fro'));
    X=X+sigma*Noise_tens;

%--------------------------------------------------------------------------------
% COMPUTE THE CP3 decomposition of X with Ninit initializations and with MaxItInit 
% iterations for each initialization, and then go on until convergence with the
% best initialization only
%--------------------------------------------------------------------------------
Ninit=10;
MaxItInit=10;
% Perform at most MaxItInit iterations for each starting point
[A_init,B_init,C_init,phi,itInit]=cp3_alsls(X,R,[],[],[],MaxItInit,[],[],Ninit);
% Then go on with the best starting point
[A_est,B_est,C_est,phi,it1]=cp3_alsls(X,R,[],[],[],[],[],[],[],A_init,B_init,C_init);

[temp,temp,temp,err_A]=solve_perm_scale(A_est,A);
[temp,temp,temp,err_B]=solve_perm_scale(B_est,B);
[temp,temp,temp,err_C]=solve_perm_scale(C_est,C);
disp(['err on A:         ',num2str(err_A)])
disp(['err on B:         ',num2str(err_B)])
disp(['err on C:         ',num2str(err_C)])
disp(['itInit:           ',num2str(itInit)])
disp(['it1:              ',num2str(it1)])