% demo 2 to test cp3_alsls algorithm.
% The purpose of this demo is to show how to use all default parameters
clear all
close all
clc

%**********************************************
%--- Choose PARAMETERS of the DEMO
%**********************************************
%--- data parameters
    data_type='complex';       % choose 'real' or 'complex' to select the kind of data to generate
    I=14;                      % Dimensions  of the tensor
    J=20;
    K=21;           
    R=12;                     % Rank of the tensor
    SNR=60;                   % SNR [dB], choose SNR=inf for a noise-free model
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
% COMPUTE THE CP3 decomposition of X with all default parameters
%--------------------------------------------------------------------------------
[A_est,B_est,C_est]=cp3_alsls(X,R);
[temp,temp,temp,err_A]=solve_perm_scale(A_est,A);
[temp,temp,temp,err_B]=solve_perm_scale(B_est,B);
[temp,temp,temp,err_C]=solve_perm_scale(C_est,C);
disp(['err on A:         ',num2str(err_A)])
disp(['err on B:         ',num2str(err_B)])
disp(['err on C:         ',num2str(err_C)])