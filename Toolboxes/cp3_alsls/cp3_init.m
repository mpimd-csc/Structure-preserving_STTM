function [A,B,C]=cp3_init(X,R,init_type)
%CP3_INIT Initialization of the loading matrices for the CP3 decomposition of X
% INPUTS: 
% - X: 3rd order tensor of size (IxJxK)
% - R: Number of rank-1 components in the PARAFAC model
% - init_type='dtld' to initialize by Direct Trilinear Decomposition if possible,
%   otherwise randomly,
% - init_type='random' to enforce random initialization.
% OUTPUTS:
% A(IxR) B(JxR) and C(KxR): Loading Matrices

[I,J,K]=size(X);
% Check if 2 dimensions are greater than the rank to see if DTLD can be used
[size_sort,perm_vec]=sort([I J K],'descend');
if ((size_sort(2))<R) 
    init_type='random';    % DTLD can not be used so use random init
end
        
if strcmp(init_type,'dtld')==1
    [A,B,C] = cp3_dtld(X,R);    
elseif strcmp(init_type,'random')==1
    if isreal(X); A=rand(I,R);B=rand(J,R);C=rand(K,R);
    else;A=rand(I,R)+j*rand(I,R);B=rand(J,R)+j*rand(J,R);C=rand(K,R)+j*rand(K,R);
    end
    % Orthogonal matrices if possible else well conditioned matrices
    if I>=R;A=orth(A);else;A=orth(A')';end
    if J>=R;B=orth(B);else;B=orth(B')';end
    if K>=R;C=orth(C);else;C=orth(C')';end
end
end

%*******************************************************************************
function [A,B,C] = cp3_dtld(X,R)
%CP3_DTLD CANDECOMP/PARAFAC by Direct Trilinear Decomposition (DTLD).
%   [A,B,C]=cp3_dtld(X,R) computes the DTLD of a third-order tensor X.
%   DTLD is a method to decompose X as a sum of R rank-1 tensors, i.e., it
%   (non-optimally) fits a rank-R candecomp/parafac (cp) model to X.
%   A, B, C are estimates of the mode-1, mode-2 and mode-3 loading matrices
%   of the cp model, respectively.
%
%   DTLD can only be used if at least two dimensions of the input tensor X 
%   are greater than or equal to the rank R of the decomposition.
%
%   DTLD already gives the exact solution in the case of an exact cp3
%   model, given that the loading matrices in the two long modes are full
%   column rank.
%
%   DTLD computes a truncated mlsvd of X with a core tensor of dimensions 
%   R by R by 2, after which A, B and C are obtained from the generalized 
%   eigenvalue decomposition (GEVD) of the matrix pencil formed by the 
%   two slices of this tensor.
%
%   DTLD can be used as a good starting point for optimization algorithms
%   designed to (optimally) fit the cp model.

%   Copyright 2010
%   Version: 12-07-10
%   Authors: Dimitri Nion (dimitri.nion@gmail.com)
%
%   References:
%   [1] E. Leurgans, R.T. Ross, and R.B. Abel, "A decomposition for 
%       three-way arrays", SIAM J. Matrix Anal. Appl., 14 (1993), 
%       pp. 1064-1083.

[I,J,K]=size(X);
% Check if DTLD is applicable
[size_sort,perm_vec]=sort([I J K],'descend');
if ((size_sort(2))<R) 
    error('cp3_init_dtld:InvalidDimensions',['The input tensor ' ...
          'must have at least two dimensions greater than or equal to R']);
end
% Now Rotate X such that the shortest dimension is in the 3 rd mode
X_perm=permute(X,perm_vec);
[I1,J1,K1]=size(X_perm);
% Truncated MLSVD
[U1,U2,U3,X_core] = mlsvd3(X_perm,[R,R,2]);
% Generalized EVD of the matrix pencil formed by these two slices
[Bti,S]=eig(X_core(:,:,1),X_core(:,:,2));
B=pinv(Bti).';
s=diag(S);
[a ind]=sort(abs(s),'descend');
s=s(ind);
B = B(:,ind);
% If X is real but B is complex (pairs of conjugate eigenvalues)
% then transform B with a similarity matrix to get a real-valued matrix
if isreal(s)~=1 && isreal(X)==1  
     r=1;
     T=[0.5 -1i*0.5;0.5 1i*0.5];  % 2 by 2 block: similarity matrix
     while r<=R
          if isreal(s(r))~=1
              B(:,r:r+1)= B(:,r:r+1)*...
                          diag(exp(-1i*sum(angle(B(1,r:r+1)))/2))*T;   
              r=r+2;
          else
              r=r+1;
          end
     end       
     B=real(B);
end
% find A
A=X_core(:,:,1)/B.';
% expand to the original space
A=U1*A;
B=U2*B;
% find C in original space
C = (((B'*B).*(A'*A))\(kr(B,A)'*reshape(X_perm,I1*J1,K1))).';
% Now place the matrices A, B and C in the right mode from perm_vec
L=cell(1,3);L{1}=A;L{2}=B;L{3}=C;
L(:,perm_vec)=L;
A=L{1};B=L{2};C=L{3};
end

%*******************************************************************************       
function C = kr(A,B)
%KR Khatri-Rao product.
[I R1]=size(A); J=size(B,1); 
C=zeros(I*J,R1);
for j=1:R1
    C(:,j)=reshape(B(:,j)*A(:,j).',I*J,1);
end
end
%*******************************************************************************
function [U1,U2,U3,S,S1,S2,S3] = mlsvd3(X,size_core)
%MLSVD3 Multilinear singular value decomposition of a third-order tensor.
[I1,I2,I3]=size(X);
[U1,S1,temp]=svd(reshape(X,I1,I3*I2),'econ'); S1=diag(S1);
[U2,S2,temp]=svd(reshape(permute(X,[2 3 1]),I2,I1*I3),'econ'); S2=diag(S2);
[U3,S3,temp]=svd(reshape(permute(X,[3 1 2]),I3,I2*I1),'econ'); S3=diag(S3);
if nargin==2
    U1=U1(:,1:min(size_core(1),I2*I3));
    U2=U2(:,1:min(size_core(2),I1*I3));
    U3=U3(:,1:min(size_core(3),I1*I2));
end
S=tmprod(tmprod(tmprod(X,U1',1),U2',2),U3',3);
end

%*******************************************************************************
function X_out = tmprod(X,U,mode)
%TMPROD mode-n tensor-matrix product.
[I,J,K]=size(X);
[M,N]=size(U);
if (mode~=1) && (mode~=2) && (mode~=3)
    error('The input variable mode should be 1, 2 or 3')
end
if N~=size(X,mode) 
    error(['The number of columns of the input matrix should be equal to dimension ',int2str(mode),' of the input tensor'])
end    
if mode==1
    X_out = reshape(U*reshape(X,I,J*K) ,M,J,K);      
elseif mode==2
    X_out = permute(reshape (U*reshape(permute(X,[2 1 3]),J,I*K), M,I,K),[2 1 3]);        
elseif mode==3
    X_out = permute(reshape (U*reshape(permute(X,[3 1 2]),K,I*J), M,I,J),[2 3 1]);
end
end
