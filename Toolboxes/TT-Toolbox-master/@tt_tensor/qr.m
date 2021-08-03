function [tt,rm]=qr(tt,op)
%Left and right orthogonalization of the TT-format
%   [TT,RM]=QR(TT,OP) Implements the TT-QR algorithm for the 
%   orthogonalization of the TT-representation. OP can be 'LR' 
%   (for the left-right orthogonalization) and 'RL' for the 
%   right-left orthogonalization. Also returns R matrix for the block case,
%   when r(d) or r(1) are not equal to 1.     
%
%
% TT-Toolbox 2.2, 2009-2012
%
%This is TT Toolbox, written by Ivan Oseledets et al.
%Institute of Numerical Mathematics, Moscow, Russia
%webpage: http://spring.inm.ras.ru/osel
%
%For all questions, bugs and suggestions please mail
%ivan.oseledets@gmail.com
%---------------------------

d=tt.d;
r=tt.r;
cr=tt.core;
ps=tt.ps;
n=tt.n;
if ( strcmp(op,'lr') || strcmp(op,'LR')) 
  %Orthogonalization from left-to-tight
  core0=cr(ps(1):ps(2)-1);
  pos1=1;
  for i=1:d-1 
   core0=reshape(core0,[r(i)*n(i),r(i+1)]);
   [core0,ru]=qr(core0,0); %nrm(i+1)=norm(ru,'fro');
   %ru=ru;%./nrm(i+1);
   core1=cr(ps(i+1):ps(i+2)-1);
   core1=reshape(core1,[r(i+1),n(i+1)*r(i+2)]);
   core1=ru*core1;
   r(i+1)=size(core0,2);
   cr(pos1:pos1-1+r(i)*n(i)*r(i+1))=core0(:);
   cr(pos1+r(i)*n(i)*r(i+1):pos1+r(i)*n(i)*r(i+1)+r(i+1)*n(i+1)*r(i+2)-1)=core1(:);
   core0=core1;
   pos1=pos1+r(i)*n(i)*r(i+1);
  end
  %pos1 points to the d-th core
  core0=cr(pos1:pos1-1+r(d)*n(d)*r(d+1));
  core0=reshape(core0,[r(d)*n(d),r(d+1)]);
  [core0,rm]=qr(core0,0);
  r(d+1)=size(core0,2);
  cr(pos1:pos1+r(d)*n(d)*r(d+1)-1)=core0(:);
 pos1=pos1+r(d)*n(d)*r(d+1)-1;
 cr=cr(1:pos1); %Truncate storage if required
 tt.core=cr; tt.r=r; 
 tt.ps=cumsum([1;tt.n.*tt.r(1:d).*tt.r(2:d+1)]);
 return
elseif (strcmp (op,'rl') || strcmp(op,'RL') )
    %Orthogonalization from right-to-left
    rm=1;
    pos1=ps(d+1)-1;
 for i=d:-1:1
    cr1=cr(ps(i):ps(i+1)-1);
    cr1=reshape(cr1,[r(i)*n(i),r(i+1)]);
    cr1=cr1*rm; 
    r(i+1)=size(cr1,2); 
    cr1=reshape(cr1,[r(i),n(i)*r(i+1)]);
    cr1=cr1.'; %It seems to be needed
    [u,rm]=qr(cr1,0); 
    rm=rm.';
    rnew=size(u,2);
    u=u.'; %n(i)*ry(i+1)xry(i)->ry(i)xn(i)xry(i+1)     
    cr(pos1-r(i+1)*n(i)*rnew+1:pos1)=u(:); 
    pos1=pos1-r(i+1)*n(i)*rnew; 
 end
 pos1 = pos1+1;
% pos1=pos1+r(1)*n(1)*r(2)-1;
cr=cr(pos1:end);
ps=cumsum([1;(n).*r(1:d).*r(2:d+1)]);
tt.core=cr;
tt.r=r;
tt.ps=ps;
return
else
  error('Option is not supported (yet)');    
end
