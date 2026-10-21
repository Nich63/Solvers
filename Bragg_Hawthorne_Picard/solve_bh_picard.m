function out = solve_bh_picard(g,source,psiSide,cfg)
% Solve Delta*psi=source(r,z,psi). Sparse LU is reused at every iteration.
% Boundary contract is in private/assemble_bh_system.m and README.md.
if nargin<4, cfg=struct(); end
if ~isfield(cfg,'relax'), cfg.relax=0.7; end
if ~isfield(cfg,'tol'), cfg.tol=1e-9; end
if ~isfield(cfg,'maxIter'), cfg.maxIter=1000; end
validateattributes(cfg.relax,{'numeric'},{'scalar','>',0,'<=',1});
validateattributes(cfg.tol,{'numeric'},{'scalar','positive','finite'});
validateattributes(cfg.maxIter,{'numeric'},{'scalar','integer','positive'});
[A,b0]=assemble_bh_system(g,0,psiSide); factor=decomposition(A,'lu');
[R,Z]=ndgrid(g.r,g.z); interior=false(g.Nr,g.Nz); interior(2:end-1,2:end-1)=true;
idx=find(interior); bc=find(~interior);
psi=zeros(g.Nr,g.Nz); history=nan(cfg.maxIter,2); converged=false;
for k=1:cfg.maxIter
    value=source(R,Z,psi); b=b0; b(idx)=value(idx);
    candidate=reshape(factor\b,g.Nr,g.Nz);
    next=(1-cfg.relax)*psi+cfg.relax*candidate;
    value=source(R,Z,next); b(idx)=value(idx);
    residual=A*next(:)-b;
    history(k,:)=[norm(residual(idx),inf),norm(residual(bc),inf)];
    psi=next;
    if ~all(isfinite(psi(:))), error('BH:Nonfinite','Picard iteration diverged.'); end
    if max(history(k,:))<cfg.tol, converged=true; break; end
end
out=struct('psi',psi,'g',g,'history',history(1:k,:),'iterations',k, ...
    'converged',converged,'pdeResidual',history(k,1),'boundaryResidual',history(k,2));
end
