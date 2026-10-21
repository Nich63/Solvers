function g = build_collision_grid(cfg)
% Build r=Rmax*s^pr, z=Zmax*t^pz with uniform s,t in [0,1].
% The axis is Dirichlet; the 1/r operator is evaluated only for r>0.

if nargin < 1; cfg = struct(); end
g.Rmax = getdef(cfg,'Rmax',8);
g.Zmax = getdef(cfg,'Zmax',16);
g.Nr   = getdef(cfg,'Nr',161);
g.Nz   = getdef(cfg,'Nz',161);
g.pr   = getdef(cfg,'pr',1.5);
g.pz   = getdef(cfg,'pz',1.5);

s = linspace(0,1,g.Nr).';
t = linspace(0,1,g.Nz).';
g.r = g.Rmax*s.^g.pr;
g.z = g.Zmax*t.^g.pz;

g.N = g.Nr*g.Nz;
g.idx = @(i,j) (j-1)*g.Nr + i;

end

function v = getdef(s,f,d)
if isfield(s,f) && ~isempty(s.(f)); v = s.(f); else; v = d; end
end
