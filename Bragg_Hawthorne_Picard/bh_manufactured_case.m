function [g,source,side,exact] = bh_manufactured_case(n,stretch)
% Smooth nonlinear manufactured benchmark, not a Dust Devil prediction.
if nargin<2, stretch=1; end
g=build_collision_grid(struct('Rmax',1,'Zmax',1,'Nr',n,'Nz',n,'pr',stretch,'pz',stretch));
[R,Z]=ndgrid(g.r,g.z);
exact=R.^2.*sin(pi*Z/2); side=sin(pi*g.z/2);
source=@(r,z,psi) -(pi/2)^2*r.^2.*sin(pi*z/2) ...
    +0.5*(psi.^2-(r.^2.*sin(pi*z/2)).^2);
end
