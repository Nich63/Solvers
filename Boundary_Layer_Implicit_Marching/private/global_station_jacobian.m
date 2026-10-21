function jacobian = global_station_jacobian( ...
    x,YPrevious,z,zetaPrevious,zetaCurrent,cfg)
% Sparse analytic Jacobian of global_station_residual.

if ~isstruct(cfg) || ~isscalar(cfg)
    error('global_station_jacobian:InvalidConfig', ...
        'cfg must be a scalar structure.');
end
if ~isfield(cfg,'Pr') || isempty(cfg.Pr)
    error('global_station_jacobian:MissingPr','cfg.Pr is required.');
end
if ~isfield(cfg,'Ri') || isempty(cfg.Ri)
    error('global_station_jacobian:MissingRi','cfg.Ri is required.');
end

Pr = cfg.Pr;
Ri = cfg.Ri;

if ~isnumeric(Pr) || ~isscalar(Pr) || ~isfinite(Pr) || Pr <= 0
    error('global_station_jacobian:InvalidPr', ...
        'cfg.Pr must be a finite positive scalar.');
end
if ~isnumeric(Ri) || ~isscalar(Ri) || ~isfinite(Ri) || Ri < 0
    error('global_station_jacobian:InvalidRi', ...
        'cfg.Ri must be a finite nonnegative scalar.');
end

if ~isnumeric(z) || ~isvector(z) || numel(z) < 2
    error('global_station_jacobian:InvalidZGrid', ...
        'z must be a numeric vector containing at least two nodes.');
end
z = reshape(z,1,[]);
nZ = numel(z);
dZ = diff(z);
if any(~isfinite(z)) || abs(z(1)) > 100*eps(max(1,z(end))) ...
        || any(dZ <= 0)
    error('global_station_jacobian:InvalidZGrid', ...
        'z must start at zero and be finite and strictly increasing.');
end

if ~isnumeric(x) || numel(x) ~= 8*nZ
    error('global_station_jacobian:InvalidCurrentState', ...
        'x must contain exactly 8*numel(z) entries.');
end
if ~isnumeric(YPrevious) || ~isequal(size(YPrevious),[8,nZ])
    error('global_station_jacobian:InvalidPreviousState', ...
        'YPrevious must have size 8-by-numel(z).');
end

if ~isnumeric(zetaPrevious) || ~isscalar(zetaPrevious) ...
        || ~isfinite(zetaPrevious) || zetaPrevious <= 0
    error('global_station_jacobian:InvalidPreviousZeta', ...
        'zetaPrevious must be a finite positive scalar.');
end
if ~isnumeric(zetaCurrent) || ~isscalar(zetaCurrent) ...
        || ~isfinite(zetaCurrent) || zetaCurrent <= zetaPrevious ...
        || zetaCurrent >= 1
    error('global_station_jacobian:InvalidCurrentZeta', ...
        'zetaCurrent must satisfy zetaPrevious < zetaCurrent < 1.');
end

dZeta = zetaCurrent-zetaPrevious;
rCurrent = 1-zetaCurrent;
YCurrent = reshape(x,8,nZ);

% The same midpoint and derivative quantities GLOBAL_STATION_RESIDUAL
% forms, since the nonlinear terms differentiate into them.
YMid = 0.5*(YCurrent(:,1:end-1)+YCurrent(:,2:end));
YPreviousMid = 0.5*(YPrevious(:,1:end-1)+YPrevious(:,2:end));
DzetaY = (YMid-YPreviousMid)/dZeta;

U = YMid(1,:);
A = YMid(2,:);
V = YMid(3,:);
B = YMid(4,:);
W = YMid(5,:);
H = YMid(7,:);

DzetaU = DzetaY(1,:);
DzetaV = DzetaY(3,:);
DzetaTheta = DzetaY(6,:);

% Discrete operator weights. mMid applies to both bounding nodes, dDz has
% opposite signs at the two nodes, and tDzeta applies to both.
nCell = nZ-1;
mMid = 0.5;
dDz = 1./dZ;
tDzeta = 0.5/dZeta;
r = rCurrent;

% State positions inside a node block, matching Y = [U;A;V;B;W;Theta;H;pb].
iU = 1; iA = 2; iV = 3; iB = 4; iW = 5; iTheta = 6; iH = 7; iPb = 8;

% Each cell contributes 46 nonzeros; the eight boundary rows contribute one
% each.
nnzTotal = 8+46*nCell;
iRow = zeros(nnzTotal,1);
iCol = zeros(nnzTotal,1);
val = zeros(nnzTotal,1);

% Boundary rows, in the order used by GLOBAL_STATION_RESIDUAL: U, V, W and
% Theta at the wall, then U, V, Theta and pb in the far field. Every one is
% linear with unit derivative.
last = 8*(nZ-1);
iRow(1:8) = (1:8).';
iCol(1:8) = [iU; iV; iW; iTheta; last+iU; last+iV; last+iTheta; last+iPb];
val(1:8) = 1;
ptr = 8;

cellIndex = 1:nCell;
rowBase = 8+(cellIndex-1)*8;
colBaseLeft = (cellIndex-1)*8;
colBaseRight = cellIndex*8;

% Equation 1: U_z - A = 0.
addEntries(1,iU,false,-dDz);
addEntries(1,iU,true,dDz);
addEntries(1,iA,false,-mMid);
addEntries(1,iA,true,-mMid);

% Equation 2: V_z - B = 0.
addEntries(2,iV,false,-dDz);
addEntries(2,iV,true,dDz);
addEntries(2,iB,false,-mMid);
addEntries(2,iB,true,-mMid);

% Equation 3: Theta_z - H = 0.
addEntries(3,iTheta,false,-dDz);
addEntries(3,iTheta,true,dDz);
addEntries(3,iH,false,-mMid);
addEntries(3,iH,true,-mMid);

% Equation 4: continuity, W_z - U_zeta = 0.
addEntries(4,iW,false,-dDz);
addEntries(4,iW,true,dDz);
addEntries(4,iU,false,-tDzeta);
addEntries(4,iU,true,-tDzeta);

% Equation 5: hydrostatic pressure, pb_z - Ri*Theta = 0.
addEntries(5,iPb,false,-dDz);
addEntries(5,iPb,true,dDz);
addEntries(5,iTheta,false,-Ri*mMid);
addEntries(5,iTheta,true,-Ri*mMid);

ct = global_convection_treatment(YPrevious,YCurrent,zetaCurrent,cfg);
if strcmp(ct.mode,'standard')
    convU6 = DzetaU*mMid+U*tDzeta;
    convU7 = DzetaV*mMid;  convQ7 = U*tDzeta;
    convU8 = DzetaTheta*mMid;  convQ8 = U*tDzeta;
else
    cU = ct.coefU;
    dD = tDzeta*ones(1,nCell);
    DU = DzetaU; DV = DzetaV; DT = DzetaTheta;
    if any(ct.useDown)
        d = ct.useDown;
        dD(d) = -mMid/ct.dZetaDown;
        DU(d) = (ct.downMid(iU,d)-U(d))/ct.dZetaDown;
        DV(d) = (ct.downMid(iV,d)-V(d))/ct.dZetaDown;
        DT(d) = (ct.downMid(iTheta,d)-YMid(iTheta,d))/ct.dZetaDown;
    end
    convU6 = mMid*cU.*DU+cU.*U.*dD;
    convU7 = mMid*cU.*DV;  convQ7 = cU.*U.*dD;
    convU8 = mMid*cU.*DT;  convQ8 = cU.*U.*dD;
end

% Equation 6: radial momentum,
%   r*A_z + U*U_zeta - W*A + U^2/r + r^2*pb_zeta - (1-V^2)/r = 0.
addEntries(6,iA,false,-r*dDz-W*mMid);
addEntries(6,iA,true,r*dDz-W*mMid);
addEntries(6,iU,false,convU6+2*U*mMid/r);
addEntries(6,iU,true,convU6+2*U*mMid/r);
addEntries(6,iW,false,-A*mMid);
addEntries(6,iW,true,-A*mMid);
addEntries(6,iV,false,2*V*mMid/r);
addEntries(6,iV,true,2*V*mMid/r);
addEntries(6,iPb,false,r^2*tDzeta);
addEntries(6,iPb,true,r^2*tDzeta);

% Equation 7: azimuthal momentum, r*B_z + U*V_zeta - W*B = 0.
addEntries(7,iB,false,-r*dDz-W*mMid);
addEntries(7,iB,true,r*dDz-W*mMid);
addEntries(7,iU,false,convU7);
addEntries(7,iU,true,convU7);
addEntries(7,iV,false,convQ7);
addEntries(7,iV,true,convQ7);
addEntries(7,iW,false,-B*mMid);
addEntries(7,iW,true,-B*mMid);

% Equation 8: energy, (r/Pr)*H_z + U*Theta_zeta - W*H = 0.
addEntries(8,iH,false,-(r/Pr)*dDz-W*mMid);
addEntries(8,iH,true,(r/Pr)*dDz-W*mMid);
addEntries(8,iU,false,convU8);
addEntries(8,iU,true,convU8);
addEntries(8,iTheta,false,convQ8);
addEntries(8,iTheta,true,convQ8);
addEntries(8,iW,false,-H*mMid);
addEntries(8,iW,true,-H*mMid);

if ptr ~= nnzTotal
    error('global_station_jacobian:AssemblyMismatch', ...
        'Assembled %d entries but preallocated %d.',ptr,nnzTotal);
end

jacobian = sparse(iRow,iCol,val,8*nZ,8*nZ);

    function addEntries(equation,state,isRightNode,contribution)

    if isRightNode
        columns = colBaseRight+state;
    else
        columns = colBaseLeft+state;
    end

    span = ptr+(1:nCell);
    iRow(span) = (rowBase+equation).';
    iCol(span) = columns.';
    if isscalar(contribution)
        val(span) = contribution;
    else
        val(span) = contribution.';
    end
    ptr = ptr+nCell;
    end

end
