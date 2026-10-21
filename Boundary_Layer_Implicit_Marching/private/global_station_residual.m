function [residual,details] = global_station_residual( ...
    x,YPrevious,z,zetaPrevious,zetaCurrent,cfg)
% Single-station residual; state order [U; A; V; B; W; Theta; H; pb].
% Midpoint differences in z; backward Euler in zeta=1-r.

if ~isstruct(cfg) || ~isscalar(cfg)
    error('global_station_residual:InvalidConfig', ...
        'cfg must be a scalar structure.');
end
if ~isfield(cfg,'Pr') || isempty(cfg.Pr)
    error('global_station_residual:MissingPr','cfg.Pr is required.');
end
if ~isfield(cfg,'Ri') || isempty(cfg.Ri)
    error('global_station_residual:MissingRi','cfg.Ri is required.');
end

Pr = cfg.Pr;
Ri = cfg.Ri;

if ~isnumeric(Pr) || ~isscalar(Pr) || ~isfinite(Pr) || Pr <= 0
    error('global_station_residual:InvalidPr', ...
        'cfg.Pr must be a finite positive scalar.');
end
if ~isnumeric(Ri) || ~isscalar(Ri) || ~isfinite(Ri) || Ri < 0
    error('global_station_residual:InvalidRi', ...
        'cfg.Ri must be a finite nonnegative scalar.');
end

if ~isnumeric(z) || ~isvector(z) || numel(z) < 2
    error('global_station_residual:InvalidZGrid', ...
        'z must be a numeric vector containing at least two nodes.');
end
z = reshape(z,1,[]);
nZ = numel(z);
dZ = diff(z);
if any(~isfinite(z)) || abs(z(1)) > 100*eps(max(1,z(end))) ...
        || any(dZ <= 0)
    error('global_station_residual:InvalidZGrid', ...
        'z must start at zero and be finite and strictly increasing.');
end

if ~isnumeric(x) || numel(x) ~= 8*nZ
    error('global_station_residual:InvalidCurrentState', ...
        'x must contain exactly 8*numel(z) entries.');
end
if ~isnumeric(YPrevious) || ~isequal(size(YPrevious),[8,nZ])
    error('global_station_residual:InvalidPreviousState', ...
        'YPrevious must have size 8-by-numel(z).');
end

if ~isnumeric(zetaPrevious) || ~isscalar(zetaPrevious) ...
        || ~isfinite(zetaPrevious) || zetaPrevious <= 0
    error('global_station_residual:InvalidPreviousZeta', ...
        'zetaPrevious must be a finite positive scalar.');
end
if ~isnumeric(zetaCurrent) || ~isscalar(zetaCurrent) ...
        || ~isfinite(zetaCurrent) || zetaCurrent <= zetaPrevious ...
        || zetaCurrent >= 1
    error('global_station_residual:InvalidCurrentZeta', ...
        'zetaCurrent must satisfy zetaPrevious < zetaCurrent < 1.');
end

dZeta = zetaCurrent-zetaPrevious;
rCurrent = 1-zetaCurrent;
YCurrent = reshape(x,8,nZ);

% Vertical midpoint values at the current and previous radial stations.
YMid = 0.5*(YCurrent(:,1:end-1)+YCurrent(:,2:end));
YPreviousMid = 0.5*(YPrevious(:,1:end-1)+YPrevious(:,2:end));

% Cell derivatives. Implicit expansion divides each column by the
% corresponding nonuniform vertical cell width.
DzY = (YCurrent(:,2:end)-YCurrent(:,1:end-1))./dZ;
DzetaY = (YMid-YPreviousMid)/dZeta;

U = YMid(1,:);
A = YMid(2,:);
V = YMid(3,:);
B = YMid(4,:);
W = YMid(5,:);
Theta = YMid(6,:);
H = YMid(7,:);

% Each column contains the eight governing-equation residuals on one
% vertical Keller box. The row order matches the documentation below, not
% the state-vector order.
cellResidual = zeros(8,nZ-1);

% Definitions: U_z = A, V_z = B, Theta_z = H.
cellResidual(1,:) = DzY(1,:)-A;
cellResidual(2,:) = DzY(3,:)-B;
cellResidual(3,:) = DzY(6,:)-H;

% Continuity and hydrostatic pressure.
cellResidual(4,:) = DzY(5,:)-DzetaY(1,:);
cellResidual(5,:) = DzY(8,:)-Ri*Theta;

% Convective zeta-derivatives. Only in reversed-flow cells under
% cfg.convectionTreatment = 'flare'/'duflare' do these differ from DzetaY
% (formulation.md Sec 4); 'standard' leaves every value untouched.
ct = global_convection_treatment(YPrevious,YCurrent,zetaCurrent,cfg);
UConv = U;
DConv = DzetaY;
if ~strcmp(ct.mode,'standard')
    UConv = ct.coefU.*U;
    if any(ct.useDown)
        DConv(:,ct.useDown) = (ct.downMid(:,ct.useDown) ...
            -YMid(:,ct.useDown))/ct.dZetaDown;
    end
end

% Radial momentum.
cellResidual(6,:) = ...
    rCurrent*DzY(2,:) ...
    + UConv.*DConv(1,:) ...
    - W.*A ...
    + U.^2/rCurrent ...
    + rCurrent^2*DzetaY(8,:) ...
    - (1-V.^2)/rCurrent;

% Azimuthal momentum.
cellResidual(7,:) = ...
    rCurrent*DzY(4,:) ...
    + UConv.*DConv(3,:) ...
    - W.*B;

% Energy.
cellResidual(8,:) = ...
    (rCurrent/Pr)*DzY(7,:) ...
    + UConv.*DConv(6,:) ...
    - W.*H;

% Four wall conditions followed by four far-field conditions. There is no
% far-field boundary condition on W.
boundaryResidual = [ ...
    YCurrent(1,1); ...
    YCurrent(3,1); ...
    YCurrent(5,1); ...
    YCurrent(6,1)-1; ...
    YCurrent(1,end); ...
    YCurrent(3,end)-1; ...
    YCurrent(6,end); ...
    YCurrent(8,end)];

residual = [boundaryResidual; cellResidual(:)];

if nargout > 1
    details.stateNames = {'U','A','V','B','W','Theta','H','pb'};
    details.equationNames = { ...
        'U_z-A', ...
        'V_z-B', ...
        'Theta_z-H', ...
        'continuity', ...
        'hydrostatic pressure', ...
        'radial momentum', ...
        'azimuthal momentum', ...
        'energy'};
    details.boundaryNames = { ...
        'U_wall', ...
        'V_wall', ...
        'W_wall', ...
        'Theta_wall-1', ...
        'U_far', ...
        'V_far-1', ...
        'Theta_far', ...
        'pb_far'};
    details.boundary = boundaryResidual;
    details.cell = cellResidual;
    details.boundaryInfNorm = norm(boundaryResidual,inf);
    details.cellInfNorm = norm(cellResidual(:),inf);
    details.totalInfNorm = norm(residual,inf);
    details.dZeta = dZeta;
    details.rCurrent = rCurrent;
end

end
