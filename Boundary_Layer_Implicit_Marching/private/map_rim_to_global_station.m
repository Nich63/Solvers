function station = map_rim_to_global_station(rim,z,cfg)
% Map the rim BVP to zeta0 using cfg.Ri and the physical z grid.
% Beyond etaMax use the asymptotic state; far-field W is not zero.

arguments
    rim (1,1) struct
    z {mustBeNumeric,mustBeReal,mustBeFinite}
    cfg (1,1) struct
end

validateRim(rim);

if ~isfield(cfg,'Ri') || isempty(cfg.Ri)
    error('map_rim_to_global_station:MissingRi', ...
        'cfg.Ri is required.');
end
if ~isfield(cfg,'zeta0') || isempty(cfg.zeta0)
    error('map_rim_to_global_station:MissingZeta0', ...
        'cfg.zeta0 is required.');
end

Ri = cfg.Ri;
zeta0 = cfg.zeta0;

validateattributes(Ri,{'numeric'}, ...
    {'scalar','real','finite','positive'},mfilename,'cfg.Ri');
validateattributes(zeta0,{'numeric'}, ...
    {'scalar','real','finite','>',0,'<',1},mfilename,'cfg.zeta0');
validateattributes(z,{'numeric'}, ...
    {'vector','nonempty','real','finite','nonnegative'},mfilename,'z');

z = reshape(z,1,[]);
wallTolerance = 100*eps(max(1,z(end)));
if abs(z(1)) > wallTolerance
    error('map_rim_to_global_station:ZGridMustStartAtWall', ...
        'The global z grid must start at z = 0.');
end
if numel(z) > 1 && any(diff(z) <= 0)
    error('map_rim_to_global_station:InvalidZGrid', ...
        'z must be strictly increasing.');
end

r0 = 1-zeta0;
s0 = -log(r0);

ri15 = Ri^(1/5);
etaScale = ri15*zeta0^(-2/5);
eta = etaScale*z;

% Initialize every point with the asymptotic rim state. This also supplies
% a controlled continuation when the global z grid extends beyond the
% finite interval used by bvp4c.
nz = numel(z);
FInfinity = rim.constants.FInfinity;

F      = FInfinity*ones(1,nz);
Fp     = zeros(1,nz);
Fpp    = zeros(1,nz);
Pi     = zeros(1,nz);
Theta  = zeros(1,nz);
Thetap = zeros(1,nz);
V      = ones(1,nz);
Vp     = zeros(1,nz);

inside = eta <= rim.etaMax;
if any(inside)
    y = deval(rim.sol,eta(inside));
    F(inside)      = y(1,:);
    Fp(inside)     = y(2,:);
    Fpp(inside)    = y(3,:);
    Pi(inside)     = y(4,:);
    Theta(inside)  = y(5,:);
    Thetap(inside) = y(6,:);
    V(inside)      = y(7,:);
    Vp(inside)     = y(8,:);
end

U = r0*Ri^(2/5)*zeta0^(1/5)*Fp;
A = r0*Ri^(3/5)*zeta0^(-1/5)*Fpp;
B = etaScale*Vp;
H = etaScale*Thetap;
pb = Ri^(4/5)*zeta0^(2/5)*Pi;

% W = psi_zeta at fixed physical z. Its far-field value is generally
% nonzero because F tends to a nonzero mass-flux constant.
W = r0*ri15*zeta0^(-2/5) .* (3/5*F-2/5*eta.*Fp);

station.zeta = zeta0;
station.r = r0;
station.s = s0;
station.z = z;
station.eta = eta;
station.Ri = Ri;
station.Pr = rim.Pr;

station.U = U;
station.A = A;
station.V = V;
station.B = B;
station.W = W;
station.Theta = Theta;
station.H = H;
station.pb = pb;

station.stateNames = {'U','A','V','B','W','Theta','H','pb'};
station.Y = [U; A; V; B; W; Theta; H; pb];

station.scales.eta = etaScale;
station.scales.U = r0*Ri^(2/5)*zeta0^(1/5);
station.scales.A = r0*Ri^(3/5)*zeta0^(-1/5);
station.scales.W = r0*ri15*zeta0^(-2/5);
station.scales.pb = Ri^(4/5)*zeta0^(2/5);

station.diagnostics.nFarFieldFilled = nnz(~inside);
station.diagnostics.etaAtZMax = eta(end);
station.diagnostics.rimEtaMax = rim.etaMax;
station.diagnostics.wallResidual = max(abs([ ...
    U(1), V(1), W(1), Theta(1)-1]));
station.diagnostics.farFieldResidual = max(abs([ ...
    U(end), V(end)-1, Theta(end), pb(end)]));
station.diagnostics.WInfinity = ...
    r0*ri15*zeta0^(-2/5)*(3/5)*FInfinity;

end

function validateRim(rim)

required = {'sol','Pr','etaMax','constants'};
missing = required(~isfield(rim,required));
if ~isempty(missing)
    error('map_rim_to_global_station:InvalidRimStruct', ...
        'rim is missing field(s): %s.',strjoin(missing,', '));
end

if ~isfield(rim.constants,'FInfinity')
    error('map_rim_to_global_station:InvalidRimStruct', ...
        'rim.constants.FInfinity is required.');
end

validateattributes(rim.etaMax,{'numeric'}, ...
    {'scalar','real','finite','positive'},mfilename,'rim.etaMax');
validateattributes(rim.constants.FInfinity,{'numeric'}, ...
    {'scalar','real','finite'},mfilename,'rim.constants.FInfinity');

end
