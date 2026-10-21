function grid = build_global_grid(cfg)
% Build the s=-log(r) radial grid and stretched vertical grid.
% Required cfg: zeta0, rMin, zMax; optional nZeta, nZ, zStretch.

arguments
    cfg (1,1) struct
end

cfg = setDefault(cfg,'nZeta',401);
cfg = setDefault(cfg,'nZ',401);
cfg = setDefault(cfg,'zStretch',1.5);

zeta0 = requireField(cfg,'zeta0');
rMin = requireField(cfg,'rMin');
zMax = requireField(cfg,'zMax');

validateattributes(zeta0,{'numeric'}, ...
    {'scalar','real','finite','>',0,'<',1},mfilename,'cfg.zeta0');
validateattributes(rMin,{'numeric'}, ...
    {'scalar','real','finite','>',eps,'<',1},mfilename,'cfg.rMin');
validateattributes(zMax,{'numeric'}, ...
    {'scalar','real','finite','positive'},mfilename,'cfg.zMax');
validateattributes(cfg.nZeta,{'numeric'}, ...
    {'scalar','real','finite','integer','>=',2},mfilename,'cfg.nZeta');
validateattributes(cfg.nZ,{'numeric'}, ...
    {'scalar','real','finite','integer','>=',2},mfilename,'cfg.nZ');
validateattributes(cfg.zStretch,{'numeric'}, ...
    {'scalar','real','finite','>=',1},mfilename,'cfg.zStretch');

r0 = 1-zeta0;
if rMin >= r0
    error('build_global_grid:InvalidRadialInterval', ...
        'cfg.rMin must be smaller than 1-cfg.zeta0.');
end

% Uniform auxiliary s grid. The governing equations still use zeta, so
% radial derivatives must be divided by the local grid.dZeta value.
s0 = -log(r0);
sMax = -log(rMin);
s = linspace(s0,sMax,cfg.nZeta);
r = exp(-s);
zeta = 1-r;

% Make the user-specified endpoints exact rather than leaving roundoff from
% the exponential/logarithm round trip.
s(1) = s0;
s(end) = sMax;
r(1) = r0;
r(end) = rMin;
zeta(1) = zeta0;
zeta(end) = 1-rMin;

dS = diff(s);
dR = diff(r);
dZeta = diff(zeta);

if any(dZeta <= 0)
    error('build_global_grid:UnresolvedZetaSpacing', ...
        ['The s grid produces repeated or non-increasing zeta nodes in ', ...
         'double precision. Increase cfg.rMin or reduce cfg.nZeta.']);
end

% Power-law vertical grid. zStretch = 1 gives a uniform grid; values above
% one cluster nodes near the wall while retaining the prescribed zMax.
xi = linspace(0,1,cfg.nZ);
z = zMax*xi.^cfg.zStretch;
z(1) = 0;
z(end) = zMax;
dZ = diff(z);

if any(dZ <= 0)
    error('build_global_grid:UnresolvedZSpacing', ...
        ['The vertical stretching produces repeated z nodes in double ', ...
         'precision. Reduce cfg.zStretch or cfg.nZ.']);
end

grid.s = s;
grid.r = r;
grid.zeta = zeta;
grid.dS = dS;
grid.dR = dR;
grid.dZeta = dZeta;

grid.zetaMid = 0.5*(zeta(1:end-1)+zeta(2:end));
grid.rMid = 1-grid.zetaMid;
grid.sMid = -log(grid.rMid);

grid.z = z;
grid.dZ = dZ;
grid.zMid = 0.5*(z(1:end-1)+z(2:end));

grid.nZeta = cfg.nZeta;
grid.nStations = cfg.nZeta;
grid.nZ = cfg.nZ;
grid.zeta0 = zeta0;
grid.rMin = rMin;
grid.zMax = zMax;
grid.zStretch = cfg.zStretch;
grid.config = cfg;

% Diagnostics that are useful when choosing grid parameters.
grid.diagnostics.minDZeta = min(dZeta);
grid.diagnostics.maxDZeta = max(dZeta);
grid.diagnostics.minDZ = min(dZ);
grid.diagnostics.maxDZ = max(dZ);
grid.diagnostics.radialSpacingRatio = max(dZeta)/min(dZeta);
grid.diagnostics.verticalSpacingRatio = max(dZ)/min(dZ);
grid.diagnostics.relativeRadialStep = abs(dR)./r(1:end-1);
grid.diagnostics.relativeRadialStepVariation = ...
    max(grid.diagnostics.relativeRadialStep) ...
    - min(grid.diagnostics.relativeRadialStep);

end

function value = requireField(s,name)

if ~isfield(s,name) || isempty(s.(name))
    error('build_global_grid:MissingConfiguration', ...
        'cfg.%s is required.',name);
end
value = s.(name);

end

function s = setDefault(s,name,value)

if ~isfield(s,name) || isempty(s.(name))
    s.(name) = value;
end

end
