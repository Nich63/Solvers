function results = solve_global_boundary_layer(cfg)
% March the boundary layer from a rim BVP or restartStation.
% Required cfg: Ri, zeta0, rMin, zMax. Stop on rejected stations.

arguments
    cfg (1,1) struct
end

cfg = setDefault(cfg,'Pr',0.72);
cfg = setDefault(cfg,'showProgress',false);
cfg = setDefault(cfg,'progressEvery',10);
cfg = setDefault(cfg,'storeStepInfo',false);
cfg = setDefault(cfg,'warnOnFailure',true);
cfg = setDefault(cfg,'convectionTreatment','standard');

% Accurate nonlinear-solver defaults. These override the loose stopping
% behavior that can make fsolve report success while individual discrete
% equations still have O(1e-6) residuals. User-supplied values are kept.
cfg = setDefault(cfg,'functionTolerance',1e-20);
cfg = setDefault(cfg,'stepTolerance',1e-12);
cfg = setDefault(cfg,'optimalityTolerance',1e-12);
cfg = setDefault(cfg,'maxIterations',300);
cfg = setDefault(cfg,'acceptTolerance',1e-7);
cfg = setDefault(cfg,'boundaryTolerance',1e-8);
cfg = setDefault(cfg,'acceptToleranceNormalized',1e-8);
cfg = setDefault(cfg,'solverAlgorithm','trust-region');
cfg = setDefault(cfg,'subproblemAlgorithm','factorization');
cfg = setDefault(cfg,'preconditionerBandwidth',Inf);

validateControllerConfiguration(cfg);

% Build all requested target stations first. The governing equations use
% zeta even though the radial nodes are generated uniformly in s=-log(r).
requestedGrid = build_global_grid(cfg);

cfg = setDefault(cfg,'maxFunctionEvaluations', ...
    max(300000,100*8*requestedGrid.nZ));

if isfield(cfg,'restartStation') && ~isempty(cfg.restartStation)
    rim = [];
    initialStation = restartStationOnGrid(cfg.restartStation, ...
        requestedGrid.z,requestedGrid.zeta(1),cfg.Ri);
else
    rim = solve_rim_bvp(cfg);
    initialStation = map_rim_to_global_station( ...
        rim,requestedGrid.z,cfg);
end

nZ = requestedGrid.nZ;
nRequested = requestedGrid.nStations;

if ~isequal(size(initialStation.Y),[8,nZ])
    error('solve_global_boundary_layer:InvalidInitialStation', ...
        'The mapped rim station must have size 8-by-grid.nZ.');
end

Y = NaN(8,nZ,nRequested);
Y(:,:,1) = initialStation.Y;

diagnostics.accepted = false(1,nRequested);
diagnostics.exitflag = NaN(1,nRequested);
diagnostics.iterations = NaN(1,nRequested);
diagnostics.functionCount = NaN(1,nRequested);
diagnostics.residualInf = NaN(1,nRequested);
diagnostics.boundaryResidualInf = NaN(1,nRequested);
diagnostics.cellResidualInf = NaN(1,nRequested);
diagnostics.normalizedCellResidualInf = NaN(1,nRequested);
diagnostics.acceptedViaNormalizedResidual = false(1,nRequested);
diagnostics.wallResidualInf = NaN(1,nRequested);
diagnostics.farFieldResidualInf = NaN(1,nRequested);
diagnostics.identityError = NaN(1,nRequested);
diagnostics.nFlaggedCells = zeros(1,nRequested);
diagnostics.WFar = NaN(1,nRequested);
diagnostics.numHalvings = NaN(1,nRequested);
diagnostics.guessType = repmat({''},1,nRequested);
diagnostics.message = repmat({''},1,nRequested);

diagnostics.accepted(1) = true;
diagnostics.numHalvings(1) = 0;
diagnostics.wallResidualInf(1) = stateWallResidual(initialStation.Y);
diagnostics.farFieldResidualInf(1) = ...
    stateFarFieldResidual(initialStation.Y);
diagnostics.identityError(1) = stateIdentityError(initialStation.Y,cfg.Pr);
diagnostics.WFar(1) = initialStation.Y(5,end);
diagnostics.guessType{1} = 'rim mapping';
diagnostics.message{1} = 'Initial station supplied by the rim mapping.';

if cfg.storeStepInfo
    stepInfo = cell(1,nRequested);
else
    stepInfo = {};
end

nAccepted = 1;
failureOccurred = false;
failureIndex = NaN;
failureCandidate = [];
failureInfo = struct();

if cfg.showProgress
    fprintf(['Global march: station 1/%d accepted at ', ...
        'zeta=%.6g, r=%.6g (rim mapping).\n'], ...
        nRequested,requestedGrid.zeta(1),requestedGrid.r(1));
end

for k = 2:nRequested
    if k >= 3
        history.YPreviousPrevious = Y(:,:,k-2);
        history.zetaPreviousPrevious = requestedGrid.zeta(k-2);
    else
        history = struct();
    end

    stationCfg = stationConvectionConfig(cfg,k,nRequested,requestedGrid.zeta);
    [YCandidate,currentInfo] = march_station_with_bisection( ...
        Y(:,:,k-1), ...
        requestedGrid.z, ...
        requestedGrid.zeta(k-1), ...
        requestedGrid.zeta(k), ...
        stationCfg, ...
        history);

    if cfg.storeStepInfo
        stepInfo{k} = currentInfo;
    end

    if ~currentInfo.accepted
        failureOccurred = true;
        failureIndex = k;
        failureCandidate = YCandidate;
        failureInfo = currentInfo;
        break
    end

    Y(:,:,k) = YCandidate;
    nAccepted = k;

    diagnostics.accepted(k) = true;
    diagnostics.exitflag(k) = currentInfo.exitflag;
    diagnostics.iterations(k) = currentInfo.iterations;
    diagnostics.functionCount(k) = currentInfo.functionCount;
    diagnostics.residualInf(k) = currentInfo.residualInf;
    diagnostics.boundaryResidualInf(k) = ...
        currentInfo.boundaryResidualInf;
    diagnostics.cellResidualInf(k) = ...
        currentInfo.residual.cellInfNorm;
    diagnostics.normalizedCellResidualInf(k) = ...
        currentInfo.normalizedCellResidualInf;
    diagnostics.acceptedViaNormalizedResidual(k) = ...
        currentInfo.acceptedViaNormalizedResidual;
    diagnostics.wallResidualInf(k) = stateWallResidual(YCandidate);
    diagnostics.farFieldResidualInf(k) = ...
        stateFarFieldResidual(YCandidate);
    diagnostics.identityError(k) = stateIdentityError(YCandidate,cfg.Pr);
    diagnostics.WFar(k) = YCandidate(5,end);
    ctAccepted = global_convection_treatment(Y(:,:,k-1),YCandidate, ...
        requestedGrid.zeta(k),stationCfg);
    diagnostics.nFlaggedCells(k) = ctAccepted.nFlagged;
    diagnostics.numHalvings(k) = currentInfo.stepHalving.numHalvings;
    diagnostics.guessType{k} = currentInfo.bestGuessType;
    diagnostics.message{k} = currentInfo.message;

    shouldPrint = cfg.showProgress && ( ...
        k == 2 || k == nRequested || mod(k-1,cfg.progressEvery) == 0);
    if shouldPrint
        % Both norms are printed: the raw one for continuity with older
        % logs, the row-normalized one because it is the one comparable
        % across r and Ri (project_handoff.md 16.4).
        fprintf(['Global march: station %d/%d accepted at ', ...
            'zeta=%.6g, r=%.6g, ||R||_inf=%.3e, ', ...
            '||R||_inf(row-normalized cells)=%.3e.\n'], ...
            k,nRequested,requestedGrid.zeta(k),requestedGrid.r(k), ...
            currentInfo.residualInf,currentInfo.normalizedCellResidualInf);
    end
end

% Discard unused NaN storage so every returned radial array has the same
% number of accepted stations.
Y = Y(:,:,1:nAccepted);
acceptedGrid = truncateGrid(requestedGrid,nAccepted);
diagnostics = truncateDiagnostics(diagnostics,nAccepted);

results.config = cfg;
results.completed = ~failureOccurred && nAccepted == nRequested;
results.nRequested = nRequested;
results.nAccepted = nAccepted;
results.lastAcceptedIndex = nAccepted;
results.stateNames = {'U','A','V','B','W','Theta','H','pb'};

results.grid = acceptedGrid;
results.requestedGrid = requestedGrid;
results.z = acceptedGrid.z;
results.s = acceptedGrid.s;
results.r = acceptedGrid.r;
results.zeta = acceptedGrid.zeta;
results.dZeta = acceptedGrid.dZeta;

results.Y = Y;
results.U = reshape(Y(1,:,:),nZ,nAccepted);
results.A = reshape(Y(2,:,:),nZ,nAccepted);
results.V = reshape(Y(3,:,:),nZ,nAccepted);
results.B = reshape(Y(4,:,:),nZ,nAccepted);
results.W = reshape(Y(5,:,:),nZ,nAccepted);
results.Theta = reshape(Y(6,:,:),nZ,nAccepted);
results.H = reshape(Y(7,:,:),nZ,nAccepted);
results.pb = reshape(Y(8,:,:),nZ,nAccepted);

results.rim = rim;
results.initialStation = initialStation;
results.diagnostics = diagnostics;

if cfg.storeStepInfo
    results.stepInfo = stepInfo(1:nAccepted);
end

results.failure.occurred = failureOccurred;
results.failure.index = failureIndex;
results.failure.candidateY = failureCandidate;
results.failure.info = failureInfo;
if failureOccurred
    results.failure.zeta = requestedGrid.zeta(failureIndex);
    results.failure.r = requestedGrid.r(failureIndex);
    results.failure.message = failureInfo.message;
else
    results.failure.zeta = NaN;
    results.failure.r = NaN;
    results.failure.message = '';
end

if failureOccurred && cfg.warnOnFailure
    warning('solve_global_boundary_layer:MarchStopped', ...
        ['Global march stopped before station %d/%d at zeta=%.6g ', ...
         '(r=%.6g). %s'], ...
        failureIndex,nRequested,results.failure.zeta, ...
        results.failure.r,results.failure.message);
end

end

function value = stateWallResidual(Y)

value = max(abs([Y(1,1),Y(3,1),Y(5,1),Y(6,1)-1]));

end

function value = stateFarFieldResidual(Y)

value = max(abs([Y(1,end),Y(3,end)-1,Y(6,end),Y(8,end)]));

end

function value = stateIdentityError(Y,Pr)

if abs(Pr-1) < 100*eps
    value = max(abs(Y(3,:)+Y(6,:)-1));
else
    value = NaN;
end

end

function grid = truncateGrid(requestedGrid,nAccepted)

grid = requestedGrid;

nodeFields = {'s','r','zeta'};
for k = 1:numel(nodeFields)
    name = nodeFields{k};
    values = requestedGrid.(name);
    grid.(name) = values(1:nAccepted);
end

cellFields = {'dS','dR','dZeta','sMid','rMid','zetaMid'};
nCells = max(0,nAccepted-1);
for k = 1:numel(cellFields)
    name = cellFields{k};
    values = requestedGrid.(name);
    grid.(name) = values(1:nCells);
end

grid.nZeta = nAccepted;
grid.nStations = nAccepted;
grid.rMin = grid.r(end);

if nCells > 0
    grid.diagnostics.minDZeta = min(grid.dZeta);
    grid.diagnostics.maxDZeta = max(grid.dZeta);
    grid.diagnostics.radialSpacingRatio = ...
        max(grid.dZeta)/min(grid.dZeta);
    grid.diagnostics.relativeRadialStep = ...
        abs(grid.dR)./grid.r(1:end-1);
    grid.diagnostics.relativeRadialStepVariation = ...
        max(grid.diagnostics.relativeRadialStep) ...
        - min(grid.diagnostics.relativeRadialStep);
else
    grid.diagnostics.minDZeta = NaN;
    grid.diagnostics.maxDZeta = NaN;
    grid.diagnostics.radialSpacingRatio = NaN;
    grid.diagnostics.relativeRadialStep = [];
    grid.diagnostics.relativeRadialStepVariation = NaN;
end

end

function diagnostics = truncateDiagnostics(diagnostics,nAccepted)

names = fieldnames(diagnostics);
for k = 1:numel(names)
    name = names{k};
    values = diagnostics.(name);
    diagnostics.(name) = values(1:nAccepted);
end

end

function station = restartStationOnGrid(rs,z,zeta0,Ri)

if abs(rs.zeta-zeta0) > 1e-14
    error('solve_global_boundary_layer:RestartZeta', ...
        'cfg.zeta0 = %.16g must equal restartStation.zeta = %.16g.',zeta0,rs.zeta);
end
zs = rs.z(:);
z = z(:);
if z(end) > zs(end)*(1+1e-14)
    error('solve_global_boundary_layer:RestartHeight', ...
        'restartStation reaches z = %g, below the new zMax = %g.',zs(end),z(end));
end
Y = zeros(8,numel(z));
for q = 1:8
    Y(q,:) = interp1(zs,rs.Y(q,:).',z,'pchip').';
end
Y(1,1) = 0; Y(3,1) = 0; Y(5,1) = 0; Y(6,1) = 1;
Y(1,end) = 0; Y(3,end) = 1; Y(6,end) = 0;
Th = Y(6,:);
dZ = diff(z).';
pb = zeros(1,numel(z));
for j = numel(z)-1:-1:1
    pb(j) = pb(j+1)-Ri*0.5*(Th(j)+Th(j+1))*dZ(j);
end
Y(8,:) = pb;
station.Y = Y;
station.zeta = zeta0;
station.source = 'restartStation';
end

function stationCfg = stationConvectionConfig(cfg,k,nRequested,zetaGrid)

stationCfg = cfg;
if ~strcmp(cfg.convectionTreatment,'duflare')
    return
end
prev = cfg.duflarePreviousSweep;
if size(prev.Y,3) ~= nRequested || numel(prev.zeta) ~= nRequested ...
        || abs(prev.zeta(k)-zetaGrid(k)) > 1e-14
    error('solve_global_boundary_layer:DuflareGridMismatch', ...
        'cfg.duflarePreviousSweep must be a complete sweep on the requested grid.');
end
if k == nRequested
    stationCfg.convectionTreatment = 'flare';
    return
end
Yk = prev.Y(:,:,k);
stationCfg.duflareDownstream = struct( ...
    'Y',prev.Y(:,:,k+1), ...
    'zeta',prev.zeta(k+1), ...
    'maskU',0.5*(Yk(1,1:end-1)+Yk(1,2:end)));
end

function validateControllerConfiguration(cfg)

if ~isfield(cfg,'Ri') || isempty(cfg.Ri)
    error('solve_global_boundary_layer:MissingRi','cfg.Ri is required.');
end
if ~isnumeric(cfg.Ri) || ~isscalar(cfg.Ri) ...
        || ~isreal(cfg.Ri) || ~isfinite(cfg.Ri) || cfg.Ri <= 0
    error('solve_global_boundary_layer:InvalidRi', ...
        'cfg.Ri must be a finite positive scalar for rim mapping.');
end
if ~isnumeric(cfg.Pr) || ~isscalar(cfg.Pr) ...
        || ~isreal(cfg.Pr) || ~isfinite(cfg.Pr) || cfg.Pr <= 0
    error('solve_global_boundary_layer:InvalidPr', ...
        'cfg.Pr must be a finite positive scalar.');
end

logicalFields = {'showProgress','storeStepInfo','warnOnFailure'};
for k = 1:numel(logicalFields)
    name = logicalFields{k};
    if ~(islogical(cfg.(name)) && isscalar(cfg.(name)))
        error('solve_global_boundary_layer:InvalidConfiguration', ...
            'cfg.%s must be a logical scalar.',name);
    end
end

if ~any(strcmp(cfg.convectionTreatment,{'standard','flare','duflare'}))
    error('solve_global_boundary_layer:InvalidConfiguration', ...
        'cfg.convectionTreatment must be ''standard'', ''flare'' or ''duflare''.');
end
if strcmp(cfg.convectionTreatment,'duflare')
    if ~isfield(cfg,'duflarePreviousSweep') || ~isstruct(cfg.duflarePreviousSweep)
        error('solve_global_boundary_layer:InvalidConfiguration', ...
            '''duflare'' needs cfg.duflarePreviousSweep with .Y and .zeta.');
    end
    if ~isfield(cfg,'maxStepHalvings') || cfg.maxStepHalvings ~= 0
        error('solve_global_boundary_layer:InvalidConfiguration', ...
            '''duflare'' sweeps must share one grid: set cfg.maxStepHalvings = 0.');
    end
end

if ~isnumeric(cfg.progressEvery) || ~isscalar(cfg.progressEvery) ...
        || ~isreal(cfg.progressEvery) || ~isfinite(cfg.progressEvery) ...
        || cfg.progressEvery < 1 || cfg.progressEvery ~= round(cfg.progressEvery)
    error('solve_global_boundary_layer:InvalidConfiguration', ...
        'cfg.progressEvery must be a positive integer.');
end

end

function s = setDefault(s,name,value)

if ~isfield(s,name) || isempty(s.(name))
    s.(name) = value;
end

end
