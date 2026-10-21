function [YCurrent,info] = march_one_station( ...
    YPrevious,z,zetaPrevious,zetaCurrent,cfg,history)
% Solve one implicit station with fsolve and a scaled analytic Jacobian.
% Accept only if boundary and raw OR normalized cell residuals pass.

if nargin < 6 || isempty(history)
    history = struct();
end

if ~isnumeric(z) || ~isvector(z) || numel(z) < 2
    error('march_one_station:InvalidZGrid', ...
        'z must be a numeric vector containing at least two nodes.');
end
z = reshape(z,1,[]);
nZ = numel(z);

if ~isnumeric(YPrevious) || ~isreal(YPrevious) ...
        || ~isequal(size(YPrevious),[8,nZ]) ...
        || any(~isfinite(YPrevious(:)))
    error('march_one_station:InvalidPreviousState', ...
        'YPrevious must be a finite 8-by-numel(z) array.');
end
if ~isstruct(cfg) || ~isscalar(cfg)
    error('march_one_station:InvalidConfig', ...
        'cfg must be a scalar structure.');
end
if ~isstruct(history) || ~isscalar(history)
    error('march_one_station:InvalidHistory', ...
        'history must be a scalar structure.');
end

cfg = setDefault(cfg,'functionTolerance',1e-20);
cfg = setDefault(cfg,'stepTolerance',1e-12);
cfg = setDefault(cfg,'optimalityTolerance',1e-12);
cfg = setDefault(cfg,'maxIterations',300);
cfg = setDefault(cfg,'maxFunctionEvaluations', ...
    max(300000,100*numel(YPrevious)));
cfg = setDefault(cfg,'acceptTolerance',1e-7);
cfg = setDefault(cfg,'boundaryTolerance',1e-8);
cfg = setDefault(cfg,'acceptToleranceNormalized',1e-8);
cfg = setDefault(cfg,'useResidualScaling',true);
cfg = setDefault(cfg,'useAnalyticJacobian',true);
cfg = setDefault(cfg,'typicalXFloor',1);
cfg = setDefault(cfg,'useExtrapolation',true);
cfg = setDefault(cfg,'retryWithPrevious',true);
cfg = setDefault(cfg,'useJacobianPattern',true);
cfg = setDefault(cfg,'solverAlgorithm','trust-region');
cfg = setDefault(cfg,'subproblemAlgorithm','factorization');
cfg = setDefault(cfg,'preconditionerBandwidth',Inf);
cfg = setDefault(cfg,'fsolveDisplay','off');

validateConfiguration(cfg);

% Evaluate once before fsolve so that coordinate and physics inputs are
% checked even if a future solver option changes the attempted guesses.
global_station_residual( ...
    YPrevious(:),YPrevious,z,zetaPrevious,zetaCurrent,cfg);

[guessList,guessNames,extrapolationRatio] = makeGuesses( ...
    YPrevious,zetaPrevious,zetaCurrent,cfg,history);

nAttempts = numel(guessList);
attemptTemplate = struct( ...
    'guessType','', ...
    'exitflag',NaN, ...
    'iterations',NaN, ...
    'functionCount',NaN, ...
    'residualInf',Inf, ...
    'boundaryResidualInf',Inf, ...
    'cellResidualInf',Inf, ...
    'normalizedCellResidualInf',NaN, ...
    'finiteSolution',false, ...
    'accepted',false, ...
    'acceptedViaNormalizedResidual',false, ...
    'scaling',struct('applied',false,'rowSpread',NaN,'columnSpread',NaN), ...
    'solverMessage','', ...
    'errorIdentifier','');
attempts = repmat(attemptTemplate,1,nAttempts);

bestScore = Inf;
bestY = YPrevious;
bestDetails = struct();
bestAttempt = NaN;
bestExitflag = NaN;
bestOutput = struct();

% The sparsity pattern is only needed by the paths that finite-difference
% the Jacobian; the analytic Jacobian is already sparse and exact.
needsPattern = ~cfg.useAnalyticJacobian ...
    && (cfg.useJacobianPattern || cfg.useResidualScaling);
if needsPattern
    jacobianPattern = makeJacobianPattern(nZ);
else
    jacobianPattern = [];
end

% Column grouping for the equilibration Jacobian is a property of the
% sparsity pattern alone, so it is computed once for every attempt.
if cfg.useResidualScaling && ~cfg.useAnalyticJacobian
    colorGroups = colorColumns(jacobianPattern);
else
    colorGroups = [];
end

analyticJacobian = @(x) global_station_jacobian( ...
    x,YPrevious,z,zetaPrevious,zetaCurrent,cfg);

for k = 1:nAttempts
    YGuess = enforceBoundaryValues(guessList{k});
    x0 = YGuess(:);

    unscaledResidual = @(x) global_station_residual( ...
        x,YPrevious,z,zetaPrevious,zetaCurrent,cfg);

    scalingApplied = false;
    rowScale = [];
    scalingInfo = struct('applied',false,'rowSpread',NaN,'columnSpread',NaN);
    if cfg.useResidualScaling
        if cfg.useAnalyticJacobian
            [rowScale,typicalXVector,scalingOk,rowSpread,columnSpread] = ...
                equilibrationFromJacobian(analyticJacobian(x0));
        else
            [rowScale,typicalXVector,scalingOk,rowSpread,columnSpread] = ...
                computeEquilibration(unscaledResidual,x0,jacobianPattern, ...
                colorGroups);
        end
        if scalingOk
            scalingApplied = true;
            scalingInfo.applied = true;
            scalingInfo.rowSpread = rowSpread;
            scalingInfo.columnSpread = columnSpread;
        end
    end

    % The row scale multiplies the residual and the Jacobian rows alike, so
    % the pair handed to fsolve stays consistent.
    if scalingApplied
        typicalX = typicalXVector;
        if cfg.useAnalyticJacobian
            rowScaleMatrix = spdiags(rowScale,0,numel(rowScale), ...
                numel(rowScale));
            residualFunction = @(x) scaledResidualAndJacobian( ...
                x,unscaledResidual,analyticJacobian,rowScale, ...
                rowScaleMatrix);
        else
            residualFunction = @(x) rowScale.*unscaledResidual(x);
        end
    else
        % Fallback: the pre-scaling behavior, kept so the old path stays
        % available and so a degenerate Jacobian cannot abort the march.
        typicalByState = max(max(abs(YGuess),[],2),cfg.typicalXFloor);
        typicalX = repmat(typicalByState,1,nZ);
        typicalX = typicalX(:);
        if cfg.useAnalyticJacobian
            residualFunction = @(x) residualAndJacobian( ...
                x,unscaledResidual,analyticJacobian);
        else
            residualFunction = unscaledResidual;
        end
    end

    options = optimoptions('fsolve', ...
        'Algorithm',cfg.solverAlgorithm, ...
        'Display',cfg.fsolveDisplay, ...
        'FunctionTolerance',cfg.functionTolerance, ...
        'StepTolerance',cfg.stepTolerance, ...
        'OptimalityTolerance',cfg.optimalityTolerance, ...
        'MaxIterations',cfg.maxIterations, ...
        'MaxFunctionEvaluations',cfg.maxFunctionEvaluations, ...
        'TypicalX',typicalX, ...
        'SpecifyObjectiveGradient',cfg.useAnalyticJacobian);

    if strcmpi(string(cfg.solverAlgorithm),'trust-region')
        options = optimoptions(options, ...
            'SubproblemAlgorithm',cfg.subproblemAlgorithm, ...
            'PrecondBandWidth',cfg.preconditionerBandwidth);
    end

    % JacobPattern is only consulted when fsolve has to estimate the
    % Jacobian itself.
    if cfg.useJacobianPattern && ~cfg.useAnalyticJacobian
        options = optimoptions(options,'JacobPattern',jacobianPattern);
    end

    solverError = [];
    try
        [xCandidate,~,exitflag,output] = fsolve( ...
            residualFunction,x0,options);
    catch ME
        xCandidate = x0;
        exitflag = -999;
        output = struct();
        solverError = ME;
    end

    YCandidate = reshape(xCandidate,8,nZ);
    [rawResidual,residualDetails] = global_station_residual( ...
        xCandidate,YPrevious,z,zetaPrevious,zetaCurrent,cfg);

    finiteSolution = isreal(YCandidate) ...
        && all(isfinite(YCandidate(:))) ...
        && isfinite(residualDetails.totalInfNorm);

    % Row-normalized cell residual at the candidate state (see header).
    normalizedCellResidualInf = NaN;
    if finiteSolution
        normalizedCellResidualInf = normalizedCellResidual( ...
            rawResidual,analyticJacobian(xCandidate));
    end

    % boundaryTolerance gates both criteria: the boundary rows have unit
    % row norm, so raw and normalized agree on them exactly.
    boundaryAccepted = finiteSolution ...
        && residualDetails.boundaryInfNorm <= cfg.boundaryTolerance;
    rawAccepted = boundaryAccepted ...
        && residualDetails.totalInfNorm <= cfg.acceptTolerance;
    normalizedAccepted = boundaryAccepted ...
        && isfinite(normalizedCellResidualInf) ...
        && normalizedCellResidualInf <= cfg.acceptToleranceNormalized;
    accepted = rawAccepted || normalizedAccepted;

    attempts(k).guessType = guessNames{k};
    attempts(k).exitflag = exitflag;
    attempts(k).iterations = getOutputValue(output,'iterations');
    attempts(k).functionCount = getOutputValue(output,'funcCount');
    attempts(k).residualInf = residualDetails.totalInfNorm;
    attempts(k).boundaryResidualInf = residualDetails.boundaryInfNorm;
    attempts(k).cellResidualInf = residualDetails.cellInfNorm;
    attempts(k).normalizedCellResidualInf = normalizedCellResidualInf;
    attempts(k).finiteSolution = finiteSolution;
    attempts(k).accepted = accepted;
    attempts(k).acceptedViaNormalizedResidual = ...
        accepted && normalizedAccepted && ~rawAccepted;
    attempts(k).scaling = scalingInfo;
    attempts(k).solverMessage = getOutputMessage(output);
    if ~isempty(solverError)
        attempts(k).errorIdentifier = solverError.identifier;
        attempts(k).solverMessage = solverError.message;
    end

    score = residualDetails.totalInfNorm;
    if ~isfinite(score)
        score = Inf;
    end
    if accepted || score < bestScore || isnan(bestAttempt)
        bestScore = score;
        bestY = YCandidate;
        bestDetails = residualDetails;
        bestAttempt = k;
        bestExitflag = exitflag;
        bestOutput = output;
    end

    if accepted
        break
    end
end

YCurrent = bestY;

info.accepted = ~isnan(bestAttempt) && attempts(bestAttempt).accepted;
info.bestAttempt = bestAttempt;
if isnan(bestAttempt)
    info.bestGuessType = '';
else
    info.bestGuessType = attempts(bestAttempt).guessType;
end
info.exitflag = bestExitflag;
info.output = bestOutput;
info.attempts = attempts(1:max(1,findAttemptCount(attempts)));
info.residual = bestDetails;
if ~isnan(bestAttempt)
    info.iterations = attempts(bestAttempt).iterations;
    info.functionCount = attempts(bestAttempt).functionCount;
    info.residualInf = attempts(bestAttempt).residualInf;
    info.boundaryResidualInf = ...
        attempts(bestAttempt).boundaryResidualInf;
    info.normalizedCellResidualInf = ...
        attempts(bestAttempt).normalizedCellResidualInf;
    info.acceptedViaNormalizedResidual = ...
        attempts(bestAttempt).acceptedViaNormalizedResidual;
else
    info.iterations = NaN;
    info.functionCount = NaN;
    info.residualInf = Inf;
    info.boundaryResidualInf = Inf;
    info.normalizedCellResidualInf = NaN;
    info.acceptedViaNormalizedResidual = false;
end
info.zetaPrevious = zetaPrevious;
info.zetaCurrent = zetaCurrent;
info.dZeta = zetaCurrent-zetaPrevious;
info.rCurrent = 1-zetaCurrent;
info.extrapolationRatio = extrapolationRatio;

if info.accepted && info.acceptedViaNormalizedResidual
    info.message = sprintf( ...
        ['Station accepted using the %s guess on the row-normalized ', ...
         'cell residual %.3e (raw residual %.3e exceeded acceptTolerance ', ...
         '%.1e, which is over-strict here -- see project_handoff.md 16.4).'], ...
        info.bestGuessType,info.normalizedCellResidualInf, ...
        info.residualInf,cfg.acceptTolerance);
elseif info.accepted
    info.message = sprintf( ...
        'Station accepted using the %s guess.',info.bestGuessType);
else
    % Quote both metrics: the raw norm alone is not interpretable across
    % r or Ri, and the normalized one is what shows whether this was a
    % near miss or a genuine non-convergence.
    info.message = sprintf( ...
        ['Station not accepted: raw residual infinity norm %.3e ', ...
         '(tolerance %.1e), row-normalized cell residual %.3e ', ...
         '(tolerance %.1e), boundary residual %.3e (tolerance %.1e). ', ...
         'The outer solver should reduce the marching step or stop.'], ...
        bestScore,cfg.acceptTolerance, ...
        info.normalizedCellResidualInf,cfg.acceptToleranceNormalized, ...
        info.boundaryResidualInf,cfg.boundaryTolerance);
end

end

function value = normalizedCellResidual(residual,J)

rowNorm = full(max(abs(J),[],2));
rowNorm(rowNorm == 0 | ~isfinite(rowNorm)) = 1;
value = norm(residual(9:end)./rowNorm(9:end),inf);

end

function [guessList,guessNames,ratio] = makeGuesses( ...
    YPrevious,zetaPrevious,zetaCurrent,cfg,history)

guessList = {};
guessNames = {};
ratio = NaN;

hasHistory = cfg.useExtrapolation ...
    && isfield(history,'YPreviousPrevious') ...
    && isfield(history,'zetaPreviousPrevious') ...
    && isequal(size(history.YPreviousPrevious),size(YPrevious)) ...
    && isreal(history.YPreviousPrevious) ...
    && all(isfinite(history.YPreviousPrevious(:))) ...
    && isscalar(history.zetaPreviousPrevious) ...
    && isreal(history.zetaPreviousPrevious) ...
    && isfinite(history.zetaPreviousPrevious) ...
    && history.zetaPreviousPrevious < zetaPrevious;

if hasHistory
    previousStep = zetaPrevious-history.zetaPreviousPrevious;
    currentStep = zetaCurrent-zetaPrevious;
    ratio = currentStep/previousStep;
    extrapolated = YPrevious ...
        + ratio*(YPrevious-history.YPreviousPrevious);
    guessList{end+1} = extrapolated;
    guessNames{end+1} = 'extrapolated';
end

if ~hasHistory || cfg.retryWithPrevious
    guessList{end+1} = YPrevious;
    guessNames{end+1} = 'previous station';
end

end

function Y = enforceBoundaryValues(Y)

Y(1,1) = 0;
Y(3,1) = 0;
Y(5,1) = 0;
Y(6,1) = 1;

Y(1,end) = 0;
Y(3,end) = 1;
Y(6,end) = 0;
Y(8,end) = 0;

end

function [F,J] = residualAndJacobian(x,residualFunction,jacobianFunction)
%RESIDUALANDJACOBIAN Objective pair for SpecifyObjectiveGradient.

F = residualFunction(x);
if nargout > 1
    J = jacobianFunction(x);
end

end

function [F,J] = scaledResidualAndJacobian( ...
    x,residualFunction,jacobianFunction,rowScale,rowScaleMatrix)

F = rowScale.*residualFunction(x);
if nargout > 1
    J = rowScaleMatrix*jacobianFunction(x);
end

end

function [rowScale,typicalX,ok,rowSpread,columnSpread] = ...
    equilibrationFromJacobian(J)

rowScale = [];
typicalX = [];
ok = false;
rowSpread = NaN;
columnSpread = NaN;

if ~all(isfinite(nonzeros(J)))
    return
end

m = size(J,1);
n = size(J,2);

rowMax = full(max(abs(J),[],2));
rowMax(rowMax == 0 | ~isfinite(rowMax)) = 1;
rowScale = 1./rowMax;

columnMax = full(max(abs(spdiags(rowScale,0,m,m)*J),[],1)).';
columnMax(columnMax == 0 | ~isfinite(columnMax)) = 1;
typicalX = 1./columnMax;

if ~all(isfinite(rowScale)) || ~all(isfinite(typicalX)) ...
        || any(rowScale <= 0) || any(typicalX <= 0)
    rowScale = [];
    typicalX = [];
    return
end

rowSpread = max(rowMax)/min(rowMax);
columnSpread = max(columnMax)/min(columnMax);
ok = true;

end

function [rowScale,typicalX,ok,rowSpread,columnSpread] = ...
    computeEquilibration(residualFunction,x0,pattern,colorGroups)

rowScale = [];
typicalX = [];
ok = false;
rowSpread = NaN;
columnSpread = NaN;

F0 = residualFunction(x0);
if any(~isfinite(F0))
    return
end

m = numel(F0);
n = numel(x0);

% The forward-difference step follows the same convention fsolve uses with
% a unit TypicalX. Only the magnitudes of the entries matter here, so this
% does not need the accuracy a Newton step would.
h = sqrt(eps)*max(1,abs(x0));

[rowIndex,columnIndex] = find(pattern);
values = zeros(numel(rowIndex),1);

for c = 1:max(colorGroups)
    columnsInGroup = (colorGroups == c).';
    xPerturbed = x0;
    xPerturbed(columnsInGroup) = ...
        xPerturbed(columnsInGroup)+h(columnsInGroup);
    FPerturbed = residualFunction(xPerturbed);
    if any(~isfinite(FPerturbed))
        return
    end
    selected = columnsInGroup(columnIndex);
    values(selected) = ...
        (FPerturbed(rowIndex(selected))-F0(rowIndex(selected))) ...
        ./h(columnIndex(selected));
end

J = sparse(rowIndex,columnIndex,values,m,n);
if ~all(isfinite(values))
    return
end

rowMax = full(max(abs(J),[],2));
rowMax(rowMax == 0 | ~isfinite(rowMax)) = 1;
rowScale = 1./rowMax;

columnMax = full(max(abs(spdiags(rowScale,0,m,m)*J),[],1)).';
columnMax(columnMax == 0 | ~isfinite(columnMax)) = 1;
typicalX = 1./columnMax;

if ~all(isfinite(rowScale)) || ~all(isfinite(typicalX)) ...
        || any(rowScale <= 0) || any(typicalX <= 0)
    rowScale = [];
    typicalX = [];
    return
end

rowSpread = max(rowMax)/min(rowMax);
columnSpread = max(columnMax)/min(columnMax);
ok = true;

end

function groups = colorColumns(pattern)

[m,n] = size(pattern);
groups = zeros(1,n);
usedRows = false(m,16);
numGroups = 0;

for j = 1:n
    rows = find(pattern(:,j));
    placed = false;
    for c = 1:numGroups
        if ~any(usedRows(rows,c))
            usedRows(rows,c) = true;
            groups(j) = c;
            placed = true;
            break
        end
    end
    if ~placed
        numGroups = numGroups+1;
        if numGroups > size(usedRows,2)
            usedRows(:,2*size(usedRows,2)) = false;
        end
        usedRows(rows,numGroups) = true;
        groups(j) = numGroups;
    end
end

end

function pattern = makeJacobianPattern(nZ)

nUnknown = 8*nZ;
pattern = spalloc(nUnknown,nUnknown,16*8*(nZ-1)+8);

% Boundary rows.
pattern(1,1) = 1;
pattern(2,3) = 1;
pattern(3,5) = 1;
pattern(4,6) = 1;
last = 8*(nZ-1);
pattern(5,last+1) = 1;
pattern(6,last+3) = 1;
pattern(7,last+6) = 1;
pattern(8,last+8) = 1;

% A conservative 8-by-16 block for every vertical cell. Each cell
% residual depends only on the two current-state nodes that bound it.
for j = 1:nZ-1
    rows = 8+(j-1)*8+(1:8);
    columns = (j-1)*8+(1:16);
    pattern(rows,columns) = 1;
end

end

function value = getOutputValue(output,name)

if isstruct(output) && isfield(output,name) ...
        && isnumeric(output.(name)) && isscalar(output.(name))
    value = output.(name);
else
    value = NaN;
end

end

function message = getOutputMessage(output)

if isstruct(output) && isfield(output,'message')
    message = char(string(output.message));
else
    message = '';
end

end

function count = findAttemptCount(attempts)

count = find(~cellfun(@isempty,{attempts.guessType}),1,'last');
if isempty(count)
    count = 1;
end

end

function validateConfiguration(cfg)

positiveFields = { ...
    'functionTolerance', ...
    'stepTolerance', ...
    'optimalityTolerance', ...
    'acceptTolerance', ...
    'boundaryTolerance', ...
    'acceptToleranceNormalized', ...
    'typicalXFloor'};

for k = 1:numel(positiveFields)
    name = positiveFields{k};
    value = cfg.(name);
    if ~isnumeric(value) || ~isscalar(value) ...
            || ~isfinite(value) || value <= 0
        error('march_one_station:InvalidConfiguration', ...
            'cfg.%s must be a finite positive scalar.',name);
    end
end

integerFields = {'maxIterations','maxFunctionEvaluations'};
for k = 1:numel(integerFields)
    name = integerFields{k};
    value = cfg.(name);
    if ~isnumeric(value) || ~isscalar(value) ...
            || ~isfinite(value) || value < 1 || value ~= round(value)
        error('march_one_station:InvalidConfiguration', ...
            'cfg.%s must be a positive integer.',name);
    end
end

logicalFields = { ...
    'useExtrapolation','retryWithPrevious','useJacobianPattern', ...
    'useResidualScaling','useAnalyticJacobian'};
for k = 1:numel(logicalFields)
    name = logicalFields{k};
    if ~(islogical(cfg.(name)) && isscalar(cfg.(name)))
        error('march_one_station:InvalidConfiguration', ...
            'cfg.%s must be a logical scalar.',name);
    end
end

validAlgorithms = {'trust-region','trust-region-dogleg', ...
    'levenberg-marquardt'};
if ~(ischar(cfg.solverAlgorithm) || isstring(cfg.solverAlgorithm)) ...
        || ~isscalar(string(cfg.solverAlgorithm)) ...
        || ~any(strcmpi(string(cfg.solverAlgorithm),validAlgorithms))
    error('march_one_station:InvalidConfiguration', ...
        'cfg.solverAlgorithm is not a valid fsolve algorithm.');
end

validSubproblemAlgorithms = {'cg','factorization'};
if ~(ischar(cfg.subproblemAlgorithm) || isstring(cfg.subproblemAlgorithm)) ...
        || ~isscalar(string(cfg.subproblemAlgorithm)) ...
        || ~any(strcmpi(string(cfg.subproblemAlgorithm), ...
        validSubproblemAlgorithms))
    error('march_one_station:InvalidConfiguration', ...
        'cfg.subproblemAlgorithm must be ''cg'' or ''factorization''.');
end

if ~isnumeric(cfg.preconditionerBandwidth) ...
        || ~isscalar(cfg.preconditionerBandwidth) ...
        || ~isreal(cfg.preconditionerBandwidth) ...
        || cfg.preconditionerBandwidth < 0 ...
        || (~isinf(cfg.preconditionerBandwidth) ...
        && cfg.preconditionerBandwidth ~= round(cfg.preconditionerBandwidth))
    error('march_one_station:InvalidConfiguration', ...
        ['cfg.preconditionerBandwidth must be a nonnegative integer ', ...
         'or Inf.']);
end

validDisplay = {'off','none','iter','iter-detailed','final','final-detailed', ...
    'notify','notify-detailed'};
if ~(ischar(cfg.fsolveDisplay) || isstring(cfg.fsolveDisplay)) ...
        || ~isscalar(string(cfg.fsolveDisplay)) ...
        || ~any(strcmpi(string(cfg.fsolveDisplay),validDisplay))
    error('march_one_station:InvalidConfiguration', ...
        'cfg.fsolveDisplay is not a valid fsolve display setting.');
end

end

function s = setDefault(s,name,value)

if ~isfield(s,name) || isempty(s.(name))
    s.(name) = value;
end

end
