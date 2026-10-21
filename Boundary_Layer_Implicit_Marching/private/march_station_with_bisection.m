function [YCurrent,info] = march_station_with_bisection( ...
    YPrevious,z,zetaPrevious,zetaCurrent,cfg,history)
% Retry rejected steps by recursive bisection, up to cfg.maxStepHalvings.
% Return the requested endpoint; keep intermediate diagnostics in info.

if nargin < 6 || isempty(history)
    history = struct();
end

cfg = setDefault(cfg,'maxStepHalvings',6);

[YCurrent,info,subLog] = recurse( ...
    YPrevious,z,zetaPrevious,zetaCurrent,cfg,history,cfg.maxStepHalvings);

info.stepHalving.numHalvings = numel(subLog);
info.stepHalving.subStations = subLog;

end

function [YCurrent,info,subLog] = recurse( ...
    YPrevious,z,zetaPrevious,zetaCurrent,cfg,history,depthLeft)

[YCurrent,info] = march_one_station( ...
    YPrevious,z,zetaPrevious,zetaCurrent,cfg,history);

subLog = emptyLog();

if info.accepted || depthLeft <= 0
    return
end

zetaMid = zetaPrevious + 0.5*(zetaCurrent-zetaPrevious);

[YMid,infoMid,subLogFirstHalf] = recurse( ...
    YPrevious,z,zetaPrevious,zetaMid,cfg,history,depthLeft-1);

subLog = [subLogFirstHalf,logEntry(zetaMid,infoMid)];

if ~infoMid.accepted
    % Cannot even bridge the first half within the remaining subdivision
    % budget: report the original (unhalved) target's own failure, tagged
    % with the bisection trail that was attempted.
    info.stepHalvingBridgeFailed = true;
    return
end

historySecondHalf.YPreviousPrevious = YPrevious;
historySecondHalf.zetaPreviousPrevious = zetaPrevious;

[YCurrent,info,subLogSecondHalf] = recurse( ...
    YMid,z,zetaMid,zetaCurrent,cfg,historySecondHalf,depthLeft-1);

subLog = [subLog,subLogSecondHalf];

end

function entry = logEntry(zeta,info)

entry.zeta = zeta;
entry.accepted = info.accepted;
entry.residualInf = info.residualInf;
entry.boundaryResidualInf = info.boundaryResidualInf;
entry.guessType = info.bestGuessType;

end

function log = emptyLog()

log = struct('zeta',{},'accepted',{},'residualInf',{},'boundaryResidualInf',{},'guessType',{});

end

function s = setDefault(s,name,value)

if ~isfield(s,name) || isempty(s.(name))
    s.(name) = value;
end

end
