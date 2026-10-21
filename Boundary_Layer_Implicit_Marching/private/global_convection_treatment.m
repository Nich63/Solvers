function ct = global_convection_treatment(YPrevious,YCurrent,zetaCurrent,cfg)
% Select standard, FLARE or DUFLARE convection coefficients.

nCell = size(YPrevious,2)-1;
mode = 'standard';
if isfield(cfg,'convectionTreatment') && ~isempty(cfg.convectionTreatment)
    mode = char(cfg.convectionTreatment);
end

ct.mode = mode;
ct.coefU = ones(1,nCell);
ct.useDown = false(1,nCell);
ct.downMid = [];
ct.dZetaDown = NaN;
ct.nFlagged = 0;

switch mode
    case 'standard'
        return
    case 'flare'
        variant = 'zero';
        if isfield(cfg,'flareVariant') && ~isempty(cfg.flareVariant)
            variant = char(cfg.flareVariant);
        end
        thr = 0;
        if isfield(cfg,'flareThreshold') && ~isempty(cfg.flareThreshold)
            thr = cfg.flareThreshold;
        end
        UPrevMid = 0.5*(YPrevious(1,1:end-1)+YPrevious(1,2:end));
        flag = UPrevMid > thr;
        switch variant
            case 'zero', ct.coefU(flag) = 0;
            case 'abs',  ct.coefU(flag) = -1;
            otherwise
                error('global_convection_treatment:UnknownVariant', ...
                    'cfg.flareVariant must be ''zero'' or ''abs''.');
        end
        ct.nFlagged = nnz(flag);
    case 'duflare'
        if ~isfield(cfg,'duflareDownstream') || isempty(cfg.duflareDownstream)
            error('global_convection_treatment:MissingDownstream', ...
                ['convectionTreatment = ''duflare'' needs ', ...
                 'cfg.duflareDownstream (Y, zeta, maskU).']);
        end
        ds = cfg.duflareDownstream;
        if ~isequal(size(ds.Y),size(YCurrent)) || ~(ds.zeta > zetaCurrent) ...
                || numel(ds.maskU) ~= nCell
            error('global_convection_treatment:BadDownstream', ...
                ['cfg.duflareDownstream must carry Y of the station size, ', ...
                 'zeta > zetaCurrent and a 1-by-nCell maskU.']);
        end
        thr = 0;
        if isfield(cfg,'flareThreshold') && ~isempty(cfg.flareThreshold)
            thr = cfg.flareThreshold;
        end
        flag = reshape(ds.maskU,1,[]) > thr;
        ct.useDown = flag;
        ct.downMid = 0.5*(ds.Y(:,1:end-1)+ds.Y(:,2:end));
        ct.dZetaDown = ds.zeta-zetaCurrent;
        ct.nFlagged = nnz(flag);
    otherwise
        error('global_convection_treatment:UnknownTreatment', ...
            ['cfg.convectionTreatment must be ''standard'', ''flare'' ', ...
             'or ''duflare''; got ''%s''.'],mode);
end
end
