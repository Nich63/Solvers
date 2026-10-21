function rim = solve_rim_bvp(cfg)

arguments
    cfg (1,1) struct = struct()
end

cfg = setDefault(cfg,'Pr',0.72);
cfg = setDefault(cfg,'etaMax',20);
cfg = setDefault(cfg,'nInitial',201);
cfg = setDefault(cfg,'nOutput',1001);
cfg = setDefault(cfg,'relTol',1e-8);
cfg = setDefault(cfg,'absTol',1e-10);
cfg = setDefault(cfg,'nMax',50000);
cfg = setDefault(cfg,'showStats',false);

Pr = cfg.Pr;

solinit = bvpinit( ...
    linspace(0,cfg.etaMax,cfg.nInitial), ...
    @initialGuess);

statsSetting = 'off';
if cfg.showStats
    statsSetting = 'on';
end

options = bvpset( ...
    'RelTol',cfg.relTol, ...
    'AbsTol',cfg.absTol, ...
    'NMax',cfg.nMax, ...
    'Stats',statsSetting);

sol = bvp4c( ...
    @(eta,y) rimODE(eta,y,Pr), ...
    @rimBC, ...
    solinit, ...
    options);

eta = linspace(0,cfg.etaMax,cfg.nOutput);
y = deval(sol,eta);

rim.sol      = sol;
rim.Pr       = Pr;
rim.etaMax   = cfg.etaMax;
rim.eta      = eta;

rim.F        = y(1,:);
rim.Fp       = y(2,:);
rim.Fpp      = y(3,:);
rim.Pi       = y(4,:);
rim.Theta    = y(5,:);
rim.Thetap   = y(6,:);
rim.V        = y(7,:);
rim.Vp       = y(8,:);

rim.constants.FppWall   = rim.Fpp(1);
rim.constants.PiWall    = rim.Pi(1);
rim.constants.heatFlux  = -rim.Thetap(1);
rim.constants.VpWall    = rim.Vp(1);
rim.constants.FInfinity = rim.F(end);

rim.diagnostics.wallResidual = max(abs([ ...
    rim.F(1), ...
    rim.Fp(1), ...
    rim.Theta(1)-1, ...
    rim.V(1)]));

rim.diagnostics.farFieldResidual = max(abs([ ...
    rim.Fp(end), ...
    rim.Pi(end), ...
    rim.Theta(end), ...
    rim.V(end)-1]));

etaProbe = 0.75*cfg.etaMax;
yProbe = deval(sol,etaProbe);
rim.diagnostics.farFieldDecayEta = etaProbe;
rim.diagnostics.farFieldDecay = max(abs([ ...
    yProbe(2), ...
    yProbe(4), ...
    yProbe(5), ...
    yProbe(7)-1]));

if abs(Pr-1) < 100*eps
    rim.diagnostics.identityError = ...
        max(abs(rim.Theta + rim.V - 1));
else
    rim.diagnostics.identityError = NaN;
end

end

function dydeta = rimODE(eta,y,Pr)

F      = y(1,:);
Fp     = y(2,:);
Fpp    = y(3,:);
Pi     = y(4,:);
Theta  = y(5,:);
Thetap = y(6,:);
Vp     = y(8,:);

dydeta = [ ...
    Fp;
    Fpp;
    (3/5)*F.*Fpp ...
        - (1/5)*Fp.^2 ...
        - (2/5)*(Pi-eta.*Theta);
    Theta;
    Thetap;
    (3/5)*Pr*F.*Thetap;
    Vp;
    (3/5)*F.*Vp];

end

function residual = rimBC(ya,yb)

residual = [ ...
    ya(1);
    ya(2);
    ya(5)-1;
    ya(7);
    yb(2);
    yb(4);
    yb(5);
    yb(7)-1];

end

function y0 = initialGuess(eta)

e = exp(-eta);
F = -(1-(1+eta)*e);

y0 = [ ...
    F;
    -eta*e;
    -(1-eta)*e;
    -e;
    e;
    -e;
    1-e;
    e];

end

function s = setDefault(s,name,value)

if ~isfield(s,name) || isempty(s.(name))
    s.(name) = value;
end

end
