function result = run_boundary_layer(makePlot)
% Requires MATLAB and Optimization Toolbox (fsolve).
if nargin < 1, makePlot=true; end
assert(license('test','Optimization_Toolbox'),'Optimization Toolbox is required.');
result=solve_global_boundary_layer(example_bl_config());
assert(result.completed,'March failed; inspect result.failure.');
fprintf('Accepted %d/%d stations; max cell residual %.3e\n', ...
    result.nAccepted,result.nRequested,max(result.diagnostics.cellResidualInf(2:end)));
if makePlot
    folder=fullfile(fileparts(mfilename('fullpath')),'output');
    if ~isfolder(folder), mkdir(folder); end
    f=figure('Name','Implicit boundary layer');
    tiledlayout(1,2);
    nexttile; plot(result.U(:,[1,end]),result.z,'LineWidth',1.4);
    xlabel('U=ru'); ylabel('z'); legend('inlet','last station'); grid on;
    nexttile; plot([result.V(:,end),result.Theta(:,end)],result.z,'LineWidth',1.4);
    xlabel('V, \Theta'); ylabel('z'); legend('V','\Theta'); grid on;
    print(f,fullfile(folder,'profiles.png'),'-dpng','-r150');
end
end
