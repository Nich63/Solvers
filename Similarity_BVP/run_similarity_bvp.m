function rim = run_similarity_bvp(makePlot)
% Standalone heated-disk rim similarity example. No toolbox required.
if nargin < 1, makePlot = true; end
rim = solve_rim_bvp(struct('Pr',1,'etaMax',20));
fprintf('Pr=1: Fpp(0)=%.10g, -Theta''(0)=%.10g, identity=%.3e\n', ...
    rim.constants.FppWall,rim.constants.heatFlux,rim.diagnostics.identityError);
if makePlot
    folder = fullfile(fileparts(mfilename('fullpath')),'output');
    if ~isfolder(folder), mkdir(folder); end
    f=figure('Name','Rim similarity');
    plot(rim.eta,[rim.Fp;rim.V;rim.Theta].','LineWidth',1.4);
    xlabel('\eta'); ylabel('Dimensionless profiles');
    legend('F''','V','\Theta','Location','best'); grid on;
    print(f,fullfile(folder,'profiles.png'),'-dpng','-r150');
end
end
