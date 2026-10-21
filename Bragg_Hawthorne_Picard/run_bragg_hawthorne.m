function out = run_bragg_hawthorne(makePlot)
if nargin<1, makePlot=true; end
[g,source,side,exact]=bh_manufactured_case(65);
out=solve_bh_picard(g,source,side);
assert(out.converged,'Picard did not converge.');
fprintf('Picard: %d iterations, PDE residual %.3e, max error %.3e\n', ...
    out.iterations,out.pdeResidual,max(abs(out.psi-exact),[],'all'));
if makePlot
    folder=fullfile(fileparts(mfilename('fullpath')),'output');
    if ~isfolder(folder), mkdir(folder); end
    f=figure('Name','Bragg-Hawthorne manufactured solution'); tiledlayout(1,2);
    nexttile; contourf(g.r,g.z,out.psi.',20); colorbar; xlabel('r'); ylabel('z'); title('\psi');
    nexttile; semilogy(out.history,'LineWidth',1.4); grid on;
    xlabel('Picard iteration'); ylabel('Raw residual'); legend('PDE','boundary');
    print(f,fullfile(folder,'solution.png'),'-dpng','-r150');
end
end
