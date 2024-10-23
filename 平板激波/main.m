clear;

addpath("funs/")

% sea-level conditions
P_inf = 101300; T_inf = 288.15; rho_inf = 1.225; R = 287;
cp = 1005; cv = 718; gamma = 1.4; mu0 = 1.735e-5;
S1 = 110.4; Pr = 0.71;

% parameters
M = 3; L = 1e-5; H = 8e-6;
Nx = 75; Ny = 80;
u_inf = M * sound_speed(T_inf);

% initialization
xs = linspace(0, L, Nx);
ys = linspace(0, H, Ny);
[xx, yy] = ndgrid(xs, ys);
dx = xs(2)-xs(1); dy = ys(2)-ys(1);
x_min = xx(1,1); x_max = xx(end,1); y_min = yy(1,1); y_max = yy(1,end);

% primitive vars
u = ones(Nx, Ny).*u_inf; v = zeros(Nx, Ny);
P = ones(Nx, Ny).*P_inf; T = ones(Nx, Ny).*T_inf;

% ENFORCE BCs
[u, v, P, T] = enforce_BCs(u, v, P, T ,M);
rho = P ./ (R.*T);

% conservative vars initialize
U = prim2cons(rho, u, v, T, cv);
E = zeros(size(U)); F = zeros(size(U)); U_bar = zeros(size(U));
E_bar = zeros(size(U)); F_bar = zeros(size(U));

% physical parameters
[a, mu, k] = phy_paras(T);

dt = CFL(u, v, dx, dy, T, mu, rho, 0.6);
t=0;

figure(2);
h = animatedline;
xlabel('Number of Iterations');
ylabel('RMSE');
title('RMSE for E_t During Iteration');

for n_step = 1:1500
    [rho, u, v, T, P, Et, ~] = cons2prim(U, R, cv);
    Et_old = Et; % convergence checker
    % to-do: I/O
    % dt CFL
    dt = CFL(u, v, dx, dy, T, mu, rho, 0.6);
    t = t+dt;
    
    % PREDICTOR
    [~, mu, k] = phy_paras(T);

    % E, F: fwd in total
    dudx_bwd = ddx_bwd(u,dx); dudy_bwd = ddy_bwd(u,dy);
    dvdx_bwd = ddx_bwd(v,dx); dvdy_bwd = ddy_bwd(v,dy);
    dudx_cent = ddx_central(u,dx); dvdx_cent = ddx_central(v,dx);
    dudy_cent = ddy_central(u,dy); dvdy_cent = ddy_central(v,dy);
    dTdx_bwd = ddx_bwd(T,dx); dTdy_bwd = ddy_bwd(T,dy);
    
    E(1,:,:) = rho.*u;
    E(2,:,:) = rho.*u.^2 + P - 2*mu.*(dudx_bwd - 1/3*(dudx_bwd+dvdy_cent));
    E(3,:,:) = rho.*u.*v - mu.*(dudy_cent + dvdx_bwd);
    E(4,:,:) = (Et + P - 2*mu.*(dudx_bwd - 1/3*(dudx_bwd+dvdy_cent))).*u...
        - v.*mu.*(dudy_cent + dvdx_bwd) - k.*dTdx_bwd;

    F(1,:,:) = rho.*v;
    F(2,:,:) = rho.*u.*v - mu.*(dudy_bwd+dvdx_cent);
    F(3,:,:) = rho.*v.^2 + P - 2*mu.*(dvdy_bwd-1/3*(dudx_cent+dvdy_bwd));
    F(4,:,:) = (Et + P - 2*mu.*(dvdy_bwd-1/3*(dudx_cent+dvdy_bwd))).*v...
        - u.*mu.*(dudy_bwd+dvdx_cent) - k.*dTdy_bwd;

    U_bar = U - dt*dEdx_fwd(E,dx) - dt*dFdy_fwd(F,dy);
    [~, u, v, T, P, ~, ~] = cons2prim(U_bar, R, cv); % update prims
    % ENFORCE BCs
    [u, v, P, T] = enforce_BCs(u, v, P, T ,M);
    rho = P ./ (R.*T);
    % U_bar with BC
    U_bar = prim2cons(rho, u, v, T, cv);
    % update other variables after BC
    [rho, u, v, T, P, Et, ~] = cons2prim(U_bar, R, cv);

    % CORRECTOR
    [~, mu, k] = phy_paras(T);
    % E_bar, F_bar: bwd in total
    % update variables to bar
    dudx_fwd = ddx_fwd(u,dx); dudy_fwd = ddy_fwd(u,dy);
    dvdx_fwd = ddx_fwd(v,dx); dvdy_fwd = ddy_fwd(v,dy);
    dudx_cent = ddx_central(u,dx); dvdx_cent = ddx_central(v,dx);
    dudy_cent = ddy_central(u,dy); dvdy_cent = ddy_central(v,dy);
    dTdx_fwd = ddx_fwd(T,dx); dTdy_fwd = ddy_fwd(T,dy);

    E(1,:,:) = rho.*u;
    E(2,:,:) = rho.*u.^2 + P - 2*mu.*(dudx_fwd - 1/3*(dudx_fwd+dvdy_cent));
    E(3,:,:) = rho.*u.*v - mu.*(dudy_cent + dvdx_fwd);
    E(4,:,:) = (Et + P - 2*mu.*(dudx_fwd - 1/3*(dudx_fwd+dvdy_cent))).*u...
        - v.*mu.*(dudy_cent + dvdx_fwd) - k.*dTdx_fwd;

    F(1,:,:) = rho.*v;
    F(2,:,:) = rho.*u.*v - mu.*(dudy_fwd+dvdx_cent);
    F(3,:,:) = rho.*v.^2 + P - 2*mu.*(dvdy_fwd-1/3*(dudx_cent+dvdy_fwd));
    F(4,:,:) = (Et + P - 2*mu.*(dvdy_fwd-1/3*(dudx_cent+dvdy_fwd))).*v...
        - u.*mu.*(dudy_fwd+dvdx_cent) - k.*dTdy_fwd;

    U = 0.5*(U + U_bar - dt*dEdx_bwd(E,dx) - dt*dFdy_bwd(F,dy));
    [~, u, v, T, P, ~, e] = cons2prim(U, R, cv); % update prims

    % ENFORCE BCs
    [u, v, P, T] = enforce_BCs(u, v, P, T ,M);
    % other variables after BC
    rho = P ./ (R.*T);
    U = prim2cons(rho, u, v, T, cv);
    % update prims
    [rho, u, v, T, P, Et, ~] = cons2prim(U, R, cv);

    % draw: rho, u, v, e, P, T, convergence
    % if mod(n_step, 50) == 0
    %     figure(1)
    %     plot_rho = subplot(2,3,1);
    %     pcolor(xx,yy,rho);
    %     shading interp
    %     axis equal
    %     tem_title = ['\rho @t = ', num2str(t,'%4.4e')];
    %     title(tem_title);
    %     xlabel('X');
    %     ylabel('Y');
    %     colorbar;
    %     % clim([0, 2]);
    %     xlim([x_min, x_max]);
    %     ylim([y_min, y_max]);
    % 
    %     plot_u = subplot(2,3,2);
    %     pcolor(xx,yy,u);
    %     shading interp
    %     axis equal
    %     tem_title = ['u@t = ', num2str(t,'%4.4e')];
    %     title(tem_title);
    %     xlabel('X');
    %     ylabel('Y');
    %     colorbar;
    %     % clim([0, 2]);
    %     xlim([x_min, x_max]);
    %     ylim([y_min, y_max]);
    % 
    %     plot_v = subplot(2,3,3);
    %     pcolor(xx,yy,v);
    %     shading interp
    %     axis equal
    %     tem_title = ['v@t = ', num2str(t,'%4.4e')];
    %     title(tem_title);
    %     xlabel('X');
    %     ylabel('Y');
    %     colorbar;
    %     % clim([0, 2]);
    %     xlim([x_min, x_max]);
    %     ylim([y_min, y_max]);
    % 
    %     plot_e = subplot(2,3,4);
    %     pcolor(xx,yy,e);
    %     shading interp
    %     axis equal
    %     tem_title = ['e@t = ', num2str(t,'%4.4e')];
    %     title(tem_title);
    %     xlabel('X');
    %     ylabel('Y');
    %     colorbar;
    %     % clim([0, 2]);
    %     xlim([x_min, x_max]);
    %     ylim([y_min, y_max]);
    % 
    %     plot_p = subplot(2,3,5);
    %     pcolor(xx,yy,P);
    %     shading interp
    %     axis equal
    %     tem_title = ['P@t = ', num2str(t,'%4.4e')];
    %     title(tem_title);
    %     xlabel('X');
    %     ylabel('Y');
    %     colorbar;
    %     % clim([0, 2]);
    %     xlim([x_min, x_max]);
    %     ylim([y_min, y_max]);
    % 
    %     plot_T = subplot(2,3,6);
    %     pcolor(xx,yy,T);
    %     shading interp
    %     axis equal
    %     tem_title = ['T@t = ', num2str(t,'%4.4e')];
    %     title(tem_title);
    %     xlabel('X');
    %     ylabel('Y');
    %     colorbar;
    %     % clim([0, 2]);
    %     xlim([x_min, x_max]);
    %     ylim([y_min, y_max]);
    % 
    %     colormap jet
    %     set(plot_T, "Colormap", hot);
    % 
    %     drawnow
    % end
    % 
    % figure(2)
    % RMSE = (mean(mean(abs(Et_old - Et)./Et_old).^2)).^0.5;
    % addpoints(h,n_step,RMSE);
    % set(gca, 'YScale', 'log')
    % grid on
    % drawnow;

end

save(['Final_', num2str(M), '.mat'], "u","v","rho","T","P","e","xx","yy");