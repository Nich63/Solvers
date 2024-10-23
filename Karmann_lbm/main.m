clear; close;
addpath("funcs/")

% physical setups
Re = 100;
mu = Re^-1;
u_inlet = 1.1;

% computation setups
t_total = 1; dt = 1e-4;
n_step = t_total / dt;

% computational domain
Nx = 512; Lx = 1;
Ny = 128; Ly = 0.25;

% initialization
xs = linspace(0, Lx, Nx+1);
ys = linspace(0, Ly, Ny+1);
[xx, yy] = ndgrid(xs, ys);
dx = xs(2)-xs(1); dy = ys(2)-ys(1);
x_min = xx(1,1); x_max = xx(end,1); y_min = yy(1,1); y_max = yy(1,end);

% 3 set of mesh grids: u, v, p
x_u = linspace(0, Lx-dx, Nx); y_u = linspace(dy/2, Ly-dy/2, Ny);
x_v = linspace(dx/2, Lx-dx/2, Nx); y_v = linspace(dy, Ly, Ny);
x_p = linspace(dx/2, Lx-dx/2, Nx); y_p = linspace(dy/2, Ly-dy/2, Ny);
[xx_u,yy_u] = ndgrid(x_u, y_u);

% variables
u_0 = ones(Nx, Ny); v_0 = zeros(Nx, Ny); p = zeros(Nx, Ny);
f_x = zeros(Nx, Ny); f_y = zeros(Nx, Ny);

% cylinder grids
cyl_center = [0.7, 0.125];
cyl_r = 0.05;
cyl_info = [cyl_center, cyl_r];

% search grid points where near distance h of cylinder
h = sqrt(dx^2 + dy^2);

info_ib_u = dist_search(x_u, y_u, h, cyl_info);
info_ib_v = dist_search(x_v, y_v, h, cyl_info);

% Enforce BCs
u_0(1,:) = u_inlet;

% wrap u,v,p,fs into a matrix
Vars = cat(3,u_0,v_0,p,f_x,f_y);
% and infos
info = [dx, dy, mu];

%****** CALCULATION BEGIN! *******
%****** BEFORE LOOP: Euler forward to start
% one step

H = -u_0.*ddx_central(u_0,dx) - vbar(v_0).*ddy_central(u_0,dy) +...
    mu*(d2dx2(u_0,dx)+d2dy2(u_0,dy));

% build a H_u for t=1
Hu = H_u(Vars, info);
Hv = H_v(Vars, info);

% Euler forward in time: u_1 = (Hu - Dp) * dt + u_0
dp1 = ddx_central(p,dx);
u_tmp = (Hu - dpdx(p,dx)) .* dt + u_0;
v_tmp = (Hv - dpdy(p,dy)) .* dt + v_0;

% Enforce bc
[u_tmp, v_tmp] = enforceBC(u_tmp, v_tmp, u_inlet);

V_x = V_cal(u_tmp, info_ib_u);
V_y = V_cal(v_tmp, info_ib_v);

f_x = check_f((V_x - u_0)/dt - Hu + dpdx(p,dx), info_ib_u);
f_y = check_f((V_y - v_0)/dt - Hv + dpdy(p,dy), info_ib_v);

% Euler again with f
u_tmp = (Hu - dpdx(p,dx) + f_x) .* dt + u_0;
v_tmp = (Hv - dpdy(p,dy) + f_y) .* dt + v_0;
[u_tmp, v_tmp] = enforceBC(u_tmp, v_tmp, u_inlet);

% now Poisson eqn: L(dp) = div(u,v) / dt
% compute div(u,v) on p grid
div = div_cal(u_0,v_0,dx,dy)/dt;
div(1,:) = 0;

% build a Laplcian matrix for P
L = getL(Nx,Ny,dx,dy);
b = div'; b = b(:);
dp = L \ b;

dd = L * dp; dd = reshape(dd, Ny, Nx)';
% to-do: Laplcian 的问题？会有奇异值？fix一个点？

dp = reshape(dp, Ny, Nx)';

    aa = d2dx2(dp,dx) + d2dy2(dp,dy);
    bb = div_cal(u_tmp,v_tmp,dx,dy) / dt;
    cc = aa-bb;
    
    ee = dd - aa;

u_1 = u_tmp - dt * dpdx(dp, dx);
v_1 = v_tmp - dt * dpdy(dp, dy);
p = p + dp;
[u_1, v_1] = enforceBC(u_1, v_1, u_inlet);

pcolor(xx_u,yy_u,u_1);
shading interp
colorbar;
drawnow

%******* START LOOP *************
for itr = 1:n_step
    Vars_0 = cat(3,u_0,v_0,p,f_x,f_y);
    Vars_1 = cat(3,u_1,v_1,p,f_x,f_y);

    u_tmp = (1.5*H_u(Vars_1,info) - 0.5*H_u(Vars_0,info) - dpdx(p,dx))*dt...
        + u_1;
    v_tmp = (1.5*H_v(Vars_1,info) - 0.5*H_v(Vars_0,info) - dpdy(p,dy))*dt...
        + v_1;
    [u_tmp, v_tmp] = enforceBC(u_tmp, v_tmp, u_inlet);

    V_x = V_cal(u_tmp, info_ib_u);
    V_y = V_cal(v_tmp, info_ib_v);
    
    f_x = check_f((V_x - u_0)/dt - Hu + dpdx(p,dx), info_ib_u);
    f_y = check_f((V_y - v_0)/dt - Hv + dpdy(p,dy), info_ib_v);

    % with f
    u_tmp = (1.5*H_u(Vars_1,info) - 0.5*H_u(Vars_0,info)...
        - dpdx(p,dx) + f_x)*dt+ u_1;
    v_tmp = (1.5*H_v(Vars_1,info) - 0.5*H_v(Vars_0,info)...
        - dpdy(p,dy) + f_y)*dt+ v_1;
    [u_tmp, v_tmp] = enforceBC(u_tmp, v_tmp, u_inlet);

    div = div_cal(u_1,v_1,dx,dy)/dt;
    div(1,:) = 0;

    b = div'; b = b(:);
    dp = L \ b;
    dd = L * dp; dd = reshape(dd, Ny, Nx)';
    dp = reshape(dp, Ny, Nx)';

    aa = d2dx2(dp,dx) + d2dy2(dp,dy);
    bb = div_cal(u_tmp,v_tmp,dx,dy) / dt;

    cc = aa-bb;
    ee = dd - aa;

    u_0 = u_1; v_0 = v_1;
    
    u_1 = u_tmp - dt * dpdx(dp, dx);
    v_1 = v_tmp - dt * dpdy(dp, dy);
    p = p + dp;
    [u_1, v_1] = enforceBC(u_1, v_1, u_inlet);

    pcolor(xx_u,yy_u,u_1);
    shading interp
    colorbar;
    drawnow
end