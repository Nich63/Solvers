clear;

Bi = 0.05;
Lc = 40;

m = (Bi*Lc)^0.5;

% power of Fourier series
n = 100;

n_eta = 1000;
eta = linspace(0,1,n_eta);
n_ksi = 1000;

% final dimensionless time
ksi_end = 0.06;
ksi = linspace(0,ksi_end,n_ksi);

Te = 20;
T0 = 100;

Ts = 1/cosh(m)*cosh(m*eta);

% steady-state T at tip
Tem_c = Ts(1)*0.9;

lams = zeros(n,1);
Ans = zeros(n,1);
Tt = zeros(n_ksi,n_eta);
for i = 1:n
    lam = (2*i-1)/2*pi;
    lams(i) = lam;
    An = -2/cosh(m)*(cosh(m)*sin(lam)/lam)*(1+m^2/lam^2)^-1;
    Ans(i) = An;
    for j = 1:n_ksi
        Tt(j,:) = Tt(j,:) + An*exp(-(lam^2+m^2)*ksi(j))*cos(lam*eta);
    end
end

theta = zeros(n_ksi,n_eta);

for k = 1:n_ksi
    theta(k,:) = Ts + Tt(k,:);
end

% characteristic time get

theta_at_tip = theta(:,1);
err = (theta_at_tip - Tem_c).^2;

[min, pos] = min(err);

time_c = ksi(pos);

%%
fprintf(['特征时间的值为',num2str(time_c),'\n']);

for kk = 2:n_ksi/10:n_ksi
    plot(eta,theta(kk,:),'LineWidth',1.5);
    hold on
    grid on
end
labels = num2str(((2:n_ksi/10:n_ksi)/n_ksi*ksi_end)');
title('Bi = 0.05, L_c = 40','FontSize',15);
ylabel('\theta','FontSize',20);
xlabel('\eta','FontSize',20);
legend(labels);