clear
clc
L = 1;
N_cell = 100;
a = 1;
b = 1;
dx = L/N_cell;
dt = 1E-4;

x = linspace(0,L,N_cell);
theta = [zeros(1,N_cell+3),ones(1,3)];

for i = 1:1000
    theta(1:3)=fliplr(theta(4:6));
    px = (theta(3:end)-theta(1:end-2))/(2*dx);
    r = (px(2:end-1)-px(1:end-2)+1e-6)./(px(3:end)-px(2:end-1)+1e-6);
    phi = (r+abs(r))./(1+abs(r));
    px_hlf = px(2:end-1) + phi.*(px(3:end)-px(2:end-1))/2;
    ppx = (px_hlf(2:end-1)-px_hlf(1:end-2))./dx;
    theta(4:end-3)  = theta(4:end-3) - a*dt.*theta(4:end-3) + b*dt.*ppx;
end
plot(x,theta(4:end-3));
