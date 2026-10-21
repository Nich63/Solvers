function [A,b] = assemble_bh_system(g,rhs,psiSide)
% Assemble Delta*psi = psi_rr - psi_r/r + psi_zz on a nonuniform grid.
% Axis/wall: psi=0; side: psiSide; top: psi_z=0. Dirichlet wins at corners.

Nr = g.Nr; Nz = g.Nz; r = g.r(:); z = g.z(:);
if isscalar(rhs); rhs = rhs*ones(Nr,Nz); end
psiSide = psiSide(:);

nnzEst = 5*g.N;
I = zeros(nnzEst,1); J = zeros(nnzEst,1); V = zeros(nnzEst,1);
b = zeros(g.N,1);
c = 0;

    function push(ii,jj,vv)
        c = c + 1; I(c) = ii; J(c) = jj; V(c) = vv;
    end

for j = 1:Nz
    for i = 1:Nr
        k = g.idx(i,j);

        % ---- Dirichlet boundaries ----
        if i == 1                       % axis
            push(k,k,1); b(k) = 0; continue
        end
        if j == 1                       % wall
            push(k,k,1); b(k) = 0; continue
        end
        if i == Nr                      % lateral inlet
            push(k,k,1); b(k) = psiSide(j); continue
        end

        % ---- top: Neumann psi_z = 0, second-order one-sided ----
        if j == Nz
            h1 = z(Nz)   - z(Nz-1);
            h2 = z(Nz-1) - z(Nz-2);
            push(k,k,                (2*h1+h2)/(h1*(h1+h2)));
            push(k,g.idx(i,Nz-1),  -(h1+h2)/(h1*h2));
            push(k,g.idx(i,Nz-2),    h1/(h2*(h1+h2)));
            b(k) = 0; continue
        end

        % ---- interior ----
        hrm = r(i) - r(i-1);  hrp = r(i+1) - r(i);
        hzm = z(j) - z(j-1);  hzp = z(j+1) - z(j);

        % psi_rr
        arr_m =  2/(hrm*(hrm+hrp));
        arr_p =  2/(hrp*(hrm+hrp));
        arr_0 = -2/(hrm*hrp);
        % psi_r
        ar_m = -hrp/(hrm*(hrm+hrp));
        ar_p =  hrm/(hrp*(hrm+hrp));
        ar_0 =  (hrp-hrm)/(hrm*hrp);
        % psi_zz
        azz_m =  2/(hzm*(hzm+hzp));
        azz_p =  2/(hzp*(hzm+hzp));
        azz_0 = -2/(hzm*hzp);

        oneOverR = 1/r(i);

        push(k,g.idx(i-1,j), arr_m - oneOverR*ar_m);
        push(k,g.idx(i+1,j), arr_p - oneOverR*ar_p);
        push(k,g.idx(i,j-1), azz_m);
        push(k,g.idx(i,j+1), azz_p);
        push(k,k,            arr_0 - oneOverR*ar_0 + azz_0);
        b(k) = rhs(i,j);
    end
end

A = sparse(I(1:c),J(1:c),V(1:c),g.N,g.N);

end
