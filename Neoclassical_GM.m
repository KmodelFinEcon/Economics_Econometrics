% for the Neoclassical Growth Model with a convex-concave production function using the Hamilton Jacobi Bellman PDE 
% Implementation by K.Tomov. 
% REF paper by Dr. B.Moll 

%General parameters 

s = 4;
a = 0.2;
r = 0.04;
d = 0.03;
AH = 0.7;
AL = 0.2;
kappa = 3; %fixed cost

kssH = (a*AH/(r+d))^(1/(1-a)) + kappa;
kstar = kappa./(1-(AL/AH).^(1/a));

I = 10000; %iterationns
kmin = 0.001*kssH;
kmax = 1.3*kssH;
k = linspace(kmin,kmax,I)';
dk = (kmax-kmin)/(I-1);

%production function
yH = AH*max(k - kappa,0).^a; yL = AL*k.^a;
y = max(yH,yL);
plot(k,y,k,yH,'--',k,yL,'--','LineWidth',2)
xlabel('$k$','FontSize',16,'interpreter','latex')
ylabel('$f(k)$','FontSize',16,'interpreter','latex')
print -depsc y.eps

maxit=10000;
crit = 10^(-6);
Delta = 1040;

dVf = zeros(I,1);
dVb = zeros(I,1);
c = zeros(I,1);

%INITIAL GUESS of paths
v0 = (k.^a).^(1-s)/(1-s)/r;
v = v0;

for n=1:maxit
    V = v;
    dVf(1:I-1) = (V(2:I)-V(1:I-1))/dk;
    dVf(I) = (y(I) - d*kmax)^(-s); %state constraint, for stability
    dVb(2:I) = (V(2:I)-V(1:I-1))/dk;
    dVb(1) = (y(1) - d*kmin)^(-s); %state constraint, for stability
       
    cf = max(dVf,10^(-10)).^(-1/s);
    muf = y - d.*k - cf;
    Hf = cf.^(1-s)/(1-s) + dVf.*muf;
    
    cb = max(dVb,10^(-10)).^(-1/s);
    mub = y - d.*k - cb;
    Hb = cb.^(1-s)/(1-s) + dVb.*mub;
    
    c0 = y - d.*k;
    dV0 = max(c0,10^(-10)).^(-s);
    H0 = c0.^(1-s)/(1-s);
      
    Ineither = (1-(muf>0)) .* (1-(mub<0));
    Iunique = (mub<0).*(1-(muf>0)) + (1-(mub<0)).*(muf>0);
    Iboth = (mub<0).*(muf>0);
    Ib = Iunique.*(mub<0) + Iboth.*(Hb>=Hf);
    If = Iunique.*(muf>0) + Iboth.*(Hf>=Hb);
    I0 = Ineither;
    
    c  = cf.*If + cb.*Ib + c0.*I0;
    u = c.^(1-s)/(1-s);

    X = -Ib.*mub/dk;
    Y = -If.*muf/dk + Ib.*mub/dk;
    Z = If.*muf/dk;

    A =spdiags(Y,0,I,I)+spdiags(X(2:I),-1,I,I)+spdiags([0;Z(1:I-1)],1,I,I);
    
    if max(abs(sum(A,2)))>10^(-12)
        disp('Improper Transition Matrix')
        break
    end    
    
    B = (r + 1/Delta)*speye(I) - A;
    b = u + V/Delta;
    V = B\b; %SOLVE SYSTEM OF EQUATIONS
    Vchange = V - v;
    v = V;   

    dist(n) = max(abs(Vchange));
    if dist(n)<crit
        disp('Value Function Converged, Iteration = ')
        disp(n)
        break
    end
end
toc;

%plotting the results

set(gca,'FontSize',14)
plot(dist,'LineWidth',2)
grid
xlabel('Iteration')
ylabel('||V^{n+1} - V^n||')

kdot = y - d.*k - c;
dV_Upwind = dVf.*If + dVb.*Ib + dV0.*I0;
Verr = c.^(1-s)/(1-s) + dV_Upwind.*kdot - r.*V;

set(gca,'FontSize',12)
plot(k,Verr,'LineWidth',2)
grid
xlabel('k')
ylabel('issue in the HJB Equation')
xlim([kmin kmax])

subplot(2,2,1)
plot(k,y,k,yH,'--',k,yL,'--',kstar,AL*kstar.^a,'o','LineWidth',2)
set(gca,'FontSize',12)
grid
xlabel('$k$','FontSize',14,'interpreter','latex')
ylabel('$f(k)$','FontSize',14,'interpreter','latex')
% print -depsc y.eps

[obj, index] = min(abs(kstar-k));
subplot(2,2,2)
set(gca,'FontSize',12)
plot(k,V,kstar,V(index),'ok','LineWidth',2)
grid
xlabel('k')
ylabel('V(k)')
xlim([kmin kmax])

subplot(2,2,3)
plot(k,c,k,y-d.*k,'LineWidth',2)
set(gca,'FontSize',16)
legend('Consumption, c(k)','Production net of depreciation, f(k) - \delta k','Location','NorthWest')
grid
xlabel('k')
ylabel('c(k)')
xlim([kmin kmax])

subplot(2,2,4)
plot(k,kdot,k,zeros(1,I),'--',kstar,0,'ok','LineWidth',2)
set(gca,'FontSize',14)
grid
xlabel('$k$','FontSize',16,'interpreter','latex')
ylabel('$s(k)$','FontSize',16,'interpreter','latex')
xlim([kmin kmax])