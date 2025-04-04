
#implementation of the single factor Ramsey–Cass–Koopmans model by. K.Tomov

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from itertools import product

# Global parameters
A = 0.5 #total factor of productivity
a = 0.3 #capital share of production
sigma = 0.10 #rate of capital depreciation
dr = 0.15 #discount rate 
E = 0.2 #elasticity of intertemporal substitution(how households are willing to shift consumption across time)

#production function 

def F(x):
    return A * x**a - sigma * x

def ramsey(t, u): #gordon ramsey cooked in this one!!!
    K, C = u
    if K <= 0:
        return [0, 0]
    dK = A * K**a - sigma * K - C
    dC = (a * A * K**(a - 1) - sigma - dr) * E * C
    return [dK, dC]

def ss():# steady-state capital and consumption
    Kss = (a * A / (sigma + dr))**(1/(1-a))
    Css = A * Kss**a - sigma * Kss
    return {"Kss": Kss, "Css": Css}

def make_events(Kss, Css):
    def event_K_neg(t, u):
        return u[0]
    event_K_neg.terminal = True
    event_K_neg.direction = -1 

    def event_C_neg(t, u):
        return u[1]
    event_C_neg.terminal = True
    event_C_neg.direction = -1

    def event_K_large(t, u):
        return u[0] - 2*Kss
    event_K_large.terminal = True
    event_K_large.direction = 1 
    
    def event_C_large(t, u):
        return u[1] - 2*Css
    event_C_large.terminal = True
    event_C_large.direction = 1

    return [event_K_neg, event_C_neg, event_K_large, event_C_large]

ss_vals = ss()
Kss, Css = ss_vals["Kss"], ss_vals["Css"]

def solve_and_plot(u0, tspan, events=None, **kwargs):
    sol = solve_ivp(fun=ramsey,
                    t_span=tspan,
                    y0=u0,
                    events=events,
                    atol=1e-8, rtol=1e-8,
                    **kwargs)
    plt.plot(sol.y[0], sol.y[1])
    return sol

events = make_events(Kss, Css)

 #first trajectory
u0a = [0.5, 0.2]
tspan = (0.0, 10.0)
sol1 = solve_and_plot(u0a, tspan)  #if no events

# Second trajectory
u0b = [0.8, 0.3]
sol2 = solve_and_plot(u0b, tspan, events=events)

def make_grid(ranges):
    return list(product(*ranges))

def fieldplot(f, U0, tspan):
    ss_vals = ss()
    Kss, Css = ss_vals["Kss"], ss_vals["Css"]
    plt.figure(figsize=(10,10))
    x_vals = np.linspace(0, 2, 300)
    plt.plot(x_vals, [F(x) for x in x_vals], color='black', linewidth=3, label='state 1')
    plt.axvline(x=Kss, color='black', linewidth=3, label='state 2')
    plt.xlim(0, 2)
    plt.ylim(0, 0.6)
    plt.xlabel("K(t)")
    plt.ylabel("C(t)")
    plt.title("Ramsey-Cass-Koopmans Model")


# solver in function of event

    events = make_events(Kss, Css)
    def pathplot(u0, tspan):
        try:
            sol = solve_ivp(fun=f, t_span=tspan, y0=u0, events=events,atol=1e-8, rtol=1e-8)
            plt.plot(sol.y[0], sol.y[1], color='blue', alpha=0.7)
        except Exception:
            plt.plot(u0[0], u0[1], 'ko', markersize=3)
            
    for z in U0:
        u0 = [z[0], z[1]]
        pathplot(u0, tspan)
        tspan2 = (tspan[0], -tspan[1])
        pathplot(u0, tspan2)
    
    return plt

K_range = np.arange(0, 2*Kss + Kss/10, Kss/10)
C_range = np.arange(0, 2*Css + Css/10, Css/10)
U0 = make_grid((K_range, C_range))
plt_field = fieldplot(ramsey, U0, (0.0, 1.0))

plt.show()