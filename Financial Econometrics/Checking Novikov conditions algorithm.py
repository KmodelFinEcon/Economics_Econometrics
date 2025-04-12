
#checking novikov conditions exercise for risk neutral pricing: Analytical vs MC

import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt

def novikov_condition_verification(theta_t, T, num_samples=15000):
    try:
        integral = integrate.quad(lambda t: 0.5 * theta_t(t)**2, 0, T)[0]
        theoretical_exp = np.exp(integral)
        print(f"Analytical expectation is {theoretical_exp}")
    except:
        theoretical_exp = None #not result outputed
        print("cannot compute")
    
    # checking the conditions using MC : Simulate the integral (0.5*theta_t^2 dt) from 0 to T
    samples = []
    for _ in range(num_samples):
        t_values = np.linspace(0, T, 1000)
        dt = T / 1000
        integral = 0
        for t in t_values:
            integral += 0.5 * theta_t(t)**2 * dt
        
        samples.append(np.exp(integral))
    
    empirical_exp = np.mean(samples)
    empirical_std = np.std(samples)
    
    print(f"Empirical expectation is {empirical_exp} ± {empirical_std}")
    
    if empirical_exp < 1e5: 
        print("Novikov's condition is satisfied")
        return True, empirical_exp
    else:
        print("Novikov's condition is not statisfied")
        return False, empirical_exp

print("version with constant process theta")
theta_const = lambda t: 0.2
novikov_condition_verification(theta_const, T=1)

print("\nversion with deterministic theta")
theta_linear = lambda t: t
novikov_condition_verification(theta_linear, T=1)

print("\nversion with exponetial theta")
theta_explosive = lambda t: 1/(1-t) if t < 0.999 else 1000  # Avoid exact division by 1
novikov_condition_verification(theta_explosive, T=0.999)

t_values = np.linspace(0, 1, 100)
plt.figure(figsize=(10, 6))
plt.plot(t_values, [theta_const(t) for t in t_values], label='Constant')
plt.plot(t_values, [theta_linear(t) for t in t_values], label='Linear')
plt.plot(t_values, [theta_explosive(t) if t < 0.999 else np.nan for t in t_values], label='Exponetial theta')
plt.xlabel('T')
plt.ylabel('0(t)')
plt.title('Comparision of different thetas')
plt.legend()
plt.grid(True)
plt.show()