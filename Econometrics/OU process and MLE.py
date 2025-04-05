
#### Ornstein-Uhlenbeck (OU) process simulation & Maximum Likelihood Estimation (MLE) ####

# mean-reverting stochastic process for long-term mean reverting asset classes

import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from scipy.integrate import quad

#global parameters:

N = int(35000)  # time steps
paths = int(3500)  # number of paths
T = 5   # time period(years)
kappa = 3   # mean reversion coefficient
theta = 0.5  # long term mean
sigma = 0.5  # volatility coefficient
X0 = 1 # initial process value
seed = 42 #np random seed for reproducibility

class OrnsteinUhlenbeckProcess:
    
    def __init__(self, N=N, paths=paths, T=T, kappa=kappa, theta=theta, sigma=sigma, X0=X0, seed=seed):

        np.random.seed(seed)
        
        self.N = N
        self.paths = paths
        self.T = T
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.X0 = X0
        self.T_vec, self.dt = np.linspace(0, T, N, retstep=True)
        self.std_asy = np.sqrt(sigma**2 / (2 * kappa))
        self.X = None
    
    def simulate_paths(self): #exact discretization 

        self.X = np.zeros((self.N, self.paths))
        self.X[0, :] = self.X0
        W = ss.norm.rvs(loc=0, scale=1, size=(self.N - 1, self.paths)) #independent log-normal numbers for simulation
        
        std_dt = np.sqrt(self.sigma**2 / (2 * self.kappa) * (1 - np.exp(-2 * self.kappa * self.dt))) #Std for time increment dt
        
        for t in range(self.N - 1):
            self.X[t + 1, :] = (
                self.theta +
                np.exp(-self.kappa * self.dt) * (self.X[t, :] - self.theta) + std_dt * W[t, :]
            )
    
    def analyze_final_distribution(self): #Analyze the distribution of the OU process at terminal time T.

        if self.X is None:
            raise ValueError(" Yo dude, simulate OU path first!")
        
        X_T = self.X[-1, :]

        mean_T = self.theta + np.exp(-self.kappa * self.T) * (self.X0 - self.theta)
        std_T = np.sqrt(self.sigma**2 / (2 * self.kappa) * (1 - np.exp(-2 * self.kappa * self.T)))
        param = ss.norm.fit(X_T) # Fit a normal distribution to the simulated terminal values
        print(f"made-up mean = {mean_T:.6f} and made-up Std = {std_T:.6f}")
        print("parm from the fitted MLE: mean = {0:.6f}, Std = {1:.6f}".format(*param))
        
        x = np.linspace(X_T.min(), X_T.max(), 100)# Plot the fitted density over the histogram of simulated values
        pdf_fitted = ss.norm.pdf(x, *param)
        
        plt.figure(figsize=(8, 5))
        plt.plot(x, pdf_fitted, color="orange", label="Normal Density")
        plt.hist(X_T, density=True, bins=50, facecolor="LightBlue", alpha=0.6, label="Simulation of X(T)")
        plt.title("Hist of X(T) with Fitted Normal Distribution")
        plt.xlabel("X(T)")
        plt.legend()
        plt.show()
        
    def plot_paths(self, N_processes=10):
        
        if self.X is None:
            raise ValueError(" yo bro, re-simulate the other paths.")
            
        #vis the plot path
        
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(self.T_vec, self.X[:, :N_processes], linewidth=0.5, label="OU paths")
        ax.plot(self.T_vec, self.theta * np.ones_like(self.T_vec), label="Long-term mean", color="black", linestyle="--")
        ax.plot(self.T_vec, (self.theta + self.std_asy) * np.ones_like(self.T_vec), label="Asymp Std +1", color="yellow", linestyle=":")
        ax.plot(self.T_vec, (self.theta - self.std_asy) * np.ones_like(self.T_vec), label="Asymp Std -1", color="yellow", linestyle=":")
        ax.set_title(f"{N_processes} constructed OU Process Paths")
        ax.set_xlabel("Time scale")
        plt.show()
    
    def compute_covariance(self, n1, n2): #comparing the made up and real covariance

        if self.X is None:
            raise ValueError("yo bro, this not simulated properly my guy")
        
        t1 = n1 * self.dt
        t2 = n2 * self.dt
        
        cov_th = self.sigma**2 / (2 * self.kappa) * (np.exp(-self.kappa * np.abs(t1 - t2)) - np.exp(-self.kappa * (t1 + t2)))    # Theoretical covariance for the OU process between times t1 and t2
        cov_emp = np.cov(self.X[n1, :], self.X[n2, :])[0, 1]
        print(f"Made up cov] = {cov_th:.4f} (t1 = {t1:.4f}, t2 = {t2:.4f})")
        print(f"real cov] = {cov_emp:.4f}")
    
    def estimate_parameters_ols(self): #discrete-time regression model
        
        if self.X is None:
            raise ValueError("simulation didn't work dudebro")
        
        X_single = self.X[:, 1]
        XX = X_single[:-1]
        YY = X_single[1:]
        
        #param linear regression
        beta, alpha, _, _, _ = ss.linregress(XX, YY)
        kappa_ols = -np.log(beta) / self.dt #The volatility sigma is estimated from the residuals.
        theta_ols = alpha / (1 - beta)
        residuals = YY - beta * XX - alpha
        std_resid = np.std(residuals, ddof=2)
        sig_ols = std_resid * np.sqrt(2 * kappa_ols / (1 - beta**2))
        
        print("\nOLS Estimation from a Single Path:")
        print(f"new theta = (long-term mean): {theta_ols:.6f} (True theta: {self.theta})")
        print(f"new sigma = (volatility): {sig_ols:.6f} (True sigma: {self.sigma})")
        print(f"new kappa = (mean reversion rate): {kappa_ols:.6f} (True kappa: {self.kappa})")
        
        return {"theta_ols": theta_ols, "kappa_ols": kappa_ols, "sigma_ols": sig_ols}
    
    def estimate_parameters_mle(self): #MLE from single path

        if self.X is None:
            raise ValueError(" MLE can't work if OU not simulated dude")
        
        X_single = self.X[:, 1]
        N_eff = len(X_single) - 1  # nb of observation
        XX = X_single[:-1]
        YY = X_single[1:]
        Sx = np.sum(XX)
        Sy = np.sum(YY)
        Sxx = np.dot(XX, XX)
        Sxy = np.dot(XX, YY)
        Syy = np.dot(YY, YY)
        N = N_eff
        

        theta_mle = (Sy * Sxx - Sx * Sxy) / (N * (Sxx - Sxy) - (Sx**2 - Sx * Sy))  # Closed-form MLE for theta
       
        kappa_mle = -(1 / self.dt) * np.log((Sxy - theta_mle * Sx - theta_mle * Sy + N * theta_mle**2) / (Sxx - 2 * theta_mle * Sx + N * theta_mle**2)) # Estimate kappa from the regression parameters
        
        sigma2_hat = (
            Syy - 2 * np.exp(-kappa_mle * self.dt) * Sxy
            + np.exp(-2 * kappa_mle * self.dt) * Sxx
            - 2 * theta_mle * (1 - np.exp(-kappa_mle * self.dt)) * (Sy - np.exp(-kappa_mle * self.dt) * Sx)
            + N * theta_mle**2 * (1 - np.exp(-kappa_mle * self.dt)) ** 2
        ) / N
        sigma_mle = np.sqrt(sigma2_hat * 2 * kappa_mle / (1 - np.exp(-2 * kappa_mle * self.dt)))# Estimate sigma^2 based on the likelihood
        
        print("\nSingle path MLE estimation:")
        print(f"theta = (long-term mean): {theta_mle:.6f} (True theta: {self.theta})")
        print(f"kappa = (mean reversion rate): {kappa_mle:.6f} (True kappa: {self.kappa})")
        print(f"sigma = (volatility): {sigma_mle:.6f} (True sigma: {self.sigma})")
        
        return {"theta_mle": theta_mle, "kappa_mle": kappa_mle, "sigma_mle": sigma_mle}
    
    def first_passage_time_analysis(self): #paths overtime and numerical integration

        if self.X is None:
            raise ValueError("not working dude.")
        
        if self.X0 > self.theta:
            condition = self.X <= self.theta
        else:
            condition = self.X >= self.theta
        
        first_indices = np.argmax(condition, axis=0)
        T_to_theta = first_indices * self.dt
        
        print(f"Empirical expected contact time from X0 to theta: {T_to_theta.mean():.4f}")
        print(f"Standard error of the contact time: {ss.sem(T_to_theta):.4f}")
        print(f"Standard deviation of the contact time: {T_to_theta.std():.4f}")
        
        # Standardize the initial deviation for theoretical analysis
        C = (self.X0 - self.theta) * np.sqrt(2 * self.kappa) / self.sigma
        
        # Plot the empirical histogram and the theoretical first hitting time density
        x_vals = np.linspace(T_to_theta.min(), T_to_theta.max(), 100)
        plt.figure(figsize=(10, 4))
        plt.plot(x_vals, self.kappa * OrnsteinUhlenbeckProcess.density_T_to_theta(self.kappa * x_vals, C), color="red", label="made up contact Time Density")
        plt.hist(T_to_theta, density=True, bins=100, facecolor="LightBlue", alpha=0.6, label="Empirical Hitting Times")
        plt.title("First PTime Distribution from X0 to theta")
        plt.xlabel("Time")
        plt.legend()
        plt.show()
        
        # Compute the theoretical expected hitting time and standard deviation via numerical integration
        theoretical_T = quad(lambda t: t * self.kappa * OrnsteinUhlenbeckProcess.density_T_to_theta(self.kappa * t, C), 0, 1000)[0]
        theoretical_std = np.sqrt(
            quad(lambda t: (t - theoretical_T)**2 * self.kappa * OrnsteinUhlenbeckProcess.density_T_to_theta(self.kappa * t, C), 0, 1000)[0]
        )
        print("Made up expected contact time: {:.4f}".format(theoretical_T))
        print("Made up std of the contact time: {:.4f}".format(theoretical_std))
    
    def density_T_to_theta(t, C):
        # Parameters:
        # t : Scaled time variable (positive)
        # C : Standardized initial deviation

        return (np.sqrt(2 / np.pi) * np.abs(C) * np.exp(-t) /
                (1 - np.exp(-2 * t))**(3/2) *
                np.exp(- (C**2 * np.exp(-2 * t)) / (2 * (1 - np.exp(-2 * t))))
               )

if __name__ == "__main__":
    # Create an instance of the OU process simulation with default parameters
    ou = OrnsteinUhlenbeckProcess()
    ou.simulate_paths()
    ou.plot_paths(N_processes=20)
    ou.analyze_final_distribution()
    
    # Compute covariance
    n1 = 4550
    n2 = 5300
    ou.compute_covariance(n1, n2)
    ou.estimate_parameters_ols()    # Estimate parameters using OLS from a single path
    ou.estimate_parameters_mle()    # Estimate parameters using MLE from a single path
    ou.first_passage_time_analysis()    # Perform first passage time analysis from X0 to theta