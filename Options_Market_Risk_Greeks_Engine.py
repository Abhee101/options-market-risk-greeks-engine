#!/usr/bin/env python
# coding: utf-8

# # 
# 
# Options Market Risk & Greeks-Based Risk Management Engine
# 
# ## Project Overview
# 
# This project develops a derivatives market-risk framework for measuring and managing the sensitivity of an options portfolio to changes in underlying asset prices, volatility, interest rates and time.
# 
# The framework uses the Black-Scholes model to calculate option prices and Greeks, followed by scenario analysis, Greeks-based P&L decomposition, delta hedging and portfolio-level risk analysis.
# 
# ## Objectives
# 
# - Calculate European option prices using Black-Scholes
# - Calculate Delta, Gamma, Vega, Theta and Rho
# - Analyse option sensitivity to market variables
# - Construct an options portfolio
# - Perform market-risk scenario analysis
# - Estimate Greeks-based portfolio P&L
# - Analyse delta-hedging requirements
# - Estimate portfolio VaR
# - Develop an options market-risk dashboard
# 
# ## Technology
# 
# - Python
# - NumPy
# - Pandas
# - SciPy
# - Matplotlib
# - yfinance

# ## Methodology
# 
# Market Data
# ↓
# Underlying Price
# ↓
# Volatility Estimation
# ↓
# Black-Scholes Option Pricing
# ↓
# Option Greeks
# ↓
# Options Portfolio Construction
# ↓
# Market Variable Shocks
# ↓
# Greeks-Based P&L
# ↓
# Delta Hedging
# ↓
# Portfolio VaR
# ↓
# Risk Dashboard

# In[48]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from scipy.stats import norm

print("Libraries imported successfully")

try:
    display
except NameError:
    display = print

try:
    get_ipython
except NameError:
    class DummyIPython:
        def run_line_magic(self, magic, arg):
            pass
    def get_ipython():
        return DummyIPython()


# In[49]:


ticker = "^NSEI"

data = yf.download(
    ticker,
    period="2y",
    auto_adjust=True,
    progress=False
)

data.head()


# In[50]:


prices = data["Close"].squeeze()

prices = prices.dropna()

print("Number of observations:", len(prices))
print("Latest NIFTY price:", round(float(prices.iloc[-1]), 2))


# In[51]:


returns = prices.pct_change().dropna()

print("Number of return observations:", len(returns))
print("Mean daily return:", returns.mean())
print("Daily volatility:", returns.std())


# In[52]:


historical_volatility = returns.std() * np.sqrt(252)

print(
    "Annualized Historical Volatility:",
    f"{historical_volatility:.2%}"
)


# In[53]:


plt.figure(figsize=(10, 5))

plt.plot(prices)

plt.title("NIFTY 50 Price History")
plt.xlabel("Date")
plt.ylabel("Index Level")
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# In[54]:


# Option parameters

S = float(prices.iloc[-1])       # Current NIFTY level
K = 24500                        # Strike price
T = 30 / 365                     # 30 days to maturity
r = 0.065                        # Risk-free rate: 6.5%
sigma = historical_volatility    # Historical volatility

option_type = "call"

print("Underlying Price (S):", S)
print("Strike Price (K):", K)
print("Time to Maturity:", T)
print("Risk-Free Rate:", r)
print("Volatility:", sigma)


# In[55]:


def black_scholes_price(S, K, T, r, sigma, option_type="call"):

    d1 = (
        np.log(S / K)
        + (r + 0.5 * sigma**2) * T
    ) / (sigma * np.sqrt(T))

    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = (
            S * norm.cdf(d1)
            - K * np.exp(-r * T) * norm.cdf(d2)
        )
    else:
        price = (
            K * np.exp(-r * T) * norm.cdf(-d2)
            - S * norm.cdf(-d1)
        )

    return price

def black_scholes_call(S, K, T, r, sigma):
    return black_scholes_price(S, K, T, r, sigma, "call")

def black_scholes_put(S, K, T, r, sigma):
    return black_scholes_price(S, K, T, r, sigma, "put")


# In[56]:


call_price = black_scholes_price(
    S, K, T, r, sigma, "call"
)

print("NIFTY Call Option Price:", round(call_price, 2))


# In[57]:


put_price = black_scholes_price(
    S, K, T, r, sigma, "put"
)

print("NIFTY Put Option Price:", round(put_price, 2))


# In[58]:


parity_left = call_price - put_price
parity_right = S - K * np.exp(-r * T)

print("Call - Put:", round(parity_left, 4))
print("S - PV(K):", round(parity_right, 4))
print("Parity Difference:", round(parity_left - parity_right, 8))


# In[59]:


def option_greeks(S, K, T, r, sigma, option_type="call"):

    d1 = (
        np.log(S / K)
        + (r + 0.5 * sigma**2) * T
    ) / (sigma * np.sqrt(T))

    d2 = d1 - sigma * np.sqrt(T)

    # Delta
    if option_type == "call":
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1

    # Gamma
    gamma = norm.pdf(d1) / (
        S * sigma * np.sqrt(T)
    )

    # Vega
    vega = S * norm.pdf(d1) * np.sqrt(T)

    # Theta
    if option_type == "call":
        theta = (
            -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
            - r * K * np.exp(-r * T) * norm.cdf(d2)
        )
    else:
        theta = (
            -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
            + r * K * np.exp(-r * T) * norm.cdf(-d2)
        )

    # Rho
    if option_type == "call":
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    else:
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)

    return {
        "Delta": delta,
        "Gamma": gamma,
        "Vega": vega,
        "Theta": theta,
        "Rho": rho
    }


# In[60]:


call_greeks = option_greeks(
    S, K, T, r, sigma, "call"
)

call_greeks


# In[61]:


put_greeks = option_greeks(
    S, K, T, r, sigma, "put"
)

put_greeks


# In[62]:


greeks_table = pd.DataFrame({
    "Greek": [
        "Delta",
        "Gamma",
        "Vega",
        "Theta",
        "Rho"
    ],
    "Call": [
        call_greeks["Delta"],
        call_greeks["Gamma"],
        call_greeks["Vega"],
        call_greeks["Theta"],
        call_greeks["Rho"]
    ],
    "Put": [
        put_greeks["Delta"],
        put_greeks["Gamma"],
        put_greeks["Vega"],
        put_greeks["Theta"],
        put_greeks["Rho"]
    ]
})

greeks_table


# In[63]:


print("Delta relationship:")
print(
    "Call Delta + |Put Delta| =",
    round(
        call_greeks["Delta"] + abs(put_greeks["Delta"]),
        4
    )
)

print("\nGamma:")
print("Call Gamma:", round(call_greeks["Gamma"], 6))
print("Put Gamma :", round(put_greeks["Gamma"], 6))

print("\nVega:")
print("Call Vega:", round(call_greeks["Vega"], 2))
print("Put Vega :", round(put_greeks["Vega"], 2))


# In[64]:


nifty_shocks = np.array([-0.05, -0.02, -0.01, 0, 0.01, 0.02, 0.05])

scenario_prices = S * (1 + nifty_shocks)

scenario_prices


# In[65]:


scenario_results = []

for shock, scenario_S in zip(nifty_shocks, scenario_prices):

    scenario_price = black_scholes_price(
        scenario_S,
        K,
        T,
        r,
        sigma,
        "call"
    )

    scenario_pnl = scenario_price - call_price

    scenario_results.append({
        "NIFTY Shock": shock,
        "NIFTY Level": scenario_S,
        "Option Price": scenario_price,
        "P&L": scenario_pnl
    })

scenario_df = pd.DataFrame(scenario_results)

scenario_df


# In[66]:


call_delta = call_greeks["Delta"]

scenario_df["Delta P&L"] = (
    call_delta
    * (scenario_df["NIFTY Level"] - S)
)

scenario_df[
    [
        "NIFTY Shock",
        "NIFTY Level",
        "P&L",
        "Delta P&L"
    ]
]


# In[67]:


call_gamma = call_greeks["Gamma"]

scenario_df["Delta-Gamma P&L"] = (
    call_delta
    * (scenario_df["NIFTY Level"] - S)
    + 0.5
    * call_gamma
    * (scenario_df["NIFTY Level"] - S) ** 2
)

scenario_df[
    [
        "NIFTY Shock",
        "P&L",
        "Delta P&L",
        "Delta-Gamma P&L"
    ]
]


# In[68]:


plt.figure(figsize=(10, 5))

plt.plot(
    scenario_df["NIFTY Shock"] * 100,
    scenario_df["P&L"],
    marker="o",
    label="Actual P&L"
)

plt.plot(
    scenario_df["NIFTY Shock"] * 100,
    scenario_df["Delta P&L"],
    marker="o",
    label="Delta P&L"
)

plt.plot(
    scenario_df["NIFTY Shock"] * 100,
    scenario_df["Delta-Gamma P&L"],
    marker="o",
    label="Delta-Gamma P&L"
)

plt.axhline(0, linewidth=0.8)

plt.title("NIFTY Option P&L Under Market Shocks")
plt.xlabel("NIFTY Shock (%)")
plt.ylabel("Option P&L (₹)")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("delta_gamma_pnl.png", dpi=300, bbox_inches="tight")
plt.show()


# In[69]:


# Step 22: Volatility Shock Analysis

implied_vol = sigma
lot_size = 1
vega = call_greeks['Vega']
vol_shocks = np.array([-0.20, -0.10, -0.05, 0, 0.05, 0.10, 0.20])

volatility_scenario = []

for shock in vol_shocks:
    shocked_vol = sigma * (1 + shock)

    shocked_option_price = black_scholes_call(
        S,
        K,
        T,
        r,
        shocked_vol
    )

    actual_pnl = (shocked_option_price - call_price) * lot_size

    vega_pnl = vega * (shocked_vol - implied_vol) * lot_size

    volatility_scenario.append([
        shock,
        shocked_vol,
        shocked_option_price,
        actual_pnl,
        vega_pnl
    ])

volatility_df = pd.DataFrame(
    volatility_scenario,
    columns=[
        'Volatility Shock',
        'Shocked IV',
        'Option Price',
        'Actual P&L',
        'Vega P&L'
    ]
)

volatility_df


# In[70]:


get_ipython().run_line_magic('whos', '')


# In[71]:


call_greeks


# In[72]:


vega = call_greeks['Vega']

print("Call Vega:", vega)


# In[73]:


# Step 22: Volatility Shock Analysis

implied_vol = sigma
lot_size = 1
vega = call_greeks['Vega']
vol_shocks = np.array([-0.20, -0.10, -0.05, 0, 0.05, 0.10, 0.20])

volatility_scenario = []

for shock in vol_shocks:

    # Apply relative volatility shock
    shocked_sigma = sigma * (1 + shock)

    # Recalculate option price using shocked volatility
    shocked_option_price = black_scholes_price(
        S,
        K,
        T,
        r,
        shocked_sigma,
        option_type='call'
    )

    # Actual P&L from repricing
    actual_pnl = (shocked_option_price - call_price)

    # Vega approximation
    vega_pnl = vega * (shocked_sigma - sigma)

    volatility_scenario.append([
        shock,
        shocked_sigma,
        shocked_option_price,
        actual_pnl,
        vega_pnl
    ])

volatility_df = pd.DataFrame(
    volatility_scenario,
    columns=[
        'Volatility Shock',
        'Shocked IV',
        'Option Price',
        'Actual P&L',
        'Vega P&L'
    ]
)

volatility_df


# In[74]:


plt.figure(figsize=(10, 6))

plt.plot(
    volatility_df['Volatility Shock'] * 100,
    volatility_df['Actual P&L'],
    marker='o',
    label='Actual P&L'
)

plt.plot(
    volatility_df['Volatility Shock'] * 100,
    volatility_df['Vega P&L'],
    marker='o',
    linestyle='--',
    label='Vega P&L'
)

plt.axhline(0, linewidth=0.8)

plt.xlabel('Implied Volatility Shock (%)')
plt.ylabel('Option P&L (₹)')
plt.title('NIFTY Call Option P&L Under Volatility Shocks')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[75]:


call_greeks


# In[76]:


theta = call_greeks['Theta']

print("Call Theta:", theta)


# In[77]:


# Step 23: Theta / Time Decay Analysis

days_passed = np.array([1, 3, 5, 7, 10, 15, 20])

time_scenario = []

for days in days_passed:

    # Reduce time to expiry
    shocked_T = T - (days / 365)

    # Make sure time doesn't become negative
    shocked_T = max(shocked_T, 0.000001)

    # Recalculate option price
    shocked_option_price = black_scholes_price(
        S,
        K,
        shocked_T,
        r,
        sigma,
        option_type='call'
    )

    # Actual P&L
    actual_pnl = shocked_option_price - call_price

    # Theta approximation
    theta_pnl = theta * days

    time_scenario.append([
        days,
        shocked_T,
        shocked_option_price,
        actual_pnl,
        theta_pnl
    ])

theta_df = pd.DataFrame(
    time_scenario,
    columns=[
        'Days Passed',
        'Remaining T',
        'Option Price',
        'Actual P&L',
        'Theta P&L'
    ]
)

theta_df


# In[78]:


plt.figure(figsize=(10, 6))

plt.plot(
    theta_df['Days Passed'],
    theta_df['Actual P&L'],
    marker='o',
    label='Actual P&L'
)

plt.plot(
    theta_df['Days Passed'],
    theta_df['Theta P&L'],
    marker='o',
    linestyle='--',
    label='Theta P&L'
)

plt.axhline(0, linewidth=0.8)

plt.xlabel('Days Passed')
plt.ylabel('Option P&L (₹)')
plt.title('NIFTY Call Option Time Decay')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[79]:


# Step 24: Combined Greeks Scenario Analysis

price_shocks = np.array([-0.10, -0.05, 0, 0.05, 0.10])
vol_shock = 0.10
days_passed = 5

combined_results = []

for price_shock in price_shocks:

    # Shocked NIFTY price
    shocked_S = S * (1 + price_shock)

    # Shocked volatility
    shocked_sigma = sigma * (1 + vol_shock)

    # Shocked time to expiry
    shocked_T = max(T - (days_passed / 365), 0.000001)

    # Full Black-Scholes repricing
    shocked_option_price = black_scholes_price(
        shocked_S,
        K,
        shocked_T,
        r,
        shocked_sigma,
        option_type='call'
    )

    # Actual P&L
    actual_pnl = shocked_option_price - call_price

    # Changes in risk factors
    delta_S = shocked_S - S
    delta_sigma = shocked_sigma - sigma
    delta_t = days_passed / 365

    # Greek approximations
    delta_pnl = call_delta * delta_S
    gamma_pnl = 0.5 * call_gamma * (delta_S ** 2)
    vega_pnl = vega * delta_sigma
    theta_pnl = theta * delta_t

    # Combined approximation
    combined_greek_pnl = (
        delta_pnl
        + gamma_pnl
        + vega_pnl
        + theta_pnl
    )

    combined_results.append([
        price_shock,
        shocked_S,
        shocked_sigma,
        shocked_T,
        actual_pnl,
        delta_pnl,
        gamma_pnl,
        vega_pnl,
        theta_pnl,
        combined_greek_pnl
    ])

combined_df = pd.DataFrame(
    combined_results,
    columns=[
        'NIFTY Shock',
        'Shocked NIFTY',
        'Shocked IV',
        'Remaining T',
        'Actual P&L',
        'Delta P&L',
        'Gamma P&L',
        'Vega P&L',
        'Theta P&L',
        'Combined Greeks P&L'
    ]
)

combined_df


# In[80]:


combined_df['Approximation Error'] = (
    combined_df['Actual P&L']
    - combined_df['Combined Greeks P&L']
)

combined_df['Error %'] = (
    combined_df['Approximation Error']
    / combined_df['Actual P&L'].replace(0, np.nan)
) * 100

combined_df


# In[81]:


plt.figure(figsize=(10, 6))

plt.plot(
    combined_df['NIFTY Shock'] * 100,
    combined_df['Actual P&L'],
    marker='o',
    label='Actual Black-Scholes P&L'
)

plt.plot(
    combined_df['NIFTY Shock'] * 100,
    combined_df['Combined Greeks P&L'],
    marker='o',
    linestyle='--',
    label='Combined Greeks P&L'
)

plt.axhline(0, linewidth=0.8)

plt.xlabel('NIFTY Price Shock (%)')
plt.ylabel('Option P&L (₹)')
plt.title('Actual vs Greek-Based Option P&L')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[82]:


# Corrected Step 25: 1-Day Monte Carlo VaR

np.random.seed(42)

n_simulations = 5000
dt = 1 / 365

Z = np.random.normal(0, 1, n_simulations)

simulated_S_1d = S * np.exp(
    (r - 0.5 * sigma**2) * dt
    + sigma * np.sqrt(dt) * Z
)

print("Mean simulated NIFTY:", simulated_S_1d.mean())
print("Minimum simulated NIFTY:", simulated_S_1d.min())
print("Maximum simulated NIFTY:", simulated_S_1d.max())


# In[83]:


# Remaining time after one day

T_1d = T - dt

simulated_option_prices = np.array([
    black_scholes_price(
        simulated_S_i,
        K,
        T_1d,
        r,
        sigma,
        option_type='call'
    )
    for simulated_S_i in simulated_S_1d
])

print("Mean simulated option price:",
      simulated_option_prices.mean())

print("Current option price:",
      call_price)


# In[84]:


# 1-Day simulated option P&L

simulated_pnl_1d = simulated_option_prices - call_price

print("Mean 1-Day P&L:", simulated_pnl_1d.mean())
print("Minimum 1-Day P&L:", simulated_pnl_1d.min())
print("Maximum 1-Day P&L:", simulated_pnl_1d.max())


# In[85]:


# 1-Day Monte Carlo VaR

var_95_1d = -np.percentile(simulated_pnl_1d, 5)
var_99_1d = -np.percentile(simulated_pnl_1d, 1)

print("1-Day Monte Carlo VaR (95%): ₹", var_95_1d)
print("1-Day Monte Carlo VaR (99%): ₹", var_99_1d)


# In[86]:


# 1-Day Monte Carlo Expected Shortfall

losses_1d = -simulated_pnl_1d

es_95_1d = losses_1d[losses_1d >= var_95_1d].mean()
es_99_1d = losses_1d[losses_1d >= var_99_1d].mean()

print("1-Day Monte Carlo ES (95%): ₹", es_95_1d)
print("1-Day Monte Carlo ES (99%): ₹", es_99_1d)


# In[87]:


# Step 26: Monte Carlo Convergence Analysis

simulation_sizes = [1000, 5000, 10000, 25000, 50000]

convergence_results = []

dt = 1 / 365

for n in simulation_sizes:

    np.random.seed(42)

    Z = np.random.normal(0, 1, n)

    simulated_S = S * np.exp(
        (r - 0.5 * sigma**2) * dt
        + sigma * np.sqrt(dt) * Z
    )

    T_1d = T - dt

    simulated_prices = np.array([
        black_scholes_price(
            S_i,
            K,
            T_1d,
            r,
            sigma,
            option_type='call'
        )
        for S_i in simulated_S
    ])

    simulated_pnl = simulated_prices - call_price

    var_95 = -np.percentile(simulated_pnl, 5)
    var_99 = -np.percentile(simulated_pnl, 1)

    losses = -simulated_pnl

    es_95 = losses[losses >= var_95].mean()
    es_99 = losses[losses >= var_99].mean()

    convergence_results.append([
        n,
        simulated_prices.mean(),
        var_95,
        var_99,
        es_95,
        es_99
    ])

convergence_df = pd.DataFrame(
    convergence_results,
    columns=[
        'Simulations',
        'MC Option Price',
        'VaR 95%',
        'VaR 99%',
        'ES 95%',
        'ES 99%'
    ])

convergence_df


# In[88]:


# Step 27: 1-Day Monte Carlo P&L Distribution

plt.figure(figsize=(10, 6))

plt.hist(
    simulated_pnl_1d,
    bins=50,
    edgecolor='black'
)

plt.axvline(
    -var_95_1d,
    linestyle='--',
    linewidth=2,
    label='95% VaR'
)

plt.axvline(
    -var_99_1d,
    linestyle='--',
    linewidth=2,
    label='99% VaR'
)

plt.axvline(
    -es_95_1d,
    linestyle=':',
    linewidth=2,
    label='95% ES'
)

plt.axvline(
    -es_99_1d,
    linestyle=':',
    linewidth=2,
    label='99% ES'
)

plt.axvline(
    0,
    linewidth=1
)

plt.xlabel('1-Day Option P&L (₹)')
plt.ylabel('Frequency')
plt.title('Monte Carlo Distribution of 1-Day NIFTY Call Option P&L')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[89]:


# Step 28: VaR vs Expected Shortfall

risk_summary = pd.DataFrame({
    'Confidence Level': ['95%', '99%'],
    'VaR (₹)': [var_95_1d, var_99_1d],
    'Expected Shortfall (₹)': [es_95_1d, es_99_1d]
})

risk_summary


# In[90]:


risk_summary['ES / VaR'] = (
    risk_summary['Expected Shortfall (₹)']
    / risk_summary['VaR (₹)']
)

risk_summary


# In[91]:


# Step 29: NIFTY Stress Testing

stress_shocks = np.array([-0.05, -0.10, -0.15, -0.20])

stress_results = []

for shock in stress_shocks:

    stressed_S = S * (1 + shock)

    stressed_price = black_scholes_price(
        stressed_S,
        K,
        T,
        r,
        sigma,
        option_type='call'
    )

    stressed_pnl = stressed_price - call_price

    stress_results.append([
        shock,
        stressed_S,
        stressed_price,
        stressed_pnl
    ])

stress_df = pd.DataFrame(
    stress_results,
    columns=[
        'NIFTY Shock',
        'Stressed NIFTY',
        'Option Price',
        'Option P&L'
    ]
)

stress_df


# In[92]:


plt.figure(figsize=(10, 6))

plt.plot(
    stress_df['NIFTY Shock'] * 100,
    stress_df['Option P&L'],
    marker='o'
)

plt.axhline(0, linewidth=0.8)

plt.xlabel('NIFTY Stress Shock (%)')
plt.ylabel('Option P&L (₹)')
plt.title('NIFTY Call Option Stress Testing')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[93]:


# Step 31: Consolidated Risk Summary

risk_dashboard = pd.DataFrame({
    'Risk Measure': [
        'Call Option Price',
        'Delta',
        'Gamma',
        'Vega',
        'Theta',
        '1-Day VaR (95%)',
        '1-Day VaR (99%)',
        '1-Day ES (95%)',
        '1-Day ES (99%)'
    ],

    'Value': [
        call_price,
        call_delta,
        call_gamma,
        vega,
        theta,
        var_95_1d,
        var_99_1d,
        es_95_1d,
        es_99_1d
    ]
})

risk_dashboard


# In[94]:


# Step 32: Risk as % of Option Value

risk_dashboard['% of Option Value'] = np.nan

risk_dashboard.loc[
    risk_dashboard['Risk Measure'] == '1-Day VaR (95%)',
    '% of Option Value'
] = (var_95_1d / call_price) * 100

risk_dashboard.loc[
    risk_dashboard['Risk Measure'] == '1-Day VaR (99%)',
    '% of Option Value'
] = (var_99_1d / call_price) * 100

risk_dashboard.loc[
    risk_dashboard['Risk Measure'] == '1-Day ES (95%)',
    '% of Option Value'
] = (es_95_1d / call_price) * 100

risk_dashboard.loc[
    risk_dashboard['Risk Measure'] == '1-Day ES (99%)',
    '% of Option Value'
] = (es_99_1d / call_price) * 100

risk_dashboard


# In[95]:


# Step 33: Call vs Put Greeks

greeks_comparison = pd.DataFrame({
    'Greek': ['Delta', 'Gamma', 'Vega', 'Theta', 'Rho'],

    'Call': [
        call_greeks['Delta'],
        call_greeks['Gamma'],
        call_greeks['Vega'],
        call_greeks['Theta'],
        call_greeks['Rho']
    ],

    'Put': [
        put_greeks['Delta'],
        put_greeks['Gamma'],
        put_greeks['Vega'],
        put_greeks['Theta'],
        put_greeks['Rho']
    ]
})

greeks_comparison


# In[96]:


print(
    "Delta Relationship:",
    call_greeks['Delta'] + abs(put_greeks['Delta'])
)

print(
    "Gamma Difference:",
    call_greeks['Gamma'] - put_greeks['Gamma']
)

print(
    "Vega Difference:",
    call_greeks['Vega'] - put_greeks['Vega']
)


# In[97]:


# Step 34: Put-Call Parity Validation

parity_difference = parity_left - parity_right

print("Put-Call Parity Difference:", parity_difference)


# In[98]:


# Step 35: Final Analytical Summary

final_summary = pd.DataFrame({
    'Metric': [
        'NIFTY Spot',
        'Strike Price',
        'Time to Expiry (Years)',
        'Risk-Free Rate',
        'Historical Volatility',
        'Call Price',
        'Put Price',
        'Call Delta',
        'Call Gamma',
        'Call Vega',
        'Call Theta',
        '1-Day VaR 95%',
        '1-Day VaR 99%',
        '1-Day ES 95%',
        '1-Day ES 99%'
    ],

    'Value': [
        S,
        K,
        T,
        r,
        sigma,
        call_price,
        put_price,
        call_greeks['Delta'],
        call_greeks['Gamma'],
        call_greeks['Vega'],
        call_greeks['Theta'],
        var_95_1d,
        var_99_1d,
        es_95_1d,
        es_99_1d
    ]
})

final_summary


# In[99]:


# Step 36.1: Option Moneyness

moneyness = S / K

percentage_moneyness = ((S - K) / K) * 100

print("Moneyness:", moneyness)
print("Distance from Strike:", percentage_moneyness, "%")


# In[100]:


# Step 36.2: Intrinsic and Time Value

intrinsic_value = max(S - K, 0)

time_value = call_price - intrinsic_value

print("Intrinsic Value: ₹", intrinsic_value)
print("Time Value: ₹", time_value)


# In[101]:


time_value_percentage = (time_value / call_price) * 100

print("Time Value as % of Call Price:",
      time_value_percentage, "%")


# In[102]:


print("Annualized Theta:", theta)
print("Approximate Daily Theta:", theta / 365)


# In[103]:


# Step 42: Covered Call Payoff

expiry_prices = np.linspace(
    S * 0.80,
    S * 1.20,
    100
)

covered_call_pnl = (
    expiry_prices
    - S
    - np.maximum(expiry_prices - K, 0)
    + call_price
)

plt.figure(figsize=(10, 6))

plt.plot(
    expiry_prices,
    covered_call_pnl,
    linewidth=2,
    label='Covered Call P&L'
)

plt.axhline(0, linewidth=1)

plt.axvline(
    K,
    linestyle='--',
    linewidth=1,
    label='Strike Price'
)

plt.xlabel('NIFTY Price at Expiry')
plt.ylabel('Strategy P&L (₹)')
plt.title('NIFTY Covered Call Payoff')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[104]:


# Step 43: Covered Call Break-even

covered_call_breakeven = S - call_price

print("Covered Call Break-even: ₹",
      covered_call_breakeven)


# In[105]:


# Step 44: Maximum Covered Call Profit

max_profit = K - S + call_price

print("Maximum Covered Call Profit: ₹",
      max_profit)


# In[106]:


# Step 45: Long NIFTY vs Covered Call

expiry_prices = np.linspace(
    S * 0.80,
    S * 1.20,
    100
)

# Long NIFTY P&L
long_nifty_pnl = expiry_prices - S

# Short Call P&L
short_call_pnl = (
    call_price
    - np.maximum(expiry_prices - K, 0)
)

# Covered Call P&L
covered_call_pnl = (
    long_nifty_pnl
    + short_call_pnl
)

comparison_df = pd.DataFrame({
    'NIFTY at Expiry': expiry_prices,
    'Long NIFTY P&L': long_nifty_pnl,
    'Short Call P&L': short_call_pnl,
    'Covered Call P&L': covered_call_pnl
})

comparison_df.head()


# In[107]:


plt.figure(figsize=(10, 6))

plt.plot(
    expiry_prices,
    long_nifty_pnl,
    linewidth=2,
    label='Long NIFTY'
)

plt.plot(
    expiry_prices,
    covered_call_pnl,
    linewidth=2,
    label='Covered Call'
)

plt.axhline(0, linewidth=1)

plt.axvline(
    K,
    linestyle='--',
    linewidth=1,
    label='Strike Price'
)

plt.xlabel('NIFTY Price at Expiry')
plt.ylabel('P&L (₹)')
plt.title('Long NIFTY vs Covered Call')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[108]:


# Step 46: Maximum Profit Comparison

long_nifty_profit_at_strike = K - S

print("Long NIFTY P&L at Strike: ₹",
      long_nifty_profit_at_strike)

print("Covered Call Maximum P&L: ₹",
      max_profit)


# In[109]:


# Step 47: Strategy Returns

comparison_df['Long NIFTY Return (%)'] = (
    comparison_df['Long NIFTY P&L'] / S
) * 100

comparison_df['Covered Call Return (%)'] = (
    comparison_df['Covered Call P&L'] / S
) * 100

comparison_df.head()


# In[110]:


max_cc_return = (max_profit / S) * 100

print("Maximum Covered Call Return:",
      max_cc_return, "%")


# In[111]:


# Step 50: Covered Call vs Long NIFTY Crossover

crossover_price = K + call_price

print("Crossover NIFTY Price: ₹",
      crossover_price)


# In[112]:


# Step 51: Covered Call Scenario Table

scenario_prices = np.array([
    20000,
    22000,
    24000,
    24500,
    25000,
    26000,
    28000,
    30000
])

long_pnl = scenario_prices - S

covered_call_pnl_scenario = (
    scenario_prices
    - S
    - np.maximum(scenario_prices - K, 0)
    + call_price
)

strategy_table = pd.DataFrame({
    'NIFTY at Expiry': scenario_prices,
    'Long NIFTY P&L': long_pnl,
    'Covered Call P&L': covered_call_pnl_scenario,
    'Covered Call Advantage': covered_call_pnl_scenario - long_pnl
})

strategy_table


# In[113]:


# Step 52.1: Inspect Historical NIFTY Data

print(data.columns)
print(data.head())
print(data.tail())


# In[114]:


print(data.shape)
print(data.index.min())
print(data.index.max())


# In[115]:


# Step 53.1: Prepare NIFTY Close Prices

nifty_close = data[('Close', '^NSEI')].copy()

nifty_close.name = 'NIFTY'

print(nifty_close.head())
print(nifty_close.tail())
print("Number of observations:", len(nifty_close))


# In[116]:


# Step 53.2: Calculate Rolling Historical Volatility

daily_returns = nifty_close.pct_change()

rolling_volatility = (
    daily_returns
    .rolling(window=21)
    .std()
    * np.sqrt(252)
)

print(rolling_volatility.tail())


# In[117]:


# Step 53.3: Backtest Parameters

backtest_days = 21
option_days = 21

risk_free_rate = r

strike_interval = 50

print("Backtest cycle:", backtest_days, "trading days")
print("Option maturity:", option_days, "trading days")
print("Risk-free rate:", risk_free_rate)
print("Strike interval:", strike_interval)


# In[118]:


# Step 54: Historical Covered Call Backtest

backtest_results = []

for i in range(21, len(nifty_close) - backtest_days, backtest_days):

    entry_date = nifty_close.index[i]
    expiry_date = nifty_close.index[i + backtest_days]

    S0 = nifty_close.iloc[i]
    ST = nifty_close.iloc[i + backtest_days]

    # Historical volatility estimated from previous 21 trading days
    historical_vol = rolling_volatility.iloc[i]

    # Skip if volatility is unavailable
    if np.isnan(historical_vol):
        continue

    # ATM strike rounded to nearest ₹50
    K_bt = round(S0 / strike_interval) * strike_interval

    # Approximate option maturity in years
    T_bt = option_days / 365

    # Estimate call premium using Black-Scholes
    call_premium_bt = black_scholes_price(
        S0,
        K_bt,
        T_bt,
        risk_free_rate,
        historical_vol,
        option_type='call'
    )

    # Long NIFTY P&L
    long_nifty_pnl = ST - S0

    # Short call P&L
    short_call_pnl = (
        call_premium_bt
        - max(ST - K_bt, 0)
    )

    # Covered call P&L
    covered_call_pnl = (
        long_nifty_pnl
        + short_call_pnl
    )

    # Returns
    long_nifty_return = long_nifty_pnl / S0

    covered_call_return = covered_call_pnl / S0

    backtest_results.append([
        entry_date,
        expiry_date,
        S0,
        ST,
        K_bt,
        historical_vol,
        call_premium_bt,
        long_nifty_pnl,
        short_call_pnl,
        covered_call_pnl,
        long_nifty_return,
        covered_call_return
    ])

backtest_df = pd.DataFrame(
    backtest_results,
    columns=[
        'Entry Date',
        'Expiry Date',
        'Entry NIFTY',
        'Expiry NIFTY',
        'Strike',
        'Historical Volatility',
        'Call Premium',
        'Long NIFTY P&L',
        'Short Call P&L',
        'Covered Call P&L',
        'Long NIFTY Return',
        'Covered Call Return'
    ]
)

backtest_df


# In[119]:


# Step 55: Backtest Summary

print("Number of option cycles:", len(backtest_df))

print(
    "Average Long NIFTY Return:",
    backtest_df['Long NIFTY Return'].mean() * 100,
    "%"
)

print(
    "Average Covered Call Return:",
    backtest_df['Covered Call Return'].mean() * 100,
    "%"
)

print(
    "Covered Call Win Rate:",
    (backtest_df['Covered Call Return'] > 0).mean() * 100,
    "%"
)

print(
    "Long NIFTY Win Rate:",
    (backtest_df['Long NIFTY Return'] > 0).mean() * 100,
    "%"
)


# In[120]:


# Step 56: Cumulative Strategy Performance

backtest_df['Long NIFTY Wealth'] = (
    1 + backtest_df['Long NIFTY Return']
).cumprod()

backtest_df['Covered Call Wealth'] = (
    1 + backtest_df['Covered Call Return']
).cumprod()

print(
    "Final Long NIFTY Wealth:",
    backtest_df['Long NIFTY Wealth'].iloc[-1]
)

print(
    "Final Covered Call Wealth:",
    backtest_df['Covered Call Wealth'].iloc[-1]
)


# In[121]:


plt.figure(figsize=(10, 6))

plt.plot(
    backtest_df['Expiry Date'],
    backtest_df['Long NIFTY Wealth'],
    linewidth=2,
    label='Long NIFTY'
)

plt.plot(
    backtest_df['Expiry Date'],
    backtest_df['Covered Call Wealth'],
    linewidth=2,
    label='Covered Call'
)

plt.xlabel('Date')
plt.ylabel('Growth of ₹1')
plt.title('Historical Performance: Long NIFTY vs Covered Call')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[122]:


# Step 57: Maximum Drawdown

def calculate_max_drawdown(wealth_series):

    running_max = wealth_series.cummax()

    drawdown = (
        wealth_series / running_max
    ) - 1

    return drawdown.min()

long_nifty_mdd = calculate_max_drawdown(
    backtest_df['Long NIFTY Wealth']
)

covered_call_mdd = calculate_max_drawdown(
    backtest_df['Covered Call Wealth']
)

print(
    "Long NIFTY Maximum Drawdown:",
    long_nifty_mdd * 100,
    "%"
)

print(
    "Covered Call Maximum Drawdown:",
    covered_call_mdd * 100,
    "%"
)


# In[123]:


# Step 58: Sharpe Ratio

annual_rf = risk_free_rate

cycle_rf = (
    (1 + annual_rf) ** (1 / 12)
) - 1

backtest_df['Long NIFTY Excess Return'] = (
    backtest_df['Long NIFTY Return'] - cycle_rf
)

backtest_df['Covered Call Excess Return'] = (
    backtest_df['Covered Call Return'] - cycle_rf
)

long_nifty_sharpe = (
    backtest_df['Long NIFTY Excess Return'].mean()
    / backtest_df['Long NIFTY Return'].std()
) * np.sqrt(12)

covered_call_sharpe = (
    backtest_df['Covered Call Excess Return'].mean()
    / backtest_df['Covered Call Return'].std()
) * np.sqrt(12)

print(
    "Long NIFTY Sharpe Ratio:",
    long_nifty_sharpe
)

print(
    "Covered Call Sharpe Ratio:",
    covered_call_sharpe
)


# In[124]:


# Step 59: Comprehensive Backtest Summary

summary = pd.DataFrame({
    'Metric': [
        'Number of Cycles',
        'Average Return',
        'Return Volatility',
        'Win Rate',
        'Best Cycle',
        'Worst Cycle',
        'Maximum Drawdown',
        'Sharpe Ratio'
    ],

    'Long NIFTY': [
        len(backtest_df),
        backtest_df['Long NIFTY Return'].mean(),
        backtest_df['Long NIFTY Return'].std(),
        (backtest_df['Long NIFTY Return'] > 0).mean(),
        backtest_df['Long NIFTY Return'].max(),
        backtest_df['Long NIFTY Return'].min(),
        long_nifty_mdd,
        long_nifty_sharpe
    ],

    'Covered Call': [
        len(backtest_df),
        backtest_df['Covered Call Return'].mean(),
        backtest_df['Covered Call Return'].std(),
        (backtest_df['Covered Call Return'] > 0).mean(),
        backtest_df['Covered Call Return'].max(),
        backtest_df['Covered Call Return'].min(),
        covered_call_mdd,
        covered_call_sharpe
    ]
})

summary


# In[125]:


# Step 60: Format Performance Metrics

summary_formatted = summary.copy()

percentage_rows = [
    'Average Return',
    'Return Volatility',
    'Win Rate',
    'Best Cycle',
    'Worst Cycle',
    'Maximum Drawdown'
]

for row in percentage_rows:
    summary_formatted.loc[
        summary_formatted['Metric'] == row,
        ['Long NIFTY', 'Covered Call']
    ] *= 100

summary_formatted


# In[126]:


# Step 61: Premium Analysis

total_premium = backtest_df['Call Premium'].sum()

average_premium = backtest_df['Call Premium'].mean()

print("Total estimated premiums received: ₹", total_premium)
print("Average premium per cycle: ₹", average_premium)


# In[127]:


# Premium yield per cycle

backtest_df['Premium Yield'] = (
    backtest_df['Call Premium']
    / backtest_df['Entry NIFTY']
)

print(
    "Average Premium Yield:",
    backtest_df['Premium Yield'].mean() * 100,
    "%"
)


# In[128]:


# Step 62: Outperformance Analysis

backtest_df['Covered Call Outperformed'] = (
    backtest_df['Covered Call Return']
    > backtest_df['Long NIFTY Return']
)

outperformance_count = (
    backtest_df['Covered Call Outperformed'].sum()
)

outperformance_percentage = (
    backtest_df['Covered Call Outperformed'].mean()
    * 100
)

print(
    "Cycles where Covered Call outperformed:",
    outperformance_count
)

print(
    "Percentage of cycles:",
    outperformance_percentage,
    "%"
)


# In[129]:


# Step 63: Market Regime Classification

def classify_market(return_value):

    if return_value > 0.02:
        return 'Bull'
    elif return_value < -0.02:
        return 'Bear'
    else:
        return 'Neutral'


backtest_df['Market Regime'] = (
    backtest_df['Long NIFTY Return']
    .apply(classify_market)
)

backtest_df['Market Regime'].value_counts()


# In[130]:


# Step 64: Performance by Market Regime

regime_analysis = backtest_df.groupby('Market Regime').agg({
    'Long NIFTY Return': ['mean', 'std'],
    'Covered Call Return': ['mean', 'std'],
    'Covered Call Outperformed': 'mean'
})

regime_analysis


# In[131]:


# Step 65: Market Regime Performance

regime_returns = backtest_df.groupby('Market Regime')[
    ['Long NIFTY Return', 'Covered Call Return']
].mean() * 100

regime_returns.plot(
    kind='bar',
    figsize=(10, 6)
)

plt.xlabel('Market Regime')
plt.ylabel('Average Return (%)')
plt.title('Covered Call vs Long NIFTY Across Market Regimes')
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.tight_layout()
plt.show()


# In[132]:


# Step 67: Underlying Return vs Covered Call Advantage

backtest_df['Covered Call Advantage'] = (
    backtest_df['Covered Call Return']
    - backtest_df['Long NIFTY Return']
)

correlation = (
    backtest_df['Long NIFTY Return']
    .corr(backtest_df['Covered Call Advantage'])
)

print(
    "Correlation between NIFTY return and Covered Call advantage:",
    correlation
)


# In[133]:


# Step 68: Final Project Results Dashboard

final_results = pd.DataFrame({
    'Category': [
        'Option Pricing',
        'Option Pricing',
        'Option Pricing',
        'Market Sensitivity',
        'Market Sensitivity',
        'Market Sensitivity',
        'Market Sensitivity',
        'Risk',
        'Risk',
        'Risk',
        'Risk',
        'Strategy',
        'Strategy',
        'Strategy'
    ],

    'Metric': [
        'Call Price',
        'Put Price',
        'Historical Volatility',
        'Delta',
        'Gamma',
        'Vega',
        'Theta',
        '1-Day VaR 95%',
        '1-Day VaR 99%',
        '1-Day ES 95%',
        '1-Day ES 99%',
        'Covered Call Break-even',
        'Maximum Covered Call Profit',
        'Maximum Covered Call Return'
    ],

    'Value': [
        call_price,
        put_price,
        sigma,
        call_greeks['Delta'],
        call_greeks['Gamma'],
        call_greeks['Vega'],
        call_greeks['Theta'],
        var_95_1d,
        var_99_1d,
        es_95_1d,
        es_99_1d,
        covered_call_breakeven,
        max_profit,
        max_cc_return
    ]
})

final_results


# In[134]:


# Step 69: Strategy Comparison

strategy_comparison = pd.DataFrame({
    'Metric': [
        'Average Return',
        'Return Volatility',
        'Win Rate',
        'Maximum Drawdown',
        'Sharpe Ratio',
        'Final Wealth',
        'Outperformance Frequency'
    ],

    'Long NIFTY': [
        backtest_df['Long NIFTY Return'].mean(),
        backtest_df['Long NIFTY Return'].std(),
        (backtest_df['Long NIFTY Return'] > 0).mean(),
        long_nifty_mdd,
        long_nifty_sharpe,
        backtest_df['Long NIFTY Wealth'].iloc[-1],
        np.nan
    ],

    'Covered Call': [
        backtest_df['Covered Call Return'].mean(),
        backtest_df['Covered Call Return'].std(),
        (backtest_df['Covered Call Return'] > 0).mean(),
        covered_call_mdd,
        covered_call_sharpe,
        backtest_df['Covered Call Wealth'].iloc[-1],
        outperformance_percentage / 100
    ]
})

strategy_comparison


# In[135]:


# Step 70: Drawdown Analysis

long_running_max = (
    backtest_df['Long NIFTY Wealth'].cummax()
)

cc_running_max = (
    backtest_df['Covered Call Wealth'].cummax()
)

long_drawdown = (
    backtest_df['Long NIFTY Wealth']
    / long_running_max
) - 1

cc_drawdown = (
    backtest_df['Covered Call Wealth']
    / cc_running_max
) - 1

plt.figure(figsize=(10, 6))

plt.plot(
    backtest_df['Expiry Date'],
    long_drawdown * 100,
    linewidth=2,
    label='Long NIFTY'
)

plt.plot(
    backtest_df['Expiry Date'],
    cc_drawdown * 100,
    linewidth=2,
    label='Covered Call'
)

plt.axhline(0, linewidth=1)

plt.xlabel('Date')
plt.ylabel('Drawdown (%)')
plt.title('Historical Drawdown: Long NIFTY vs Covered Call')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[136]:


# Step 71: Annualized Risk and Return

long_annual_return = (
    (1 + backtest_df['Long NIFTY Return'].mean()) ** 12
) - 1

cc_annual_return = (
    (1 + backtest_df['Covered Call Return'].mean()) ** 12
) - 1

long_annual_vol = (
    backtest_df['Long NIFTY Return'].std()
    * np.sqrt(12)
)

cc_annual_vol = (
    backtest_df['Covered Call Return'].std()
    * np.sqrt(12)
)

print("Long NIFTY annualized return:",
      long_annual_return * 100, "%")

print("Covered Call annualized return:",
      cc_annual_return * 100, "%")

print("Long NIFTY annualized volatility:",
      long_annual_vol * 100, "%")

print("Covered Call annualized volatility:",
      cc_annual_vol * 100, "%")


# In[137]:


# Risk-Return Plot

plt.figure(figsize=(8, 6))

plt.scatter(
    long_annual_vol * 100,
    long_annual_return * 100,
    s=100,
    label='Long NIFTY'
)

plt.scatter(
    cc_annual_vol * 100,
    cc_annual_return * 100,
    s=100,
    label='Covered Call'
)

plt.xlabel('Annualized Volatility (%)')
plt.ylabel('Annualized Return (%)')
plt.title('Risk-Return Comparison')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[138]:


# Step 72: Final Model Validation

print("===== MODEL VALIDATION =====")

print(
    "Put-Call Parity Difference:",
    parity_difference
)

print(
    "Delta Relationship:",
    call_greeks['Delta'] + abs(put_greeks['Delta'])
)

print(
    "Gamma Difference:",
    call_greeks['Gamma'] - put_greeks['Gamma']
)

print(
    "Vega Difference:",
    call_greeks['Vega'] - put_greeks['Vega']
)

print(
    "VaR 99% > VaR 95%:",
    var_99_1d > var_95_1d
)

print(
    "ES 95% > VaR 95%:",
    es_95_1d > var_95_1d
)

print(
    "ES 99% > VaR 99%:",
    es_99_1d > var_99_1d
)


# In[139]:


# Step 75: Extract Key Findings

print("===== OPTION PRICING =====")
print(f"NIFTY Spot: ₹{S:,.2f}")
print(f"Strike Price: ₹{K:,.2f}")
print(f"Call Price: ₹{call_price:,.2f}")
print(f"Put Price: ₹{put_price:,.2f}")
print(f"Historical Volatility: {sigma*100:.2f}%")

print("\n===== GREEKS =====")
print(f"Delta: {call_greeks['Delta']:.6f}")
print(f"Gamma: {call_greeks['Gamma']:.8f}")
print(f"Vega: {call_greeks['Vega']:.2f}")
print(f"Theta: {call_greeks['Theta']:.2f}")
print(f"Rho: {call_greeks['Rho']:.2f}")

print("\n===== MONTE CARLO RISK =====")
print(f"VaR 95%: ₹{var_95_1d:,.2f}")
print(f"VaR 99%: ₹{var_99_1d:,.2f}")
print(f"ES 95%: ₹{es_95_1d:,.2f}")
print(f"ES 99%: ₹{es_99_1d:,.2f}")

print("\n===== COVERED CALL =====")
print(f"Break-even: ₹{covered_call_breakeven:,.2f}")
print(f"Maximum Profit: ₹{max_profit:,.2f}")
print(f"Maximum Return: {max_cc_return:.2f}%")

print("\n===== MODEL VALIDATION =====")
print(f"Put-Call Parity Difference: {parity_difference:.10f}")
print(f"Delta Relationship: "
      f"{call_greeks['Delta'] + abs(put_greeks['Delta']):.6f}")


# In[140]:


# Step 76.1: Option Time Value

intrinsic_value = max(S - K, 0)
time_value = call_price - intrinsic_value
time_value_pct = time_value / call_price * 100

print(f"Intrinsic Value: ₹{intrinsic_value:,.2f}")
print(f"Time Value: ₹{time_value:,.2f}")
print(f"Time Value % of Premium: {time_value_pct:.2f}%")


# In[141]:


# Step 76.2: VaR Relative to Option Premium

print(
    "95% VaR as % of option value:",
    var_95_1d / call_price * 100,
    "%"
)

print(
    "99% VaR as % of option value:",
    var_99_1d / call_price * 100,
    "%"
)

print(
    "95% ES as % of option value:",
    es_95_1d / call_price * 100,
    "%"
)

print(
    "99% ES as % of option value:",
    es_99_1d / call_price * 100,
    "%"
)


# In[142]:


# Step 83: Final Backtest Numbers

print("===== FINAL BACKTEST NUMBERS =====")

print("Cycles:", len(backtest_df))

print(
    "Avg NIFTY Return:",
    backtest_df['Long NIFTY Return'].mean() * 100
)

print(
    "Avg Covered Call Return:",
    backtest_df['Covered Call Return'].mean() * 100
)

print(
    "NIFTY Volatility:",
    backtest_df['Long NIFTY Return'].std() * 100
)

print(
    "Covered Call Volatility:",
    backtest_df['Covered Call Return'].std() * 100
)

print(
    "NIFTY Max Drawdown:",
    long_nifty_mdd * 100
)

print(
    "Covered Call Max Drawdown:",
    covered_call_mdd * 100
)

print(
    "NIFTY Sharpe:",
    long_nifty_sharpe
)

print(
    "Covered Call Sharpe:",
    covered_call_sharpe
)

print(
    "Covered Call Outperformance:",
    outperformance_percentage
)

print(
    "Final NIFTY Wealth:",
    backtest_df['Long NIFTY Wealth'].iloc[-1]
)

print(
    "Final Covered Call Wealth:",
    backtest_df['Covered Call Wealth'].iloc[-1]
)


# In[143]:


# Step 89: NIFTY Descriptive Statistics

nifty_returns = nifty_close.pct_change().dropna()

starting_nifty = nifty_close.iloc[0]
ending_nifty = nifty_close.iloc[-1]

overall_return = (ending_nifty / starting_nifty) - 1

mean_daily_return = nifty_returns.mean()
daily_volatility = nifty_returns.std()
annualized_volatility = daily_volatility * np.sqrt(252)

minimum_daily_return = nifty_returns.min()
maximum_daily_return = nifty_returns.max()

nifty_running_max = nifty_close.cummax()
nifty_drawdown = (nifty_close / nifty_running_max) - 1
nifty_max_drawdown = nifty_drawdown.min()

print("===== NIFTY DESCRIPTIVE STATISTICS =====")

print(f"Starting NIFTY: ₹{starting_nifty:,.2f}")
print(f"Ending NIFTY: ₹{ending_nifty:,.2f}")
print(f"Overall Return: {overall_return * 100:.2f}%")

print(f"Mean Daily Return: {mean_daily_return * 100:.4f}%")
print(f"Daily Volatility: {daily_volatility * 100:.4f}%")
print(f"Annualized Volatility: {annualized_volatility * 100:.2f}%")

print(f"Minimum Daily Return: {minimum_daily_return * 100:.2f}%")
print(f"Maximum Daily Return: {maximum_daily_return * 100:.2f}%")

print(f"Maximum Drawdown: {nifty_max_drawdown * 100:.2f}%")


# In[144]:


# Step 90: Return Distribution

return_skewness = nifty_returns.skew()
return_kurtosis = nifty_returns.kurt()

print("===== RETURN DISTRIBUTION =====")

print(f"Skewness: {return_skewness:.4f}")
print(f"Excess Kurtosis: {return_kurtosis:.4f}")


# In[145]:


# Step 91: NIFTY Historical Price

plt.figure(figsize=(11, 6))

plt.plot(
    nifty_close.index,
    nifty_close.values,
    linewidth=2
)

plt.xlabel('Date')
plt.ylabel('NIFTY Level')
plt.title('NIFTY 50 Historical Price Movement')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[146]:


# Step 92: Rolling Volatility

plt.figure(figsize=(11, 6))

plt.plot(
    rolling_volatility.index,
    rolling_volatility * 100,
    linewidth=2
)

plt.xlabel('Date')
plt.ylabel('21-Day Rolling Volatility (%)')
plt.title('NIFTY 21-Day Rolling Historical Volatility')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[147]:


# Step 93: Descriptive Statistics Table

descriptive_stats = pd.DataFrame({
    'Statistic': [
        'Starting NIFTY',
        'Ending NIFTY',
        'Overall Return',
        'Mean Daily Return',
        'Daily Volatility',
        'Annualized Volatility',
        'Minimum Daily Return',
        'Maximum Daily Return',
        'Maximum Drawdown',
        'Skewness',
        'Excess Kurtosis'
    ],

    'Value': [
        starting_nifty,
        ending_nifty,
        overall_return * 100,
        mean_daily_return * 100,
        daily_volatility * 100,
        annualized_volatility * 100,
        minimum_daily_return * 100,
        maximum_daily_return * 100,
        nifty_max_drawdown * 100,
        return_skewness,
        return_kurtosis
    ]
})

descriptive_stats


# In[148]:


# Step 94: Black-Scholes Valuation Summary

intrinsic_call = max(S - K, 0)
time_value_call = call_price - intrinsic_call

intrinsic_put = max(K - S, 0)
time_value_put = put_price - intrinsic_put

bs_summary = pd.DataFrame({
    'Parameter': [
        'NIFTY Spot Price',
        'Strike Price',
        'Time to Expiry (Years)',
        'Risk-Free Rate',
        'Historical Volatility',
        'Call Price',
        'Call Intrinsic Value',
        'Call Time Value',
        'Put Price',
        'Put Intrinsic Value',
        'Put Time Value'
    ],

    'Value': [
        S,
        K,
        T,
        r,
        sigma,
        call_price,
        intrinsic_call,
        time_value_call,
        put_price,
        intrinsic_put,
        time_value_put
    ]
})

bs_summary


# In[149]:


# Step 95: Put-Call Parity

parity_lhs = call_price - put_price

parity_rhs = S - K * np.exp(-r * T)

parity_difference = parity_lhs - parity_rhs

print("===== PUT-CALL PARITY =====")

print(f"C - P: ₹{parity_lhs:,.6f}")

print(
    f"S - K*e^(-rT): ₹{parity_rhs:,.6f}"
)

print(
    f"Difference: ₹{parity_difference:,.10f}"
)


# In[150]:


# Step 96: Option Moneyness

moneyness = S / K

percentage_difference = (
    (S - K) / K
) * 100

print(f"S/K Moneyness Ratio: {moneyness:.6f}")
print(
    f"Spot vs Strike Difference: "
    f"{percentage_difference:.2f}%"
)


# In[151]:


# Step 97: Premium Composition

call_intrinsic_percentage = (
    intrinsic_call / call_price
) * 100

call_time_value_percentage = (
    time_value_call / call_price
) * 100

print(
    f"Call Intrinsic Value: ₹{intrinsic_call:,.2f}"
)

print(
    f"Call Time Value: ₹{time_value_call:,.2f}"
)

print(
    f"Intrinsic Value %: {call_intrinsic_percentage:.2f}%"
)

print(
    f"Time Value %: {call_time_value_percentage:.2f}%"
)


# In[152]:


# Step 98: Final Option Valuation Table

option_valuation_table = pd.DataFrame({
    'Metric': [
        'NIFTY Spot',
        'Strike',
        'Time to Expiry',
        'Risk-Free Rate',
        'Volatility',
        'Call Premium',
        'Put Premium',
        'Call Intrinsic Value',
        'Call Time Value',
        'Put-Call Parity Difference'
    ],

    'Value': [
        f"₹{S:,.2f}",
        f"₹{K:,.2f}",
        f"{T:.4f} years",
        f"{r*100:.2f}%",
        f"{sigma*100:.2f}%",
        f"₹{call_price:,.2f}",
        f"₹{put_price:,.2f}",
        f"₹{intrinsic_call:,.2f}",
        f"₹{time_value_call:,.2f}",
        f"₹{parity_difference:.10f}"
    ]
})

option_valuation_table


# In[153]:


# Step 101: Downside Buffer

downside_buffer_pct = (
    call_price / S
) * 100

print(
    f"Downside buffer from premium: "
    f"{downside_buffer_pct:.2f}%"
)


# In[154]:


# Step 102: Greeks Summary

greeks_summary = pd.DataFrame({
    'Greek': [
        'Delta',
        'Gamma',
        'Vega',
        'Theta',
        'Rho'
    ],

    'Call': [
        call_greeks['Delta'],
        call_greeks['Gamma'],
        call_greeks['Vega'],
        call_greeks['Theta'],
        call_greeks['Rho']
    ],

    'Put': [
        put_greeks['Delta'],
        put_greeks['Gamma'],
        put_greeks['Vega'],
        put_greeks['Theta'],
        put_greeks['Rho']
    ]
})

greeks_summary


# In[155]:


# Step 103: Greek Relationships

delta_sum = (
    call_greeks['Delta']
    + abs(put_greeks['Delta'])
)

gamma_difference = (
    call_greeks['Gamma']
    - put_greeks['Gamma']
)

vega_difference = (
    call_greeks['Vega']
    - put_greeks['Vega']
)

print("===== GREEK VALIDATION =====")

print(
    f"Call Delta + |Put Delta| = {delta_sum:.6f}"
)

print(
    f"Call Gamma - Put Gamma = {gamma_difference:.10f}"
)

print(
    f"Call Vega - Put Vega = {vega_difference:.6f}"
)


# In[156]:


# Step 105: Covered Call Delta

covered_call_delta = (
    1 - call_greeks['Delta']
)

print(
    f"Long NIFTY Delta: +1.0000"
)

print(
    f"Short Call Delta: {-call_greeks['Delta']:.4f}"
)

print(
    f"Covered Call Delta: {covered_call_delta:.4f}"
)


# In[157]:


# Step 106: Covered Call Gamma

covered_call_gamma = -call_greeks['Gamma']

print(
    f"Covered Call Gamma: {covered_call_gamma:.8f}"
)


# In[158]:


# Step 107: Covered Call Vega

covered_call_vega = -call_greeks['Vega']

print(
    f"Covered Call Vega: {covered_call_vega:.2f}"
)

print(
    f"Effect of 1 percentage-point volatility increase: "
    f"₹{covered_call_vega * 0.01:.2f}"
)


# In[159]:


# Step 108: Covered Call Theta

covered_call_theta = -call_greeks['Theta']

print(
    f"Long Call Theta: {call_greeks['Theta']:.4f}"
)

print(
    f"Covered Call Theta: {covered_call_theta:.4f}"
)


# In[160]:


# Step 109: Covered Call Rho

covered_call_rho = -call_greeks['Rho']

print(
    f"Covered Call Rho: {covered_call_rho:.4f}"
)


# In[161]:


# Step 110: Covered Call Greeks

covered_call_greeks = pd.DataFrame({
    'Greek': [
        'Delta',
        'Gamma',
        'Vega',
        'Theta',
        'Rho'
    ],

    'Covered Call': [
        covered_call_delta,
        covered_call_gamma,
        covered_call_vega,
        covered_call_theta,
        covered_call_rho
    ]
})

covered_call_greeks


# In[162]:


# Step 114.1: NIFTY Price × Volatility Sensitivity Matrix

price_shocks = np.array([-0.10, -0.05, 0, 0.05, 0.10])
volatility_shocks = np.array([-0.20, -0.10, 0, 0.10, 0.20])

sensitivity_matrix = []

for price_shock in price_shocks:

    row = []

    scenario_spot = S * (1 + price_shock)

    for vol_shock in volatility_shocks:

        scenario_vol = sigma * (1 + vol_shock)

        scenario_option_price = black_scholes_price(
            scenario_spot,
            K,
            T,
            r,
            scenario_vol,
            option_type='call'
        )

        row.append(scenario_option_price)

    sensitivity_matrix.append(row)

sensitivity_df = pd.DataFrame(
    sensitivity_matrix,
    index=[
        '-10% NIFTY',
        '-5% NIFTY',
        'Base',
        '+5% NIFTY',
        '+10% NIFTY'
    ],
    columns=[
        '-20% Vol',
        '-10% Vol',
        'Base Vol',
        '+10% Vol',
        '+20% Vol'
    ]
)

sensitivity_df


# In[163]:


# Step 115: Option P&L Sensitivity

pnl_sensitivity_df = sensitivity_df - call_price

pnl_sensitivity_df


# In[164]:


# Step 116: Sensitivity Heatmap

plt.figure(figsize=(10, 6))

plt.imshow(
    pnl_sensitivity_df.values,
    aspect='auto'
)

plt.xticks(
    range(len(pnl_sensitivity_df.columns)),
    pnl_sensitivity_df.columns
)

plt.yticks(
    range(len(pnl_sensitivity_df.index)),
    pnl_sensitivity_df.index
)

plt.xlabel('Volatility Shock')
plt.ylabel('NIFTY Price Shock')
plt.title('Call Option P&L Sensitivity to NIFTY and Volatility')

plt.colorbar(label='Option P&L (₹)')

plt.tight_layout()
plt.show()


# In[165]:


# Step 119: Covered Call Sensitivity

covered_call_sensitivity = []

for price_shock in price_shocks:

    row = []

    scenario_spot = S * (1 + price_shock)

    for vol_shock in volatility_shocks:

        scenario_vol = sigma * (1 + vol_shock)

        scenario_call_price = black_scholes_price(
            scenario_spot,
            K,
            T,
            r,
            scenario_vol,
            option_type='call'
        )

        covered_call_scenario_pnl = (
            scenario_spot
            - S
            - (scenario_call_price - call_price)
        )

        row.append(covered_call_scenario_pnl)

    covered_call_sensitivity.append(row)

covered_call_sensitivity_df = pd.DataFrame(
    covered_call_sensitivity,
    index=[
        '-10% NIFTY',
        '-5% NIFTY',
        'Base',
        '+5% NIFTY',
        '+10% NIFTY'
    ],
    columns=[
        '-20% Vol',
        '-10% Vol',
        'Base Vol',
        '+10% Vol',
        '+20% Vol'
    ]
)

covered_call_sensitivity_df


# In[166]:


# Step 120: Selected Scenario Analysis

selected_scenarios = [
    ('Bear Market / Low Vol', -0.10, -0.20),
    ('Bear Market / High Vol', -0.10, 0.20),
    ('Mild Bear / Base Vol', -0.05, 0),
    ('Base Case', 0, 0),
    ('Mild Bull / Base Vol', 0.05, 0),
    ('Bull Market / Low Vol', 0.10, -0.20),
    ('Bull Market / High Vol', 0.10, 0.20)
]

scenario_analysis = []

for name, price_shock, vol_shock in selected_scenarios:

    scenario_spot = S * (1 + price_shock)
    scenario_vol = sigma * (1 + vol_shock)

    scenario_call = black_scholes_price(
        scenario_spot,
        K,
        T,
        r,
        scenario_vol,
        option_type='call'
    )

    option_pnl = scenario_call - call_price

    cc_pnl = (
        scenario_spot
        - S
        - option_pnl
    )

    scenario_analysis.append([
        name,
        price_shock * 100,
        vol_shock * 100,
        scenario_spot,
        scenario_call,
        option_pnl,
        cc_pnl
    ])

scenario_analysis_df = pd.DataFrame(
    scenario_analysis,
    columns=[
        'Scenario',
        'NIFTY Shock (%)',
        'Volatility Shock (%)',
        'Scenario NIFTY',
        'Scenario Call Price',
        'Call P&L',
        'Covered Call P&L'
    ]
)

scenario_analysis_df


# In[167]:


# Step 122: Covered Call Scenario P&L

base_vol_scenario = []

for price_shock in price_shocks:

    scenario_spot = S * (1 + price_shock)

    scenario_call = black_scholes_price(
        scenario_spot,
        K,
        T,
        r,
        sigma,
        option_type='call'
    )

    scenario_cc_pnl = (
        scenario_spot
        - S
        - (scenario_call - call_price)
    )

    base_vol_scenario.append(scenario_cc_pnl)

plt.figure(figsize=(10, 6))

plt.plot(
    price_shocks * 100,
    base_vol_scenario,
    marker='o',
    linewidth=2
)

plt.axhline(0, linewidth=1)

plt.xlabel('NIFTY Price Shock (%)')
plt.ylabel('Covered Call P&L (₹)')
plt.title('Covered Call P&L Under NIFTY Price Shocks')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[168]:


# Step 124: Identify Monte Carlo Variables

global_names = list(globals().keys())

print("Variables containing 'pnl':")

for name in global_names:
    if 'pnl' in name.lower():
        obj = globals()[name]
        print(
            name,
            "| Type:", type(obj).__name__,
            "| Shape:", getattr(obj, 'shape', 'N/A'),
            "| Length:", len(obj) if hasattr(obj, '__len__') else 'N/A'
        )

print("\nVariables containing 'sim':")

for name in global_names:
    if 'sim' in name.lower():
        obj = globals()[name]
        print(
            name,
            "| Type:", type(obj).__name__,
            "| Shape:", getattr(obj, 'shape', 'N/A'),
            "| Length:", len(obj) if hasattr(obj, '__len__') else 'N/A'
        )


# In[169]:


# Step 125: Monte Carlo P&L Distribution

mc_pnl = simulated_pnl_1d

print("===== MONTE CARLO P&L DISTRIBUTION =====")

print(f"Number of simulations: {len(mc_pnl):,}")
print(f"Mean P&L: ₹{np.mean(mc_pnl):,.2f}")
print(f"Median P&L: ₹{np.median(mc_pnl):,.2f}")
print(f"Minimum P&L: ₹{np.min(mc_pnl):,.2f}")
print(f"Maximum P&L: ₹{np.max(mc_pnl):,.2f}")
print(f"Standard Deviation: ₹{np.std(mc_pnl):,.2f}")


# In[170]:


# Step 126: Monte Carlo VaR Validation

var_95_mc = -np.percentile(mc_pnl, 5)
var_99_mc = -np.percentile(mc_pnl, 1)

print("===== MONTE CARLO VaR =====")

print(f"95% VaR: ₹{var_95_mc:,.2f}")
print(f"99% VaR: ₹{var_99_mc:,.2f}")


# In[171]:


# Step 127: Monte Carlo Expected Shortfall

var_threshold_95 = np.percentile(mc_pnl, 5)
var_threshold_99 = np.percentile(mc_pnl, 1)

es_95_mc = -np.mean(
    mc_pnl[mc_pnl <= var_threshold_95]
)

es_99_mc = -np.mean(
    mc_pnl[mc_pnl <= var_threshold_99]
)

print("===== MONTE CARLO EXPECTED SHORTFALL =====")

print(f"95% ES: ₹{es_95_mc:,.2f}")
print(f"99% ES: ₹{es_99_mc:,.2f}")


# In[172]:


# Step 128: VaR vs Expected Shortfall

print("===== RISK MEASURE VALIDATION =====")

print(f"95% VaR: ₹{var_95_mc:,.2f}")
print(f"95% ES : ₹{es_95_mc:,.2f}")

print()

print(f"99% VaR: ₹{var_99_mc:,.2f}")
print(f"99% ES : ₹{es_99_mc:,.2f}")

print()

print(
    "95% ES > 95% VaR:",
    es_95_mc > var_95_mc
)

print(
    "99% ES > 99% VaR:",
    es_99_mc > var_99_mc
)


# In[173]:


# Step 129: Monte Carlo P&L Distribution

plt.figure(figsize=(11, 6))

plt.hist(
    mc_pnl,
    bins=60
)

plt.axvline(
    -var_95_mc,
    linewidth=2,
    label=f'95% VaR = ₹{var_95_mc:.2f}'
)

plt.axvline(
    -var_99_mc,
    linewidth=2,
    label=f'99% VaR = ₹{var_99_mc:.2f}'
)

plt.xlabel('One-Day P&L (₹)')
plt.ylabel('Frequency')
plt.title('Monte Carlo Simulation: One-Day Option P&L Distribution')

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[174]:


# Step 130: Monte Carlo Tail Risk

plt.figure(figsize=(11, 6))

plt.hist(
    mc_pnl,
    bins=60
)

plt.axvline(
    -var_95_mc,
    linewidth=2,
    label='95% VaR'
)

plt.axvline(
    -var_99_mc,
    linewidth=2,
    label='99% VaR'
)

plt.axvline(
    -es_95_mc,
    linewidth=2,
    linestyle='--',
    label='95% ES'
)

plt.axvline(
    -es_99_mc,
    linewidth=2,
    linestyle='--',
    label='99% ES'
)

plt.xlabel('One-Day P&L (₹)')
plt.ylabel('Frequency')
plt.title('Monte Carlo Tail Risk: VaR and Expected Shortfall')

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[175]:


# Step 131: VaR and ES as Percentage of Option Premium

var_95_pct = var_95_mc / call_price * 100
var_99_pct = var_99_mc / call_price * 100

es_95_pct = es_95_mc / call_price * 100
es_99_pct = es_99_mc / call_price * 100

print("===== RISK RELATIVE TO OPTION PREMIUM =====")

print(f"95% VaR: {var_95_pct:.2f}% of premium")
print(f"99% VaR: {var_99_pct:.2f}% of premium")
print(f"95% ES : {es_95_pct:.2f}% of premium")
print(f"99% ES : {es_99_pct:.2f}% of premium")


# In[176]:


# Step 134: Monte Carlo Risk Summary

mc_risk_summary = pd.DataFrame({
    'Risk Measure': [
        '95% VaR',
        '99% VaR',
        '95% Expected Shortfall',
        '99% Expected Shortfall'
    ],

    'Loss (₹)': [
        var_95_mc,
        var_99_mc,
        es_95_mc,
        es_99_mc
    ],

    '% of Option Premium': [
        var_95_pct,
        var_99_pct,
        es_95_pct,
        es_99_pct
    ]
})

mc_risk_summary


# In[177]:


# Step 136: Inspect Stress-Test Inputs

print("NIFTY Shocks:")
print(nifty_shocks)

print("\nScenario Prices:")
print(scenario_prices)

print("\nScenario P&L:")
print(scenario_pnl)

print("\nStressed P&L:")
print(stressed_pnl)


# In[178]:


# Step 137: Stress-Test Table

stress_results = []

for shock in nifty_shocks:

    shocked_spot = S * (1 + shock)

    shocked_option_price = black_scholes_price(
        shocked_spot,
        K,
        T,
        r,
        sigma,
        option_type='call'
    )

    option_pnl = shocked_option_price - call_price

    stress_results.append([
        shock * 100,
        shocked_spot,
        shocked_option_price,
        option_pnl
    ])

stress_df = pd.DataFrame(
    stress_results,
    columns=[
        'NIFTY Shock (%)',
        'Shocked NIFTY',
        'Option Price',
        'Option P&L'
    ]
)

stress_df


# In[179]:


# Step 138: Stress Loss

stress_df['Stress Loss'] = -stress_df['Option P&L']

stress_df


# In[180]:


# Step 139: Stress Loss as Percentage of Premium

stress_df['Loss (% of Premium)'] = (
    stress_df['Stress Loss'] / call_price
) * 100

stress_df


# In[181]:


# Step 140: Worst Stress Scenario

worst_stress = stress_df.loc[
    stress_df['Stress Loss'].idxmax()
]

print("===== WORST STRESS SCENARIO =====")

print(
    f"NIFTY Shock: "
    f"{worst_stress['NIFTY Shock (%)']:.2f}%"
)

print(
    f"Shocked NIFTY: "
    f"₹{worst_stress['Shocked NIFTY']:,.2f}"
)

print(
    f"Option Price: "
    f"₹{worst_stress['Option Price']:,.2f}"
)

print(
    f"Option P&L: "
    f"₹{worst_stress['Option P&L']:,.2f}"
)

print(
    f"Stress Loss: "
    f"₹{worst_stress['Stress Loss']:,.2f}"
)

print(
    f"Loss as % of Premium: "
    f"{worst_stress['Loss (% of Premium)']:.2f}%"
)


# In[182]:


# Step 141: Stress-Test P&L Chart

plt.figure(figsize=(10, 6))

plt.plot(
    stress_df['NIFTY Shock (%)'],
    stress_df['Option P&L'],
    marker='o',
    linewidth=2
)

plt.axhline(
    0,
    linewidth=1
)

plt.xlabel('NIFTY Shock (%)')
plt.ylabel('Option P&L (₹)')
plt.title('NIFTY Call Option Stress-Test P&L')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[183]:


# Step 142: Stress Loss vs Monte Carlo VaR

stress_to_var_ratio = (
    worst_stress['Stress Loss'] / var_99_mc
)

print(
    f"Worst Stress Loss / 99% VaR: "
    f"{stress_to_var_ratio:.2f}x"
)


# In[184]:


# Step 145: Covered Call Stress Test

covered_call_stress = []

for shock in nifty_shocks:

    shocked_spot = S * (1 + shock)

    shocked_call_price = black_scholes_price(
        shocked_spot,
        K,
        T,
        r,
        sigma,
        option_type='call'
    )

    long_nifty_pnl = shocked_spot - S

    short_call_pnl = call_price - shocked_call_price

    total_cc_pnl = (
        long_nifty_pnl +
        short_call_pnl
    )

    covered_call_stress.append([
        shock * 100,
        shocked_spot,
        long_nifty_pnl,
        short_call_pnl,
        total_cc_pnl
    ])

covered_call_stress_df = pd.DataFrame(
    covered_call_stress,
    columns=[
        'NIFTY Shock (%)',
        'Shocked NIFTY',
        'Long NIFTY P&L',
        'Short Call P&L',
        'Covered Call P&L'
    ]
)

covered_call_stress_df


# In[185]:


# Step 146: Stress Comparison

covered_call_stress_df['Long NIFTY Loss'] = (
    -covered_call_stress_df['Long NIFTY P&L']
)

covered_call_stress_df['Covered Call Loss'] = (
    -covered_call_stress_df['Covered Call P&L']
)

covered_call_stress_df['Loss Reduction'] = (
    covered_call_stress_df['Long NIFTY Loss']
    - covered_call_stress_df['Covered Call Loss']
)

covered_call_stress_df


# In[186]:


# Step 147: Worst Covered Call Stress

worst_cc_stress = covered_call_stress_df.loc[
    covered_call_stress_df['Covered Call Loss'].idxmax()
]

print("===== WORST COVERED CALL STRESS =====")

print(
    f"NIFTY Shock: "
    f"{worst_cc_stress['NIFTY Shock (%)']:.2f}%"
)

print(
    f"Long NIFTY Loss: "
    f"₹{worst_cc_stress['Long NIFTY Loss']:,.2f}"
)

print(
    f"Covered Call Loss: "
    f"₹{worst_cc_stress['Covered Call Loss']:,.2f}"
)

print(
    f"Loss Reduction: "
    f"₹{worst_cc_stress['Loss Reduction']:,.2f}"
)


# In[187]:


# Step 148: Downside Cushion

worst_long_loss = worst_cc_stress['Long NIFTY Loss']
worst_cc_loss = worst_cc_stress['Covered Call Loss']

downside_cushion = (
    worst_long_loss - worst_cc_loss
)

downside_cushion_pct = (
    downside_cushion / worst_long_loss
) * 100

print(
    f"Downside Cushion: ₹{downside_cushion:,.2f}"
)

print(
    f"Downside Cushion: {downside_cushion_pct:.2f}%"
)


# In[188]:


# Step 149: Long NIFTY vs Covered Call Stress Test

plt.figure(figsize=(10, 6))

plt.plot(
    covered_call_stress_df['NIFTY Shock (%)'],
    covered_call_stress_df['Long NIFTY P&L'],
    marker='o',
    linewidth=2,
    label='Long NIFTY'
)

plt.plot(
    covered_call_stress_df['NIFTY Shock (%)'],
    covered_call_stress_df['Covered Call P&L'],
    marker='o',
    linewidth=2,
    label='Covered Call'
)

plt.axhline(
    0,
    linewidth=1
)

plt.xlabel('NIFTY Shock (%)')
plt.ylabel('P&L (₹)')
plt.title('Stress-Test Comparison: Long NIFTY vs Covered Call')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[189]:


# Step 151: Expiry Price Range

expiry_shocks = np.linspace(-0.20, 0.20, 81)

expiry_prices = S * (1 + expiry_shocks)

print(f"Minimum expiry NIFTY: ₹{expiry_prices.min():,.2f}")
print(f"Maximum expiry NIFTY: ₹{expiry_prices.max():,.2f}")


# In[190]:


# Step 152: Long NIFTY P&L at Expiry

long_nifty_expiry_pnl = expiry_prices - S

print(
    f"Long NIFTY P&L at current level: "
    f"₹{long_nifty_expiry_pnl[40]:,.2f}"
)


# In[191]:


# Step 153: Short Call P&L at Expiry

short_call_expiry_pnl = (
    call_price
    - np.maximum(expiry_prices - K, 0)
)


# In[192]:


# Step 154: Covered Call P&L at Expiry

covered_call_expiry_pnl = (
    long_nifty_expiry_pnl
    + short_call_expiry_pnl
)

print(
    f"Covered Call P&L at current NIFTY: "
    f"₹{covered_call_expiry_pnl[40]:,.2f}"
)


# In[193]:


# Step 155: Exact Covered Call Expiry P&L

covered_call_expiry_pnl = (
    expiry_prices
    - S
    + call_price
    - np.maximum(expiry_prices - K, 0)
)

print(
    f"Minimum Covered Call P&L: "
    f"₹{covered_call_expiry_pnl.min():,.2f}"
)

print(
    f"Maximum Covered Call P&L: "
    f"₹{covered_call_expiry_pnl.max():,.2f}"
)


# In[194]:


# Step 156: Break-Even Price

break_even = S - call_price

print(
    f"Covered Call Break-Even: ₹{break_even:,.2f}"
)

print(
    f"Break-Even as % of Spot: "
    f"{(break_even / S - 1) * 100:.2f}%"
)


# In[195]:


# Step 157: Maximum Profit

max_profit = K - S + call_price

print(
    f"Maximum Covered Call Profit: "
    f"₹{max_profit:,.2f}"
)

print(
    f"Maximum Profit (% of NIFTY): "
    f"{(max_profit / S) * 100:.2f}%"
)


# In[196]:


# Step 158: Upside Cap

upside_to_strike = K - S

print(
    f"Strike relative to current NIFTY: "
    f"{(K / S - 1) * 100:.2f}%"
)

print(
    f"Distance to strike: "
    f"₹{upside_to_strike:,.2f}"
)


# In[197]:


# Step 159: Maximum Theoretical Downside

maximum_downside = S - call_price

print(
    f"Maximum Theoretical Loss: "
    f"₹{maximum_downside:,.2f}"
)

print(
    f"Maximum Loss (% of NIFTY): "
    f"{(maximum_downside / S) * 100:.2f}%"
)


# In[198]:


# Step 160: Covered Call Payoff Table

selected_expiry_shocks = np.array([
    -20, -10, -5, 0, 5, 10, 20
])

selected_expiry_prices = S * (
    1 + selected_expiry_shocks / 100
)

selected_long_pnl = (
    selected_expiry_prices - S
)

selected_call_intrinsic = np.maximum(
    selected_expiry_prices - K,
    0
)

selected_covered_call_pnl = (
    selected_long_pnl
    + call_price
    - selected_call_intrinsic
)

payoff_table = pd.DataFrame({
    'NIFTY Change (%)': selected_expiry_shocks,
    'Expiry NIFTY': selected_expiry_prices,
    'Long NIFTY P&L': selected_long_pnl,
    'Call Intrinsic Value': selected_call_intrinsic,
    'Covered Call P&L': selected_covered_call_pnl
})

payoff_table


# In[199]:


# Step 161: Covered Call Payoff Diagram

plt.figure(figsize=(11, 6))

plt.plot(
    expiry_prices,
    long_nifty_expiry_pnl,
    linewidth=2,
    label='Long NIFTY'
)

plt.plot(
    expiry_prices,
    covered_call_expiry_pnl,
    linewidth=2,
    label='Covered Call'
)

plt.axhline(
    0,
    linewidth=1
)

plt.axvline(
    K,
    linewidth=1,
    linestyle='--',
    label=f'Strike = ₹{K:,.0f}'
)

plt.axvline(
    break_even,
    linewidth=1,
    linestyle='--',
    label=f'Break-even = ₹{break_even:,.0f}'
)

plt.xlabel('NIFTY at Expiry')
plt.ylabel('P&L (₹)')
plt.title('Covered Call Payoff at Expiry')

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[200]:


# Step 163: Premium Cushion

premium_cushion_pct = (
    call_price / S
) * 100

print(
    f"Premium Cushion: ₹{call_price:,.2f}"
)

print(
    f"Premium Cushion (% of NIFTY): "
    f"{premium_cushion_pct:.2f}%"
)


# In[201]:


# Step 164: Covered Call Payoff Summary

payoff_summary = pd.DataFrame({
    'Metric': [
        'Current NIFTY',
        'Strike Price',
        'Call Premium',
        'Break-Even',
        'Maximum Profit',
        'Maximum Theoretical Loss',
        'Premium Cushion (%)',
        'Strike vs Spot (%)'
    ],

    'Value': [
        S,
        K,
        call_price,
        break_even,
        max_profit,
        -maximum_downside,
        premium_cushion_pct,
        ((K / S) - 1) * 100
    ]
})

payoff_summary


# In[202]:


# Step 168: Find Backtest DataFrames

for name in list(globals().keys()):

    obj = globals()[name]

    if isinstance(obj, pd.DataFrame):

        if len(obj) <= 30:

            print("\nVariable:", name)
            print("Shape:", obj.shape)
            print("Columns:", list(obj.columns))


# In[203]:


# Step 170: Backtest Cycle Count

print("Number of backtest cycles:", len(backtest_df))


# In[204]:


# Step 171A: Inspect Backtest DataFrame

print("Backtest DataFrame shape:", backtest_df.shape)

print("\nColumns:")
for i, col in enumerate(backtest_df.columns):
    print(i, "->", repr(col))

print("\nFirst 5 rows:")
print(backtest_df.head())


# In[205]:


# Step 171: Validate Existing Wealth

print("===== WEALTH VALIDATION =====")

print(
    "Final Long NIFTY Wealth:",
    backtest_df['Long NIFTY Wealth'].iloc[-1]
)

print(
    "Final Covered Call Wealth:",
    backtest_df['Covered Call Wealth'].iloc[-1]
)

print("\nExpected from previous results:")

print("NIFTY Wealth: 0.9655603085780504")
print("Covered Call Wealth: 1.0309587070848725")


# In[206]:


# Step 172: Independently Recalculate Wealth

nifty_wealth_check = (
    1 + backtest_df['Long NIFTY Return']
).cumprod()

covered_call_wealth_check = (
    1 + backtest_df['Covered Call Return']
).cumprod()

print("===== WEALTH CHECK =====")

print(
    "Stored NIFTY Wealth:",
    backtest_df['Long NIFTY Wealth'].iloc[-1]
)

print(
    "Calculated NIFTY Wealth:",
    nifty_wealth_check.iloc[-1]
)

print()

print(
    "Stored Covered Call Wealth:",
    backtest_df['Covered Call Wealth'].iloc[-1]
)

print(
    "Calculated Covered Call Wealth:",
    covered_call_wealth_check.iloc[-1]
)


# In[207]:


# Step 173: Wealth Consistency Check

nifty_wealth_difference = np.max(
    np.abs(
        backtest_df['Long NIFTY Wealth']
        - nifty_wealth_check
    )
)

covered_call_wealth_difference = np.max(
    np.abs(
        backtest_df['Covered Call Wealth']
        - covered_call_wealth_check
    )
)

print(
    f"Maximum NIFTY Wealth Difference: "
    f"{nifty_wealth_difference:.12f}"
)

print(
    f"Maximum Covered Call Wealth Difference: "
    f"{covered_call_wealth_difference:.12f}"
)


# In[208]:


# Step 174: Cumulative Wealth Curve

plt.figure(figsize=(11, 6))

plt.plot(
    backtest_df['Expiry Date'],
    backtest_df['Long NIFTY Wealth'],
    marker='o',
    linewidth=2,
    label='Long NIFTY'
)

plt.plot(
    backtest_df['Expiry Date'],
    backtest_df['Covered Call Wealth'],
    marker='o',
    linewidth=2,
    label='Covered Call'
)

plt.axhline(
    1,
    linewidth=1
)

plt.xlabel('Expiry Date')
plt.ylabel('Cumulative Wealth')
plt.title('Cumulative Wealth: Long NIFTY vs Covered Call')

plt.legend()
plt.grid(True)
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# In[209]:


# Step 175: Cumulative Return

nifty_final_wealth = backtest_df[
    'Long NIFTY Wealth'
].iloc[-1]

covered_call_final_wealth = backtest_df[
    'Covered Call Wealth'
].iloc[-1]

nifty_cumulative_return = (
    nifty_final_wealth - 1
) * 100

covered_call_cumulative_return = (
    covered_call_final_wealth - 1
) * 100

print("===== CUMULATIVE RETURN =====")

print(
    f"Long NIFTY: "
    f"{nifty_cumulative_return:.2f}%"
)

print(
    f"Covered Call: "
    f"{covered_call_cumulative_return:.2f}%"
)


# In[210]:


# Step 176: Wealth Advantage

wealth_advantage = (
    covered_call_final_wealth
    - nifty_final_wealth
)

wealth_advantage_pct = (
    wealth_advantage / nifty_final_wealth
) * 100

print(
    f"Absolute Wealth Advantage: "
    f"{wealth_advantage:.4f}"
)

print(
    f"Relative Wealth Advantage: "
    f"{wealth_advantage_pct:.2f}%"
)


# In[211]:


# Step 177: Outperformance Validation

outperformance_count = (
    backtest_df['Covered Call Outperformed']
    .sum()
)

total_cycles = len(backtest_df)

outperformance_rate = (
    outperformance_count
    / total_cycles
) * 100

print("===== OUTPERFORMANCE =====")

print(
    f"Covered Call Outperformed: "
    f"{outperformance_count} / {total_cycles} cycles"
)

print(
    f"Outperformance Rate: "
    f"{outperformance_rate:.2f}%"
)


# In[212]:


# Step 178: Drawdown

backtest_df['Long NIFTY Drawdown'] = (
    backtest_df['Long NIFTY Wealth']
    / backtest_df['Long NIFTY Wealth'].cummax()
    - 1
)

backtest_df['Covered Call Drawdown'] = (
    backtest_df['Covered Call Wealth']
    / backtest_df['Covered Call Wealth'].cummax()
    - 1
)

print("===== MAX DRAWDOWN =====")

print(
    f"Long NIFTY: "
    f"{backtest_df['Long NIFTY Drawdown'].min() * 100:.2f}%"
)

print(
    f"Covered Call: "
    f"{backtest_df['Covered Call Drawdown'].min() * 100:.2f}%"
)


# In[213]:


# Step 179: Drawdown Comparison

plt.figure(figsize=(11, 6))

plt.plot(
    backtest_df['Expiry Date'],
    backtest_df['Long NIFTY Drawdown'] * 100,
    marker='o',
    linewidth=2,
    label='Long NIFTY'
)

plt.plot(
    backtest_df['Expiry Date'],
    backtest_df['Covered Call Drawdown'] * 100,
    marker='o',
    linewidth=2,
    label='Covered Call'
)

plt.axhline(
    0,
    linewidth=1
)

plt.xlabel('Expiry Date')
plt.ylabel('Drawdown (%)')
plt.title('Drawdown Comparison: Long NIFTY vs Covered Call')

plt.legend()
plt.grid(True)
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# In[214]:


# Step 180: Return Volatility

nifty_volatility = (
    backtest_df['Long NIFTY Return'].std()
)

covered_call_volatility = (
    backtest_df['Covered Call Return'].std()
)

print("===== VOLATILITY =====")

print(
    f"Long NIFTY Volatility: "
    f"{nifty_volatility * 100:.2f}%"
)

print(
    f"Covered Call Volatility: "
    f"{covered_call_volatility * 100:.2f}%"
)


# In[215]:


# Step 181: Volatility Reduction

volatility_reduction = (
    1
    - covered_call_volatility
    / nifty_volatility
) * 100

print(
    f"Covered Call Volatility Reduction: "
    f"{volatility_reduction:.2f}%"
)


# In[216]:


# Step 182: Return Comparison

plt.figure(figsize=(11, 6))

plt.plot(
    backtest_df['Expiry Date'],
    backtest_df['Long NIFTY Return'] * 100,
    marker='o',
    linewidth=2,
    label='Long NIFTY'
)

plt.plot(
    backtest_df['Expiry Date'],
    backtest_df['Covered Call Return'] * 100,
    marker='o',
    linewidth=2,
    label='Covered Call'
)

plt.axhline(
    0,
    linewidth=1
)

plt.xlabel('Expiry Date')
plt.ylabel('Cycle Return (%)')
plt.title('Cycle-Level Return Comparison')

plt.legend()
plt.grid(True)
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# In[217]:


# Step 183: Market Regime Performance

regime_analysis = (
    backtest_df
    .groupby('Market Regime')
    .agg(
        Cycles=('Market Regime', 'count'),
        Avg_NIFTY_Return=('Long NIFTY Return', 'mean'),
        Avg_Covered_Call_Return=('Covered Call Return', 'mean'),
        Avg_Premium_Yield=('Premium Yield', 'mean'),
        Outperformance_Rate=('Covered Call Outperformed', 'mean')
    )
)

regime_analysis['Outperformance_Rate'] *= 100

regime_analysis


# In[218]:


# Step 184: Covered Call Advantage by Market Regime

advantage_by_regime = (
    backtest_df
    .groupby('Market Regime')
    .agg(
        Cycles=('Covered Call Advantage', 'count'),
        Average_Advantage=('Covered Call Advantage', 'mean'),
        Minimum_Advantage=('Covered Call Advantage', 'min'),
        Maximum_Advantage=('Covered Call Advantage', 'max')
    )
)

advantage_by_regime


# In[219]:


# Step 185: Final Backtest Summary

final_backtest_summary = pd.DataFrame({
    'Metric': [
        'Number of Cycles',
        'Average Return',
        'Volatility',
        'Maximum Drawdown',
        'Final Wealth',
        'Cumulative Return',
        'Outperformance Rate'
    ],

    'Long NIFTY': [
        len(backtest_df),
        backtest_df['Long NIFTY Return'].mean() * 100,
        nifty_volatility * 100,
        backtest_df['Long NIFTY Drawdown'].min() * 100,
        nifty_final_wealth,
        nifty_cumulative_return,
        np.nan
    ],

    'Covered Call': [
        len(backtest_df),
        backtest_df['Covered Call Return'].mean() * 100,
        covered_call_volatility * 100,
        backtest_df['Covered Call Drawdown'].min() * 100,
        covered_call_final_wealth,
        covered_call_cumulative_return,
        outperformance_rate
    ]
})

final_backtest_summary


# In[220]:


# Step 186: Identify Sharpe Variables

for name in list(globals().keys()):
    if 'sharpe' in name.lower():
        obj = globals()[name]

        print(
            name,
            "| Type:", type(obj),
            "| Value:", obj if np.isscalar(obj) else "Non-scalar"
        )


# In[221]:


# Step 186B: Identify Risk-Free Rate Variables

for name in list(globals().keys()):
    if (
        'risk_free' in name.lower()
        or name.lower() in ['rf', 'riskfree', 'risk_free_rate']
    ):
        obj = globals()[name]

        print(
            name,
            "| Type:", type(obj),
            "| Value:", obj if np.isscalar(obj) else "Non-scalar"
        )


# In[222]:


# Step 187: Raw Sharpe Ratio

nifty_mean_return = (
    backtest_df['Long NIFTY Return'].mean()
)

covered_call_mean_return = (
    backtest_df['Covered Call Return'].mean()
)

nifty_raw_sharpe = (
    nifty_mean_return / nifty_volatility
)

covered_call_raw_sharpe = (
    covered_call_mean_return / covered_call_volatility
)

print("===== RAW SHARPE =====")

print(
    f"Long NIFTY Raw Sharpe: "
    f"{nifty_raw_sharpe:.4f}"
)

print(
    f"Covered Call Raw Sharpe: "
    f"{covered_call_raw_sharpe:.4f}"
)


# In[223]:


# Step 188: Convert Annual Risk-Free Rate to Monthly Equivalent

annual_rf = 0.065

monthly_rf = (
    (1 + annual_rf) ** (1 / 12)
    - 1
)

print(
    f"Annual Risk-Free Rate: "
    f"{annual_rf * 100:.2f}%"
)

print(
    f"Monthly Equivalent Risk-Free Rate: "
    f"{monthly_rf * 100:.4f}%"
)


# In[224]:


# Step 189: Period Sharpe Ratio

nifty_excess_return = (
    nifty_mean_return - monthly_rf
)

covered_call_excess_return = (
    covered_call_mean_return - monthly_rf
)

nifty_sharpe_period = (
    nifty_excess_return / nifty_volatility
)

covered_call_sharpe_period = (
    covered_call_excess_return
    / covered_call_volatility
)

print("===== PERIOD SHARPE =====")

print(
    f"Long NIFTY Sharpe: "
    f"{nifty_sharpe_period:.4f}"
)

print(
    f"Covered Call Sharpe: "
    f"{covered_call_sharpe_period:.4f}"
)


# In[225]:


# Step 190: Annualized Sharpe Ratio

nifty_sharpe_annual = (
    nifty_sharpe_period * np.sqrt(12)
)

covered_call_sharpe_annual = (
    covered_call_sharpe_period * np.sqrt(12)
)

print("===== ANNUALIZED SHARPE =====")

print(
    f"Long NIFTY Annualized Sharpe: "
    f"{nifty_sharpe_annual:.4f}"
)

print(
    f"Covered Call Annualized Sharpe: "
    f"{covered_call_sharpe_annual:.4f}"
)


# In[226]:


# Step 191: Sharpe Method Comparison

sharpe_comparison = pd.DataFrame({
    'Method': [
        'Original Backtest',
        'Raw Sharpe',
        'Period Sharpe',
        'Annualized Sharpe'
    ],

    'Long NIFTY': [
        -0.6221946960690026,
        nifty_raw_sharpe,
        nifty_sharpe_period,
        nifty_sharpe_annual
    ],

    'Covered Call': [
        -0.6623767925842156,
        covered_call_raw_sharpe,
        covered_call_sharpe_period,
        covered_call_sharpe_annual
    ]
})

sharpe_comparison


# In[227]:


# Step 192: Inspect Existing Excess Returns

print(
    backtest_df[
        [
            'Long NIFTY Return',
            'Long NIFTY Excess Return',
            'Covered Call Return',
            'Covered Call Excess Return'
        ]
    ].head(10)
)


# In[228]:


print("\n===== AVERAGE EXCESS RETURNS =====")

print(
    "Long NIFTY:",
    backtest_df['Long NIFTY Excess Return'].mean()
)

print(
    "Covered Call:",
    backtest_df['Covered Call Excess Return'].mean()
)


# In[229]:


# Step 193: Validate Excess Return Construction

print("===== EXCESS RETURN VALIDATION =====")

test_nifty = (
    backtest_df['Long NIFTY Return']
    - backtest_df['Long NIFTY Excess Return']
)

test_cc = (
    backtest_df['Covered Call Return']
    - backtest_df['Covered Call Excess Return']
)

print(
    "NIFTY implied risk-free rate:"
)

print(test_nifty.head())

print("\nCovered Call implied risk-free rate:")

print(test_cc.head())


# In[230]:


# Step 194: Identify Original Risk-Free Rate

original_rf_period = 0.005262

implied_annual_rf = (
    (1 + original_rf_period) ** 12
    - 1
)

print(
    f"Original Period Risk-Free Rate: "
    f"{original_rf_period * 100:.4f}%"
)

print(
    f"Implied Annual Risk-Free Rate: "
    f"{implied_annual_rf * 100:.2f}%"
)


# In[231]:


# Step 195: Reproduce Original Sharpe Ratio

original_rf = 0.005262

nifty_excess = (
    backtest_df['Long NIFTY Return']
    - original_rf
)

covered_call_excess = (
    backtest_df['Covered Call Return']
    - original_rf
)

nifty_sharpe_reproduced = (
    nifty_excess.mean()
    / backtest_df['Long NIFTY Return'].std()
)

covered_call_sharpe_reproduced = (
    covered_call_excess.mean()
    / backtest_df['Covered Call Return'].std()
)

print("===== REPRODUCED SHARPE =====")

print(
    f"Long NIFTY: "
    f"{nifty_sharpe_reproduced:.6f}"
)

print(
    f"Covered Call: "
    f"{covered_call_sharpe_reproduced:.6f}"
)


# In[232]:


# Step 197: Sortino Ratio

def calculate_sortino(returns, risk_free):

    excess_returns = returns - risk_free

    downside_returns = excess_returns[
        excess_returns < 0
    ]

    downside_deviation = np.sqrt(
        np.mean(downside_returns ** 2)
    )

    if downside_deviation == 0:
        return np.nan

    return (
        excess_returns.mean()
        / downside_deviation
    )

nifty_sortino = calculate_sortino(
    backtest_df['Long NIFTY Return'],
    original_rf
)

covered_call_sortino = calculate_sortino(
    backtest_df['Covered Call Return'],
    original_rf
)

print("===== SORTINO RATIO =====")

print(
    f"Long NIFTY Sortino: "
    f"{nifty_sortino:.6f}"
)

print(
    f"Covered Call Sortino: "
    f"{covered_call_sortino:.6f}"
)


# In[233]:


# Step 198: Annualized Return

num_cycles = len(backtest_df)

years = num_cycles / 12

nifty_annualized_return = (
    nifty_final_wealth ** (1 / years)
    - 1
)

covered_call_annualized_return = (
    covered_call_final_wealth ** (1 / years)
    - 1
)

print("===== ANNUALIZED RETURN =====")

print(
    f"Long NIFTY: "
    f"{nifty_annualized_return * 100:.2f}%"
)

print(
    f"Covered Call: "
    f"{covered_call_annualized_return * 100:.2f}%"
)


# In[234]:


# Step 198B: Calmar Ratio

nifty_max_dd = abs(
    backtest_df['Long NIFTY Drawdown'].min()
)

covered_call_max_dd = abs(
    backtest_df['Covered Call Drawdown'].min()
)

nifty_calmar = (
    nifty_annualized_return
    / nifty_max_dd
)

covered_call_calmar = (
    covered_call_annualized_return
    / covered_call_max_dd
)

print("===== CALMAR RATIO =====")

print(
    f"Long NIFTY Calmar: "
    f"{nifty_calmar:.6f}"
)

print(
    f"Covered Call Calmar: "
    f"{covered_call_calmar:.6f}"
)


# In[235]:


# Step 199: Risk-Adjusted Performance Summary

risk_adjusted_summary = pd.DataFrame({
    'Metric': [
        'Average Return',
        'Volatility',
        'Maximum Drawdown',
        'Sharpe Ratio',
        'Sortino Ratio',
        'Calmar Ratio',
        'Final Wealth',
        'Cumulative Return'
    ],

    'Long NIFTY': [
        backtest_df['Long NIFTY Return'].mean() * 100,
        nifty_volatility * 100,
        backtest_df['Long NIFTY Drawdown'].min() * 100,
        nifty_sharpe_reproduced,
        nifty_sortino,
        nifty_calmar,
        nifty_final_wealth,
        nifty_cumulative_return
    ],

    'Covered Call': [
        backtest_df['Covered Call Return'].mean() * 100,
        covered_call_volatility * 100,
        backtest_df['Covered Call Drawdown'].min() * 100,
        covered_call_sharpe_reproduced,
        covered_call_sortino,
        covered_call_calmar,
        covered_call_final_wealth,
        covered_call_cumulative_return
    ]
})

risk_adjusted_summary


# In[236]:


# Step 201: Market Regime Distribution

regime_counts = (
    backtest_df['Market Regime']
    .value_counts()
)

print("===== MARKET REGIME DISTRIBUTION =====")
print(regime_counts)


# In[237]:


# Step 201B: Market Regime Percentages

regime_percentages = (
    backtest_df['Market Regime']
    .value_counts(normalize=True)
    * 100
)

print("===== MARKET REGIME PERCENTAGES =====")
print(regime_percentages)


# In[238]:


# Step 202: Market Regime Performance

regime_performance = (
    backtest_df
    .groupby('Market Regime')
    .agg(
        Cycles=('Market Regime', 'count'),

        Avg_NIFTY_Return=(
            'Long NIFTY Return',
            'mean'
        ),

        Avg_Covered_Call_Return=(
            'Covered Call Return',
            'mean'
        ),

        Avg_Premium_Yield=(
            'Premium Yield',
            'mean'
        ),

        Outperformance_Rate=(
            'Covered Call Outperformed',
            'mean'
        ),

        Avg_Covered_Call_Advantage=(
            'Covered Call Advantage',
            'mean'
        )
    )
)

regime_performance['Avg_NIFTY_Return'] *= 100
regime_performance['Avg_Covered_Call_Return'] *= 100
regime_performance['Avg_Premium_Yield'] *= 100
regime_performance['Outperformance_Rate'] *= 100
regime_performance['Avg_Covered_Call_Advantage'] *= 100

print("===== MARKET REGIME PERFORMANCE =====")

regime_performance


# In[239]:


# Step 203: Regime-Level Volatility

regime_volatility = (
    backtest_df
    .groupby('Market Regime')
    .agg(
        NIFTY_Volatility=(
            'Long NIFTY Return',
            'std'
        ),

        Covered_Call_Volatility=(
            'Covered Call Return',
            'std'
        )
    )
)

regime_volatility['NIFTY_Volatility'] *= 100
regime_volatility['Covered_Call_Volatility'] *= 100

regime_volatility['Volatility_Reduction'] = (
    1
    - regime_volatility['Covered_Call_Volatility']
    / regime_volatility['NIFTY_Volatility']
) * 100

print("===== REGIME VOLATILITY =====")

regime_volatility


# In[240]:


# Step 204: Worst Cycle Return by Regime

regime_downside = (
    backtest_df
    .groupby('Market Regime')
    .agg(
        Worst_NIFTY_Return=(
            'Long NIFTY Return',
            'min'
        ),

        Worst_Covered_Call_Return=(
            'Covered Call Return',
            'min'
        )
    )
)

regime_downside['Worst_NIFTY_Return'] *= 100
regime_downside['Worst_Covered_Call_Return'] *= 100

print("===== WORST CYCLE RETURN BY REGIME =====")

regime_downside


# In[241]:


# Step 205: Complete Regime Analysis

complete_regime_analysis = (
    regime_performance
    .join(regime_volatility)
    .join(regime_downside)
)

complete_regime_analysis


# In[242]:


# Step 206: Best Market Regime

best_regime = (
    complete_regime_analysis[
        'Avg_Covered_Call_Advantage'
    ].idxmax()
)

best_advantage = (
    complete_regime_analysis.loc[
        best_regime,
        'Avg_Covered_Call_Advantage'
    ]
)

print(
    f"Best regime for Covered Call: "
    f"{best_regime}"
)

print(
    f"Average advantage: "
    f"{best_advantage:.2f}%"
)


# In[243]:


# Step 206: Best Market Regime

best_regime = (
    complete_regime_analysis[
        'Avg_Covered_Call_Advantage'
    ].idxmax()
)

best_advantage = (
    complete_regime_analysis.loc[
        best_regime,
        'Avg_Covered_Call_Advantage'
    ]
)

print(
    f"Best regime for Covered Call: "
    f"{best_regime}"
)

print(
    f"Average advantage: "
    f"{best_advantage:.2f}%"
)


# In[244]:


# Step 209: Bear Market Premium Cushion

bear_data = backtest_df[
    backtest_df['Market Regime'] == 'Bear'
]

bear_nifty_avg = (
    bear_data['Long NIFTY Return'].mean()
)

bear_cc_avg = (
    bear_data['Covered Call Return'].mean()
)

bear_cushion = (
    bear_cc_avg - bear_nifty_avg
)

print("===== BEAR MARKET PREMIUM CUSHION =====")

print(
    f"Average NIFTY Return: "
    f"{bear_nifty_avg * 100:.2f}%"
)

print(
    f"Average Covered Call Return: "
    f"{bear_cc_avg * 100:.2f}%"
)

print(
    f"Premium Cushion: "
    f"{bear_cushion * 100:.2f}%"
)


# In[245]:


# Step 210: Bull Market Upside Sacrifice

bull_data = backtest_df[
    backtest_df['Market Regime'] == 'Bull'
]

bull_nifty_avg = (
    bull_data['Long NIFTY Return'].mean()
)

bull_cc_avg = (
    bull_data['Covered Call Return'].mean()
)

bull_opportunity_cost = (
    bull_nifty_avg - bull_cc_avg
)

print("===== BULL MARKET UPSIDE SACRIFICE =====")

print(
    f"Average NIFTY Return: "
    f"{bull_nifty_avg * 100:.2f}%"
)

print(
    f"Average Covered Call Return: "
    f"{bull_cc_avg * 100:.2f}%"
)

print(
    f"Upside Sacrifice: "
    f"{bull_opportunity_cost * 100:.2f}%"
)


# In[246]:


# Step 211: Neutral Market Benefit

neutral_data = backtest_df[
    backtest_df['Market Regime'] == 'Neutral'
]

neutral_nifty_avg = (
    neutral_data['Long NIFTY Return'].mean()
)

neutral_cc_avg = (
    neutral_data['Covered Call Return'].mean()
)

neutral_benefit = (
    neutral_cc_avg - neutral_nifty_avg
)

print("===== NEUTRAL MARKET BENEFIT =====")

print(
    f"Average NIFTY Return: "
    f"{neutral_nifty_avg * 100:.2f}%"
)

print(
    f"Average Covered Call Return: "
    f"{neutral_cc_avg * 100:.2f}%"
)

print(
    f"Covered Call Benefit: "
    f"{neutral_benefit * 100:.2f}%"
)


# In[247]:


# Step 212: Premium Yield by Market Regime

plt.figure(figsize=(9, 6))

premium_by_regime = (
    backtest_df
    .groupby('Market Regime')['Premium Yield']
    .mean()
    * 100
)

premium_by_regime.plot(
    kind='bar'
)

plt.xlabel('Market Regime')
plt.ylabel('Average Premium Yield (%)')
plt.title('Average Option Premium Yield by Market Regime')

plt.xticks(rotation=0)
plt.grid(axis='y')

plt.tight_layout()
plt.show()


# In[248]:


# Step 213: Covered Call Trade-Off by Market Regime

tradeoff_table = pd.DataFrame({
    'Regime': [
        'Bear',
        'Neutral',
        'Bull'
    ],

    'NIFTY_Return': [
        bear_nifty_avg * 100,
        neutral_nifty_avg * 100,
        bull_nifty_avg * 100
    ],

    'Covered_Call_Return': [
        bear_cc_avg * 100,
        neutral_cc_avg * 100,
        bull_cc_avg * 100
    ],

    'Covered_Call_Advantage': [
        bear_cushion * 100,
        neutral_benefit * 100,
        -bull_opportunity_cost * 100
    ],

    'Outperformance_Rate': [
        bear_data['Covered Call Outperformed'].mean() * 100,
        neutral_data['Covered Call Outperformed'].mean() * 100,
        bull_data['Covered Call Outperformed'].mean() * 100
    ]
})

tradeoff_table


# In[249]:


# Step 216: Strategy Payoff Summary

strategy_summary = pd.DataFrame({
    'Characteristic': [
        'Income Generation',
        'Downside Protection',
        'Upside Participation',
        'Best Market Environment',
        'Weakest Market Environment',
        'Overall Volatility',
        'Overall Max Drawdown',
        'Historical Outperformance'
    ],

    'Covered Call Finding': [
        'Generates option premium',
        'Provides partial cushion, not full protection',
        'Limited by short call',
        'Neutral / sideways markets',
        'Strong Bull markets',
        '1.93%',
        '-6.22%',
        '77.27% of cycles'
    ]
})

strategy_summary


# In[250]:


# Step 216: Strategy Payoff Summary

strategy_summary = pd.DataFrame({
    'Characteristic': [
        'Income Generation',
        'Downside Protection',
        'Upside Participation',
        'Best Market Environment',
        'Weakest Market Environment',
        'Overall Volatility',
        'Overall Max Drawdown',
        'Historical Outperformance'
    ],

    'Covered Call Finding': [
        'Generates option premium',
        'Provides partial cushion, not full protection',
        'Limited by short call',
        'Neutral / sideways markets',
        'Strong Bull markets',
        '1.93%',
        '-6.22%',
        '77.27% of cycles'
    ]
})

strategy_summary


# In[251]:


# Step 218: Volatility-Premium Correlation

vol_premium_corr = (
    backtest_df[
        ['Historical Volatility', 'Call Premium']
    ]
    .corr()
    .iloc[0, 1]
)

print(
    f"Correlation between Historical Volatility "
    f"and Call Premium: {vol_premium_corr:.4f}"
)


# In[252]:


# Step 219: Volatility vs Premium

plt.figure(figsize=(9, 6))

plt.scatter(
    backtest_df['Historical Volatility'] * 100,
    backtest_df['Call Premium']
)

plt.xlabel('Historical Volatility (%)')
plt.ylabel('Call Premium')
plt.title('Historical Volatility vs Call Premium')

plt.grid(True)

plt.tight_layout()
plt.show()


# In[253]:


# Step 220: Volatility vs Premium Yield

vol_yield_corr = (
    backtest_df[
        ['Historical Volatility', 'Premium Yield']
    ]
    .corr()
    .iloc[0, 1]
)

print(
    f"Correlation between Historical Volatility "
    f"and Premium Yield: {vol_yield_corr:.4f}"
)


# In[254]:


# Step 221: Premium Yield vs Covered Call Advantage

premium_advantage_corr = (
    backtest_df[
        ['Premium Yield', 'Covered Call Advantage']
    ]
    .corr()
    .iloc[0, 1]
)

print(
    f"Correlation between Premium Yield "
    f"and Covered Call Advantage: "
    f"{premium_advantage_corr:.4f}"
)


# In[255]:


# Step 222: Premium Yield vs Strategy Return

premium_return_corr = (
    backtest_df[
        ['Premium Yield', 'Covered Call Return']
    ]
    .corr()
    .iloc[0, 1]
)

print(
    f"Correlation between Premium Yield "
    f"and Covered Call Return: "
    f"{premium_return_corr:.4f}"
)


# In[256]:


# Step 223: Premium Relationship Summary

premium_relationships = pd.DataFrame({
    'Relationship': [
        'Volatility vs Call Premium',
        'Volatility vs Premium Yield',
        'Premium Yield vs Covered Call Advantage',
        'Premium Yield vs Covered Call Return'
    ],

    'Correlation': [
        vol_premium_corr,
        vol_yield_corr,
        premium_advantage_corr,
        premium_return_corr
    ]
})

premium_relationships


# In[257]:


# Step 224: Highest Premium Yield Cycles

highest_premium_cycles = (
    backtest_df[
        [
            'Entry Date',
            'Expiry Date',
            'Market Regime',
            'Historical Volatility',
            'Call Premium',
            'Premium Yield',
            'Covered Call Return',
            'Covered Call Advantage'
        ]
    ]
    .sort_values(
        'Premium Yield',
        ascending=False
    )
    .head(10)
)

highest_premium_cycles


# In[258]:


# Step 225: Highest Volatility Cycles

highest_volatility_cycles = (
    backtest_df[
        [
            'Entry Date',
            'Expiry Date',
            'Market Regime',
            'Historical Volatility',
            'Call Premium',
            'Premium Yield',
            'Covered Call Return',
            'Covered Call Advantage'
        ]
    ]
    .sort_values(
        'Historical Volatility',
        ascending=False
    )
    .head(10)
)

highest_volatility_cycles


# In[259]:


# Step 226: Volatility Regimes

backtest_df['Volatility Regime'] = pd.qcut(
    backtest_df['Historical Volatility'],
    q=3,
    labels=['Low Volatility', 'Medium Volatility', 'High Volatility']
)

volatility_regime_analysis = (
    backtest_df
    .groupby('Volatility Regime', observed=True)
    .agg(
        Cycles=('Historical Volatility', 'count'),

        Avg_Volatility=(
            'Historical Volatility',
            'mean'
        ),

        Avg_Call_Premium=(
            'Call Premium',
            'mean'
        ),

        Avg_Premium_Yield=(
            'Premium Yield',
            'mean'
        ),

        Avg_Covered_Call_Return=(
            'Covered Call Return',
            'mean'
        ),

        Avg_Covered_Call_Advantage=(
            'Covered Call Advantage',
            'mean'
        ),

        Outperformance_Rate=(
            'Covered Call Outperformed',
            'mean'
        )
    )
)

volatility_regime_analysis[
    'Avg_Volatility'
] *= 100

volatility_regime_analysis[
    'Avg_Premium_Yield'
] *= 100

volatility_regime_analysis[
    'Avg_Covered_Call_Return'
] *= 100

volatility_regime_analysis[
    'Avg_Covered_Call_Advantage'
] *= 100

volatility_regime_analysis[
    'Outperformance_Rate'
] *= 100

print("===== VOLATILITY REGIME ANALYSIS =====")

volatility_regime_analysis


# In[260]:


# Step 228: Premium Yield Across Volatility Regimes

plt.figure(figsize=(9, 6))

volatility_regime_analysis[
    'Avg_Premium_Yield'
].plot(
    kind='bar'
)

plt.xlabel('Volatility Regime')
plt.ylabel('Average Premium Yield (%)')
plt.title('Premium Yield Across Volatility Regimes')

plt.xticks(rotation=0)
plt.grid(axis='y')

plt.tight_layout()
plt.show()


# In[261]:


# Step 229: Covered Call Advantage Across Volatility Regimes

plt.figure(figsize=(9, 6))

volatility_regime_analysis[
    'Avg_Covered_Call_Advantage'
].plot(
    kind='bar'
)

plt.xlabel('Volatility Regime')
plt.ylabel('Average Covered Call Advantage (%)')
plt.title('Covered Call Advantage Across Volatility Regimes')

plt.xticks(rotation=0)
plt.grid(axis='y')

plt.tight_layout()
plt.show()


# In[262]:


# Step 231: Volatility -> Premium Yield Regression

import statsmodels.api as sm

X = backtest_df['Historical Volatility']
y = backtest_df['Premium Yield']

X = sm.add_constant(X)

volatility_premium_model = sm.OLS(y, X).fit()

print(volatility_premium_model.summary())


# In[263]:


# Step 232: Premium Yield -> Covered Call Advantage

X = backtest_df['Premium Yield']
y = backtest_df['Covered Call Advantage']

X = sm.add_constant(X)

premium_advantage_model = sm.OLS(y, X).fit()

print(premium_advantage_model.summary())


# In[264]:


# Step 233: Multivariate Covered Call Advantage Regression

X = backtest_df[
    [
        'Premium Yield',
        'Long NIFTY Return'
    ]
]

y = backtest_df['Covered Call Advantage']

X = sm.add_constant(X)

multivariate_model = sm.OLS(y, X).fit()

print(multivariate_model.summary())


# In[265]:


# Step 234: Full Covered Call Regression

X = backtest_df[
    [
        'Premium Yield',
        'Long NIFTY Return',
        'Historical Volatility'
    ]
]

y = backtest_df['Covered Call Advantage']

X = sm.add_constant(X)

full_model = sm.OLS(y, X).fit()

print(full_model.summary())


# In[266]:


# Step 235: Regression Model Comparison

model_comparison = pd.DataFrame({
    'Model': [
        'Volatility → Premium Yield',
        'Premium Yield → CC Advantage',
        'Premium Yield + NIFTY Return → CC Advantage',
        'Premium Yield + NIFTY Return + Volatility → CC Advantage'
    ],

    'R_squared': [
        volatility_premium_model.rsquared,
        premium_advantage_model.rsquared,
        multivariate_model.rsquared,
        full_model.rsquared
    ],

    'Adjusted_R_squared': [
        volatility_premium_model.rsquared_adj,
        premium_advantage_model.rsquared_adj,
        multivariate_model.rsquared_adj,
        full_model.rsquared_adj
    ]
})

model_comparison


# In[267]:


# Step 236: Extract Important Regression Statistics

regression_results = pd.DataFrame({
    'Model': [
        'Volatility → Premium Yield',
        'Premium Yield → CC Advantage',
        'Multivariate',
        'Full Model'
    ],

    'R_squared': [
        volatility_premium_model.rsquared,
        premium_advantage_model.rsquared,
        multivariate_model.rsquared,
        full_model.rsquared
    ],

    'Adjusted_R_squared': [
        volatility_premium_model.rsquared_adj,
        premium_advantage_model.rsquared_adj,
        multivariate_model.rsquared_adj,
        full_model.rsquared_adj
    ],

    'AIC': [
        volatility_premium_model.aic,
        premium_advantage_model.aic,
        multivariate_model.aic,
        full_model.aic
    ],

    'BIC': [
        volatility_premium_model.bic,
        premium_advantage_model.bic,
        multivariate_model.bic,
        full_model.bic
    ]
})

regression_results


# In[268]:


# Step 237: Full Model Coefficients

full_model_coefficients = pd.DataFrame({
    'Coefficient': full_model.params,
    'Std_Error': full_model.bse,
    't_statistic': full_model.tvalues,
    'p_value': full_model.pvalues
})

full_model_coefficients


# In[269]:


# Step 238: Regression Residual Diagnostics

residuals = full_model.resid

print("Residual Mean:", residuals.mean())
print("Residual Std:", residuals.std())
print("Residual Skewness:", residuals.skew())
print("Residual Kurtosis:", residuals.kurtosis())


# In[270]:


# Step 239: Premium Yield vs Covered Call Advantage

plt.figure(figsize=(9, 6))

plt.scatter(
    backtest_df['Premium Yield'] * 100,
    backtest_df['Covered Call Advantage'] * 100
)

plt.xlabel('Premium Yield (%)')
plt.ylabel('Covered Call Advantage (%)')
plt.title('Premium Yield vs Covered Call Advantage')

plt.grid(True)

plt.tight_layout()
plt.show()


# In[271]:


# Step 241: Compact Regression Results

print("===== STEP 231: VOLATILITY -> PREMIUM YIELD =====")

print(f"R-squared: {volatility_premium_model.rsquared:.6f}")
print(
    f"Volatility coefficient: "
    f"{volatility_premium_model.params['Historical Volatility']:.6f}"
)
print(
    f"Volatility p-value: "
    f"{volatility_premium_model.pvalues['Historical Volatility']:.6f}"
)

print("\n===== STEP 232: PREMIUM YIELD -> CC ADVANTAGE =====")

print(f"R-squared: {premium_advantage_model.rsquared:.6f}")
print(
    f"Premium Yield coefficient: "
    f"{premium_advantage_model.params['Premium Yield']:.6f}"
)
print(
    f"Premium Yield p-value: "
    f"{premium_advantage_model.pvalues['Premium Yield']:.6f}"
)

print("\n===== STEP 233: PREMIUM + NIFTY RETURN =====")

print(f"R-squared: {multivariate_model.rsquared:.6f}")

for variable in [
    'Premium Yield',
    'Long NIFTY Return'
]:
    print(
        f"{variable}: "
        f"coef={multivariate_model.params[variable]:.6f}, "
        f"p={multivariate_model.pvalues[variable]:.6f}"
    )

print("\n===== STEP 234: FULL MODEL =====")

print(f"R-squared: {full_model.rsquared:.6f}")
print(f"Adjusted R-squared: {full_model.rsquared_adj:.6f}")

for variable in [
    'Premium Yield',
    'Long NIFTY Return',
    'Historical Volatility'
]:
    print(
        f"{variable}: "
        f"coef={full_model.params[variable]:.6f}, "
        f"p={full_model.pvalues[variable]:.6f}"
    )


# In[272]:


# Step 242: Multicollinearity Check

correlation_matrix = backtest_df[
    [
        'Premium Yield',
        'Long NIFTY Return',
        'Historical Volatility'
    ]
].corr()

print("===== PREDICTOR CORRELATION MATRIX =====")

print(correlation_matrix)


# In[273]:


# Step 243: Variance Inflation Factor

from statsmodels.stats.outliers_influence import variance_inflation_factor

X_vif = backtest_df[
    [
        'Premium Yield',
        'Long NIFTY Return',
        'Historical Volatility'
    ]
].copy()

X_vif = sm.add_constant(X_vif)

vif_table = pd.DataFrame({
    'Variable': X_vif.columns,
    'VIF': [
        variance_inflation_factor(
            X_vif.values,
            i
        )
        for i in range(X_vif.shape[1])
    ]
})

vif_table


# In[274]:


# Step 244: Identify Potentially Influential Observations

influence = full_model.get_influence()

cooks_distance = influence.cooks_distance[0]

influence_df = backtest_df[
    [
        'Entry Date',
        'Expiry Date',
        'Market Regime',
        'Premium Yield',
        'Long NIFTY Return',
        'Covered Call Advantage'
    ]
].copy()

influence_df['Cooks Distance'] = cooks_distance

influence_df = influence_df.sort_values(
    'Cooks Distance',
    ascending=False
)

print("===== MOST INFLUENTIAL OBSERVATIONS =====")

influence_df.head(5)


# In[275]:


# Step 246: Bootstrap Regression Coefficients

np.random.seed(42)

n_bootstrap = 5000

bootstrap_coefficients = []

for _ in range(n_bootstrap):

    sample = backtest_df.sample(
        n=len(backtest_df),
        replace=True
    )

    X = sample[
        [
            'Premium Yield',
            'Long NIFTY Return',
            'Historical Volatility'
        ]
    ]

    y = sample['Covered Call Advantage']

    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()

    bootstrap_coefficients.append([
        model.params['Premium Yield'],
        model.params['Long NIFTY Return'],
        model.params['Historical Volatility']
    ])

bootstrap_df = pd.DataFrame(
    bootstrap_coefficients,
    columns=[
        'Premium_Yield',
        'NIFTY_Return',
        'Historical_Volatility'
    ]
)

bootstrap_summary = bootstrap_df.quantile(
    [0.025, 0.50, 0.975]
)

print("===== BOOTSTRAP 95% CONFIDENCE INTERVALS =====")

bootstrap_summary


# In[276]:


# Step 247A: Premium-Based Model

X = backtest_df[
    [
        'Premium Yield',
        'Long NIFTY Return'
    ]
]

y = backtest_df['Covered Call Advantage']

X = sm.add_constant(X)

premium_model_clean = sm.OLS(y, X).fit()

print("===== PREMIUM-BASED MODEL =====")

print(f"R-squared: {premium_model_clean.rsquared:.6f}")
print(
    f"Adjusted R-squared: "
    f"{premium_model_clean.rsquared_adj:.6f}"
)

for variable in [
    'Premium Yield',
    'Long NIFTY Return'
]:
    print(
        f"{variable}: "
        f"coef={premium_model_clean.params[variable]:.6f}, "
        f"p={premium_model_clean.pvalues[variable]:.6f}"
    )


# In[277]:


# Step 247B: Volatility-Based Model

X = backtest_df[
    [
        'Historical Volatility',
        'Long NIFTY Return'
    ]
]

y = backtest_df['Covered Call Advantage']

X = sm.add_constant(X)

volatility_model_clean = sm.OLS(y, X).fit()

print("===== VOLATILITY-BASED MODEL =====")

print(f"R-squared: {volatility_model_clean.rsquared:.6f}")
print(
    f"Adjusted R-squared: "
    f"{volatility_model_clean.rsquared_adj:.6f}"
)

for variable in [
    'Historical Volatility',
    'Long NIFTY Return'
]:
    print(
        f"{variable}: "
        f"coef={volatility_model_clean.params[variable]:.6f}, "
        f"p={volatility_model_clean.pvalues[variable]:.6f}"
    )


# In[278]:


# Step 248: Compare Clean Models

clean_model_comparison = pd.DataFrame({
    'Model': [
        'Premium Yield + NIFTY Return',
        'Historical Volatility + NIFTY Return'
    ],

    'R_squared': [
        premium_model_clean.rsquared,
        volatility_model_clean.rsquared
    ],

    'Adjusted_R_squared': [
        premium_model_clean.rsquared_adj,
        volatility_model_clean.rsquared_adj
    ],

    'AIC': [
        premium_model_clean.aic,
        volatility_model_clean.aic
    ],

    'BIC': [
        premium_model_clean.bic,
        volatility_model_clean.bic
    ]
})

clean_model_comparison


# In[279]:


# Step 249: Regime-Based Return Analysis

regime_return_analysis = (
    backtest_df
    .groupby('Market Regime')
    .agg(
        Cycles=('Market Regime', 'count'),

        Avg_NIFTY_Return=(
            'Long NIFTY Return',
            'mean'
        ),

        Avg_Covered_Call_Return=(
            'Covered Call Return',
            'mean'
        ),

        Avg_Covered_Call_Advantage=(
            'Covered Call Advantage',
            'mean'
        ),

        Avg_Premium_Yield=(
            'Premium Yield',
            'mean'
        ),

        Outperformance_Rate=(
            'Covered Call Outperformed',
            'mean'
        )
    )
)

regime_return_analysis[
    'Avg_NIFTY_Return'
] *= 100

regime_return_analysis[
    'Avg_Covered_Call_Return'
] *= 100

regime_return_analysis[
    'Avg_Covered_Call_Advantage'
] *= 100

regime_return_analysis[
    'Avg_Premium_Yield'
] *= 100

regime_return_analysis[
    'Outperformance_Rate'
] *= 100

print("===== REGIME RETURN ANALYSIS =====")

regime_return_analysis


# In[280]:


# Step 250: Regime-Based Volatility

regime_volatility = (
    backtest_df
    .groupby('Market Regime')
    .agg(
        NIFTY_Volatility=(
            'Long NIFTY Return',
            'std'
        ),

        Covered_Call_Volatility=(
            'Covered Call Return',
            'std'
        ),

        NIFTY_Worst_Return=(
            'Long NIFTY Return',
            'min'
        ),

        Covered_Call_Worst_Return=(
            'Covered Call Return',
            'min'
        ),

        NIFTY_Best_Return=(
            'Long NIFTY Return',
            'max'
        ),

        Covered_Call_Best_Return=(
            'Covered Call Return',
            'max'
        )
    )
)

regime_volatility *= 100

print("===== REGIME RISK ANALYSIS =====")

regime_volatility


# In[281]:


# Step 251: Bull Market Analysis

bull_df = backtest_df[
    backtest_df['Market Regime'] == 'Bull'
]

print("===== BULL MARKET =====")

print(
    f"Cycles: {len(bull_df)}"
)

print(
    f"Average NIFTY Return: "
    f"{bull_df['Long NIFTY Return'].mean() * 100:.4f}%"
)

print(
    f"Average Covered Call Return: "
    f"{bull_df['Covered Call Return'].mean() * 100:.4f}%"
)

print(
    f"Average Covered Call Advantage: "
    f"{bull_df['Covered Call Advantage'].mean() * 100:.4f}%"
)

print(
    f"Outperformance Rate: "
    f"{bull_df['Covered Call Outperformed'].mean() * 100:.2f}%"
)


# In[282]:


# Step 252: Bear Market Analysis

bear_df = backtest_df[
    backtest_df['Market Regime'] == 'Bear'
]

print("===== BEAR MARKET =====")

print(
    f"Cycles: {len(bear_df)}"
)

print(
    f"Average NIFTY Return: "
    f"{bear_df['Long NIFTY Return'].mean() * 100:.4f}%"
)

print(
    f"Average Covered Call Return: "
    f"{bear_df['Covered Call Return'].mean() * 100:.4f}%"
)

print(
    f"Average Covered Call Advantage: "
    f"{bear_df['Covered Call Advantage'].mean() * 100:.4f}%"
)

print(
    f"Outperformance Rate: "
    f"{bear_df['Covered Call Outperformed'].mean() * 100:.2f}%"
)


# In[283]:


# Step 253: Neutral Market Analysis

neutral_df = backtest_df[
    backtest_df['Market Regime'] == 'Neutral'
]

print("===== NEUTRAL MARKET =====")

print(
    f"Cycles: {len(neutral_df)}"
)

print(
    f"Average NIFTY Return: "
    f"{neutral_df['Long NIFTY Return'].mean() * 100:.4f}%"
)

print(
    f"Average Covered Call Return: "
    f"{neutral_df['Covered Call Return'].mean() * 100:.4f}%"
)

print(
    f"Average Covered Call Advantage: "
    f"{neutral_df['Covered Call Advantage'].mean() * 100:.4f}%"
)

print(
    f"Outperformance Rate: "
    f"{neutral_df['Covered Call Outperformed'].mean() * 100:.2f}%"
)


# In[284]:


# Step 254: Regime Performance Comparison

regime_plot = regime_return_analysis[
    [
        'Avg_NIFTY_Return',
        'Avg_Covered_Call_Return'
    ]
]

regime_plot.plot(
    kind='bar',
    figsize=(10, 6)
)

plt.xlabel('Market Regime')
plt.ylabel('Average Return (%)')
plt.title('NIFTY vs Covered Call Performance by Market Regime')

plt.xticks(rotation=0)
plt.grid(axis='y')

plt.tight_layout()
plt.show()


# In[285]:


# Step 255: Covered Call Advantage by Regime

regime_return_analysis[
    'Avg_Covered_Call_Advantage'
].plot(
    kind='bar',
    figsize=(9, 6)
)

plt.axhline(
    0,
    linewidth=1
)

plt.xlabel('Market Regime')
plt.ylabel('Average Covered Call Advantage (%)')
plt.title('Covered Call Advantage Across Market Regimes')

plt.xticks(rotation=0)
plt.grid(axis='y')

plt.tight_layout()
plt.show()


# In[286]:


# Step 256: Regime-Based Volatility Reduction

regime_volatility['Volatility_Reduction'] = (
    (
        regime_volatility['NIFTY_Volatility']
        -
        regime_volatility['Covered_Call_Volatility']
    )
    /
    regime_volatility['NIFTY_Volatility']
) * 100

print("===== REGIME VOLATILITY REDUCTION =====")

print(
    regime_volatility[
        ['NIFTY_Volatility',
         'Covered_Call_Volatility',
         'Volatility_Reduction']
    ]
)


# In[287]:


# Step 257: Return Difference

backtest_df['Return Difference'] = (
    backtest_df['Covered Call Return']
    -
    backtest_df['Long NIFTY Return']
)

print("===== RETURN DIFFERENCE =====")

print(
    backtest_df[
        [
            'Entry Date',
            'Market Regime',
            'Long NIFTY Return',
            'Covered Call Return',
            'Return Difference'
        ]
    ].head()
)

print(
    "\nAverage Return Difference:",
    backtest_df['Return Difference'].mean()
)


# In[288]:


# Step 258: Overall Paired t-Test

from scipy.stats import ttest_rel

t_stat, p_value = ttest_rel(
    backtest_df['Covered Call Return'],
    backtest_df['Long NIFTY Return']
)

print("===== OVERALL PAIRED T-TEST =====")

print(f"t-statistic: {t_stat:.6f}")
print(f"p-value: {p_value:.6f}")


# In[289]:


# Step 259: Wilcoxon Signed-Rank Test

from scipy.stats import wilcoxon

wilcoxon_stat, wilcoxon_p = wilcoxon(
    backtest_df['Return Difference']
)

print("===== WILCOXON SIGNED-RANK TEST =====")

print(f"Statistic: {wilcoxon_stat:.6f}")
print(f"p-value: {wilcoxon_p:.6f}")


# In[290]:


# Step 260: Regime-Level Paired Tests

regime_tests = []

for regime in ['Bull', 'Bear', 'Neutral']:

    temp = backtest_df[
        backtest_df['Market Regime'] == regime
    ]

    t_stat, p_value = ttest_rel(
        temp['Covered Call Return'],
        temp['Long NIFTY Return']
    )

    regime_tests.append({
        'Market Regime': regime,
        'Cycles': len(temp),
        'Mean NIFTY Return':
            temp['Long NIFTY Return'].mean(),
        'Mean Covered Call Return':
            temp['Covered Call Return'].mean(),
        'Mean Advantage':
            temp['Covered Call Advantage'].mean(),
        't-statistic': t_stat,
        'p-value': p_value
    })

regime_test_df = pd.DataFrame(regime_tests)

regime_test_df[
    [
        'Mean NIFTY Return',
        'Mean Covered Call Return',
        'Mean Advantage'
    ]
] *= 100

print("===== REGIME PAIRED T-TEST =====")

regime_test_df


# In[291]:


# Step 261: Regime Wilcoxon Tests

wilcoxon_regime_tests = []

for regime in ['Bull', 'Bear', 'Neutral']:

    temp = backtest_df[
        backtest_df['Market Regime'] == regime
    ]

    try:

        statistic, p_value = wilcoxon(
            temp['Return Difference']
        )

    except ValueError:

        statistic = np.nan
        p_value = np.nan

    wilcoxon_regime_tests.append({
        'Market Regime': regime,
        'Cycles': len(temp),
        'Wilcoxon Statistic': statistic,
        'Wilcoxon p-value': p_value
    })

wilcoxon_regime_df = pd.DataFrame(
    wilcoxon_regime_tests
)

print("===== REGIME WILCOXON TEST =====")

wilcoxon_regime_df


# In[292]:


# Step 262: Confidence Interval for Covered Call Advantage

from scipy.stats import t

advantage = backtest_df['Covered Call Advantage']

n = len(advantage)

mean_advantage = advantage.mean()
std_advantage = advantage.std(ddof=1)

standard_error = std_advantage / np.sqrt(n)

critical_value = t.ppf(
    0.975,
    df=n-1
)

ci_lower = (
    mean_advantage
    -
    critical_value * standard_error
)

ci_upper = (
    mean_advantage
    +
    critical_value * standard_error
)

print("===== 95% CI FOR COVERED CALL ADVANTAGE =====")

print(
    f"Mean Advantage: "
    f"{mean_advantage * 100:.4f}%"
)

print(
    f"95% CI: "
    f"{ci_lower * 100:.4f}% "
    f"to "
    f"{ci_upper * 100:.4f}%"
)


# In[293]:


# Step 263: Cohen's d Effect Size

cohens_d = (
    advantage.mean()
    /
    advantage.std(ddof=1)
)

print("===== EFFECT SIZE =====")

print(
    f"Cohen's d: {cohens_d:.6f}"
)


# In[294]:


# Step 264: Bootstrap Covered Call Advantage

np.random.seed(42)

bootstrap_advantage = []

for _ in range(5000):

    sample = backtest_df[
        'Covered Call Advantage'
    ].sample(
        n=len(backtest_df),
        replace=True
    )

    bootstrap_advantage.append(
        sample.mean()
    )

bootstrap_advantage = np.array(
    bootstrap_advantage
)

bootstrap_ci = np.percentile(
    bootstrap_advantage,
    [2.5, 97.5]
)

print("===== BOOTSTRAP ADVANTAGE =====")

print(
    f"Bootstrap Mean: "
    f"{bootstrap_advantage.mean() * 100:.4f}%"
)

print(
    f"95% Bootstrap CI: "
    f"{bootstrap_ci[0] * 100:.4f}% "
    f"to "
    f"{bootstrap_ci[1] * 100:.4f}%"
)


# In[295]:


# Step 265: Statistical Summary

statistical_summary = pd.DataFrame({
    'Metric': [
        'Number of Cycles',
        'Mean Covered Call Advantage',
        'Paired t-test p-value',
        'Wilcoxon p-value',
        '95% CI Lower',
        '95% CI Upper',
        "Cohen's d",
        'Bootstrap CI Lower',
        'Bootstrap CI Upper'
    ],

    'Value': [
        len(backtest_df),
        mean_advantage * 100,
        p_value,
        wilcoxon_p,
        ci_lower * 100,
        ci_upper * 100,
        cohens_d,
        bootstrap_ci[0] * 100,
        bootstrap_ci[1] * 100
    ]
})

print("===== FINAL STATISTICAL SUMMARY =====")

statistical_summary


# In[296]:


# Step 266: Contribution to Overall Covered Call Advantage

regime_contribution = (
    backtest_df
    .groupby('Market Regime')
    .agg(
        Cycles=('Market Regime', 'count'),
        Total_Advantage=('Covered Call Advantage', 'sum'),
        Average_Advantage=('Covered Call Advantage', 'mean')
    )
)

regime_contribution['Contribution_%'] = (
    regime_contribution['Total_Advantage']
    / backtest_df['Covered Call Advantage'].sum()
    * 100
)

print("===== REGIME CONTRIBUTION TO TOTAL ADVANTAGE =====")

regime_contribution


# In[297]:


# Step 267: Sensitivity to Extreme Bull Cycle

extreme_index = 7

sensitivity_df = backtest_df.drop(
    index=extreme_index
)

print("===== SENSITIVITY WITHOUT EXTREME BULL CYCLE =====")

print(
    f"Original cycles: {len(backtest_df)}"
)

print(
    f"Sensitivity cycles: {len(sensitivity_df)}"
)

print(
    f"Original average advantage: "
    f"{backtest_df['Covered Call Advantage'].mean() * 100:.4f}%"
)

print(
    f"Adjusted average advantage: "
    f"{sensitivity_df['Covered Call Advantage'].mean() * 100:.4f}%"
)

print(
    f"Original final wealth: "
    f"{backtest_df['Covered Call Wealth'].iloc[-1]:.6f}"
)

print(
    f"Adjusted final wealth: "
    f"{sensitivity_df['Covered Call Wealth'].iloc[-1]:.6f}"
)


# In[298]:


# Step 268: Wins and Losses

wins = (
    backtest_df['Covered Call Advantage'] > 0
).sum()

losses = (
    backtest_df['Covered Call Advantage'] < 0
).sum()

ties = (
    backtest_df['Covered Call Advantage'] == 0
).sum()

print("===== COVERED CALL WINS / LOSSES =====")

print(f"Wins: {wins}")
print(f"Losses: {losses}")
print(f"Ties: {ties}")

print(
    f"Win Rate: "
    f"{wins / len(backtest_df) * 100:.2f}%"
)


# In[299]:


# Step 269: Average Win / Average Loss

positive_advantages = backtest_df.loc[
    backtest_df['Covered Call Advantage'] > 0,
    'Covered Call Advantage'
]

negative_advantages = backtest_df.loc[
    backtest_df['Covered Call Advantage'] < 0,
    'Covered Call Advantage'
]

average_gain = positive_advantages.mean()

average_loss = negative_advantages.mean()

print("===== GAIN / LOSS ANALYSIS =====")

print(
    f"Average winning advantage: "
    f"{average_gain * 100:.4f}%"
)

print(
    f"Average losing advantage: "
    f"{average_loss * 100:.4f}%"
)

print(
    f"Gain/Loss ratio: "
    f"{abs(average_gain / average_loss):.4f}"
)


# In[300]:


# Step 270: Profit Factor

total_gains = positive_advantages.sum()

total_losses = abs(
    negative_advantages.sum()
)

profit_factor = (
    total_gains / total_losses
)

print("===== PROFIT FACTOR =====")

print(
    f"Profit Factor: {profit_factor:.4f}"
)


# In[301]:


# Step 271: Maximum Consecutive Losses

is_loss = (
    backtest_df['Covered Call Advantage'] < 0
)

max_consecutive_losses = 0
current_losses = 0

for loss in is_loss:

    if loss:
        current_losses += 1
        max_consecutive_losses = max(
            max_consecutive_losses,
            current_losses
        )
    else:
        current_losses = 0

print("===== CONSECUTIVE LOSSES =====")

print(
    f"Maximum consecutive losing cycles: "
    f"{max_consecutive_losses}"
)


# In[302]:


# Step 272: Premium Yield Statistics

premium_stats = pd.Series({
    'Mean Premium Yield':
        backtest_df['Premium Yield'].mean(),

    'Median Premium Yield':
        backtest_df['Premium Yield'].median(),

    'Minimum Premium Yield':
        backtest_df['Premium Yield'].min(),

    'Maximum Premium Yield':
        backtest_df['Premium Yield'].max(),

    'Std Dev Premium Yield':
        backtest_df['Premium Yield'].std()
})

print("===== PREMIUM YIELD STATISTICS =====")

print(
    premium_stats * 100
)


# In[303]:


# Step 273: Final Strategy Metrics

final_strategy_metrics = pd.DataFrame({
    'Metric': [
        'Number of Cycles',
        'Average NIFTY Return',
        'Average Covered Call Return',
        'Average Covered Call Advantage',
        'NIFTY Volatility',
        'Covered Call Volatility',
        'NIFTY Maximum Drawdown',
        'Covered Call Maximum Drawdown',
        'NIFTY Sharpe Ratio',
        'Covered Call Sharpe Ratio',
        'Covered Call Win Rate',
        'Average Winning Advantage',
        'Average Losing Advantage',
        'Profit Factor',
        'Maximum Consecutive Losses'
    ],

    'Value': [
        len(backtest_df),

        backtest_df['Long NIFTY Return'].mean() * 100,

        backtest_df['Covered Call Return'].mean() * 100,

        backtest_df['Covered Call Advantage'].mean() * 100,

        backtest_df['Long NIFTY Return'].std() * 100,

        backtest_df['Covered Call Return'].std() * 100,

        (backtest_df['Long NIFTY Wealth'] /
         backtest_df['Long NIFTY Wealth'].cummax() - 1).min() * 100,

        (backtest_df['Covered Call Wealth'] /
         backtest_df['Covered Call Wealth'].cummax() - 1).min() * 100,

        backtest_df['Long NIFTY Return'].mean()
        / backtest_df['Long NIFTY Return'].std(),

        backtest_df['Covered Call Return'].mean()
        / backtest_df['Covered Call Return'].std(),

        wins / len(backtest_df) * 100,

        average_gain * 100,

        average_loss * 100,

        profit_factor,

        max_consecutive_losses
    ]
})

print("===== FINAL STRATEGY METRICS =====")

final_strategy_metrics


# In[304]:


# Step 273: Final Strategy Metrics

final_strategy_metrics = pd.DataFrame({
    'Metric': [
        'Number of Cycles',
        'Average NIFTY Return',
        'Average Covered Call Return',
        'Average Covered Call Advantage',
        'NIFTY Volatility',
        'Covered Call Volatility',
        'NIFTY Maximum Drawdown',
        'Covered Call Maximum Drawdown',
        'NIFTY Sharpe Ratio',
        'Covered Call Sharpe Ratio',
        'Covered Call Win Rate',
        'Average Winning Advantage',
        'Average Losing Advantage',
        'Profit Factor',
        'Maximum Consecutive Losses'
    ],

    'Value': [
        len(backtest_df),

        backtest_df['Long NIFTY Return'].mean() * 100,

        backtest_df['Covered Call Return'].mean() * 100,

        backtest_df['Covered Call Advantage'].mean() * 100,

        backtest_df['Long NIFTY Return'].std() * 100,

        backtest_df['Covered Call Return'].std() * 100,

        (backtest_df['Long NIFTY Wealth'] /
         backtest_df['Long NIFTY Wealth'].cummax() - 1).min() * 100,

        (backtest_df['Covered Call Wealth'] /
         backtest_df['Covered Call Wealth'].cummax() - 1).min() * 100,

        backtest_df['Long NIFTY Return'].mean()
        / backtest_df['Long NIFTY Return'].std(),

        backtest_df['Covered Call Return'].mean()
        / backtest_df['Covered Call Return'].std(),

        wins / len(backtest_df) * 100,

        average_gain * 100,

        average_loss * 100,

        profit_factor,

        max_consecutive_losses
    ]
})

print("===== FINAL STRATEGY METRICS =====")

final_strategy_metrics


# In[305]:


# Step 275: Save Final Backtest Dataset

backtest_df.to_csv(
    'NIFTY_Covered_Call_Final_Backtest.csv',
    index=False
)

print(
    "Final backtest dataset saved successfully."
)


# In[306]:


# Step 276: Master Results Table

master_results = pd.DataFrame({
    'Metric': [
        'Backtest Cycles',
        'Average NIFTY Return (%)',
        'Average Covered Call Return (%)',
        'Average Covered Call Advantage (%)',
        'NIFTY Volatility (%)',
        'Covered Call Volatility (%)',
        'Volatility Reduction (%)',
        'NIFTY Maximum Drawdown (%)',
        'Covered Call Maximum Drawdown (%)',
        'NIFTY Sharpe Ratio',
        'Covered Call Sharpe Ratio',
        'Covered Call Outperformance Rate (%)',
        'Average Premium Yield (%)',
        'Profit Factor'
    ],

    'Value': [
        len(backtest_df),

        backtest_df['Long NIFTY Return'].mean() * 100,

        backtest_df['Covered Call Return'].mean() * 100,

        backtest_df['Covered Call Advantage'].mean() * 100,

        backtest_df['Long NIFTY Return'].std() * 100,

        backtest_df['Covered Call Return'].std() * 100,

        (
            1 -
            backtest_df['Covered Call Return'].std()
            /
            backtest_df['Long NIFTY Return'].std()
        ) * 100,

        (
            backtest_df['Long NIFTY Wealth']
            /
            backtest_df['Long NIFTY Wealth'].cummax()
            - 1
        ).min() * 100,

        (
            backtest_df['Covered Call Wealth']
            /
            backtest_df['Covered Call Wealth'].cummax()
            - 1
        ).min() * 100,

        backtest_df['Long NIFTY Return'].mean()
        /
        backtest_df['Long NIFTY Return'].std(),

        backtest_df['Covered Call Return'].mean()
        /
        backtest_df['Covered Call Return'].std(),

        backtest_df['Covered Call Outperformed'].mean() * 100,

        backtest_df['Premium Yield'].mean() * 100,

        profit_factor
    ]
})

print("===== MASTER RESULTS =====")

master_results


# In[307]:


# Step 277: Final Regime Table

final_regime_table = (
    backtest_df
    .groupby('Market Regime')
    .agg(
        Cycles=('Market Regime', 'count'),

        NIFTY_Return=(
            'Long NIFTY Return',
            'mean'
        ),

        Covered_Call_Return=(
            'Covered Call Return',
            'mean'
        ),

        Covered_Call_Advantage=(
            'Covered Call Advantage',
            'mean'
        ),

        Premium_Yield=(
            'Premium Yield',
            'mean'
        ),

        Outperformance_Rate=(
            'Covered Call Outperformed',
            'mean'
        )
    )
)

final_regime_table[
    [
        'NIFTY_Return',
        'Covered_Call_Return',
        'Covered_Call_Advantage',
        'Premium_Yield',
        'Outperformance_Rate'
    ]
] *= 100

print("===== FINAL REGIME TABLE =====")

final_regime_table


# In[308]:


# Step 278: Final Statistical Tests

overall_t_stat, overall_t_p = ttest_rel(
    backtest_df['Covered Call Return'],
    backtest_df['Long NIFTY Return']
)

overall_w_stat, overall_w_p = wilcoxon(
    backtest_df['Return Difference']
)

statistical_results = pd.DataFrame({
    'Test': [
        'Paired t-test',
        'Wilcoxon signed-rank test',
        'Cohen\'s d'
    ],

    'Statistic': [
        overall_t_stat,
        overall_w_stat,
        cohens_d
    ],

    'p-value': [
        overall_t_p,
        overall_w_p,
        np.nan
    ]
})

print("===== FINAL STATISTICAL TESTS =====")

statistical_results


# In[309]:


# Step 279: Final Wealth Comparison

nifty_final_wealth = (
    backtest_df['Long NIFTY Wealth'].iloc[-1]
)

cc_final_wealth = (
    backtest_df['Covered Call Wealth'].iloc[-1]
)

wealth_difference = (
    cc_final_wealth - nifty_final_wealth
)

relative_improvement = (
    wealth_difference
    /
    nifty_final_wealth
) * 100

print("===== FINAL WEALTH COMPARISON =====")

print(
    f"NIFTY final wealth: "
    f"{nifty_final_wealth:.6f}"
)

print(
    f"Covered Call final wealth: "
    f"{cc_final_wealth:.6f}"
)

print(
    f"Absolute wealth difference: "
    f"{wealth_difference:.6f}"
)

print(
    f"Relative improvement: "
    f"{relative_improvement:.4f}%"
)


# In[310]:


# Step 280: Wealth Comparison

plt.figure(figsize=(12, 6))

plt.plot(
    backtest_df['Entry Date'],
    backtest_df['Long NIFTY Wealth'],
    label='NIFTY'
)

plt.plot(
    backtest_df['Entry Date'],
    backtest_df['Covered Call Wealth'],
    label='Covered Call'
)

plt.xlabel('Date')
plt.ylabel('Growth of ₹1')
plt.title('NIFTY vs Covered Call Wealth')

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# In[311]:


# Step 281: Drawdown Comparison

nifty_drawdown = (
    backtest_df['Long NIFTY Wealth']
    /
    backtest_df['Long NIFTY Wealth'].cummax()
    - 1
)

cc_drawdown = (
    backtest_df['Covered Call Wealth']
    /
    backtest_df['Covered Call Wealth'].cummax()
    - 1
)

plt.figure(figsize=(12, 6))

plt.plot(
    backtest_df['Entry Date'],
    nifty_drawdown * 100,
    label='NIFTY'
)

plt.plot(
    backtest_df['Entry Date'],
    cc_drawdown * 100,
    label='Covered Call'
)

plt.xlabel('Date')
plt.ylabel('Drawdown (%)')
plt.title('NIFTY vs Covered Call Drawdown')

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# In[312]:


# Step 282: Premium Yield vs NIFTY Return

plt.figure(figsize=(9, 6))

plt.scatter(
    backtest_df['Long NIFTY Return'] * 100,
    backtest_df['Premium Yield'] * 100
)

plt.axvline(
    0,
    linewidth=1
)

plt.xlabel('NIFTY Return (%)')
plt.ylabel('Premium Yield (%)')
plt.title('Premium Yield vs NIFTY Return')

plt.grid(True)

plt.tight_layout()
plt.show()


# In[313]:


# Step 283: Covered Call Advantage vs NIFTY Return

plt.figure(figsize=(9, 6))

plt.scatter(
    backtest_df['Long NIFTY Return'] * 100,
    backtest_df['Covered Call Advantage'] * 100
)

plt.axhline(
    0,
    linewidth=1
)

plt.axvline(
    0,
    linewidth=1
)

plt.xlabel('NIFTY Return (%)')
plt.ylabel('Covered Call Advantage (%)')
plt.title('Covered Call Advantage vs NIFTY Return')

plt.grid(True)

plt.tight_layout()
plt.show()


# In[314]:


# Step 284: Covered Call Advantage vs Historical Volatility

plt.figure(figsize=(9, 6))

plt.scatter(
    backtest_df['Historical Volatility'] * 100,
    backtest_df['Covered Call Advantage'] * 100
)

plt.axhline(
    0,
    linewidth=1
)

plt.xlabel('Historical Volatility (%)')
plt.ylabel('Covered Call Advantage (%)')
plt.title('Covered Call Advantage vs Historical Volatility')

plt.grid(True)

plt.tight_layout()
plt.show()


# In[315]:


get_ipython().run_line_magic('pip', 'install openpyxl')


# In[316]:


# Step 285: Export Final Analytical Tables

with pd.ExcelWriter(
    'NIFTY_Covered_Call_Final_Results.xlsx'
) as writer:

    master_results.to_excel(
        writer,
        sheet_name='Master Results',
        index=False
    )

    final_regime_table.to_excel(
        writer,
        sheet_name='Regime Analysis'
    )

    statistical_results.to_excel(
        writer,
        sheet_name='Statistical Tests',
        index=False
    )

    regime_contribution.to_excel(
        writer,
        sheet_name='Regime Contribution'
    )

print("Final analytical results exported successfully.")


# In[317]:


# Step 286: Final Research Summary

print("=" * 70)
print("NIFTY COVERED CALL STRATEGY — FINAL RESEARCH SUMMARY")
print("=" * 70)

print("\n--- BACKTEST ---")
print(f"Number of cycles: {len(backtest_df)}")

print("\n--- RETURNS ---")
print(
    f"Average NIFTY Return: "
    f"{backtest_df['Long NIFTY Return'].mean() * 100:.4f}%"
)

print(
    f"Average Covered Call Return: "
    f"{backtest_df['Covered Call Return'].mean() * 100:.4f}%"
)

print(
    f"Average Covered Call Advantage: "
    f"{backtest_df['Covered Call Advantage'].mean() * 100:.4f}%"
)

print("\n--- RISK ---")

nifty_vol = backtest_df['Long NIFTY Return'].std()
cc_vol = backtest_df['Covered Call Return'].std()

print(f"NIFTY Volatility: {nifty_vol * 100:.4f}%")
print(f"Covered Call Volatility: {cc_vol * 100:.4f}%")

print(
    f"Volatility Reduction: "
    f"{(1 - cc_vol / nifty_vol) * 100:.4f}%"
)

nifty_dd = (
    backtest_df['Long NIFTY Wealth']
    / backtest_df['Long NIFTY Wealth'].cummax()
    - 1
).min()

cc_dd = (
    backtest_df['Covered Call Wealth']
    / backtest_df['Covered Call Wealth'].cummax()
    - 1
).min()

print(f"NIFTY Maximum Drawdown: {nifty_dd * 100:.4f}%")
print(f"Covered Call Maximum Drawdown: {cc_dd * 100:.4f}%")

print("\n--- SHARPE RATIO ---")

nifty_sharpe = (
    backtest_df['Long NIFTY Return'].mean()
    / nifty_vol
)

cc_sharpe = (
    backtest_df['Covered Call Return'].mean()
    / cc_vol
)

print(f"NIFTY Sharpe: {nifty_sharpe:.4f}")
print(f"Covered Call Sharpe: {cc_sharpe:.4f}")

print("\n--- OUTPERFORMANCE ---")

win_rate = (
    backtest_df['Covered Call Outperformed'].mean()
    * 100
)

print(
    f"Covered Call Outperformance Rate: "
    f"{win_rate:.2f}%"
)

print("\n--- WEALTH ---")

print(
    f"Final NIFTY Wealth: "
    f"{backtest_df['Long NIFTY Wealth'].iloc[-1]:.6f}"
)

print(
    f"Final Covered Call Wealth: "
    f"{backtest_df['Covered Call Wealth'].iloc[-1]:.6f}"
)

print("\n--- MARKET REGIMES ---")

for regime in ['Bull', 'Bear', 'Neutral']:

    temp = backtest_df[
        backtest_df['Market Regime'] == regime
    ]

    print(f"\n{regime}:")
    print(f"  Cycles: {len(temp)}")
    print(
        f"  NIFTY Return: "
        f"{temp['Long NIFTY Return'].mean() * 100:.4f}%"
    )
    print(
        f"  Covered Call Return: "
        f"{temp['Covered Call Return'].mean() * 100:.4f}%"
    )
    print(
        f"  Advantage: "
        f"{temp['Covered Call Advantage'].mean() * 100:.4f}%"
    )
    print(
        f"  Outperformance: "
        f"{temp['Covered Call Outperformed'].mean() * 100:.2f}%"
    )

print("\n" + "=" * 70)
print("END OF FINAL RESEARCH SUMMARY")
print("=" * 70)


# In[318]:


# Step 288: Final Statistical Validation

from scipy.stats import ttest_rel, wilcoxon

differences = (
    backtest_df['Covered Call Return']
    - backtest_df['Long NIFTY Return']
)

t_stat, t_p = ttest_rel(
    backtest_df['Covered Call Return'],
    backtest_df['Long NIFTY Return']
)

w_stat, w_p = wilcoxon(differences)

print("===== FINAL STATISTICAL VALIDATION =====")

print(f"Mean Advantage: {differences.mean() * 100:.4f}%")

print(f"\nPaired t-test")
print(f"t-statistic: {t_stat:.6f}")
print(f"p-value: {t_p:.6f}")

print(f"\nWilcoxon Signed-Rank Test")
print(f"Statistic: {w_stat:.6f}")
print(f"p-value: {w_p:.6f}")

print("\nConclusion at 5% significance level:")

if t_p < 0.05:
    print("Paired t-test: Significant difference")
else:
    print("Paired t-test: Not statistically significant")

if w_p < 0.05:
    print("Wilcoxon test: Significant difference")
else:
    print("Wilcoxon test: Not statistically significant")


# # FINAL PROJECT OUTPUTS
# 
# This section contains the final results, tables and figures used in the project report.
# 
# ## 1. Option Pricing
# ## 2. Greeks Analysis
# ## 3. Scenario Analysis
# ## 4. Monte Carlo Simulation
# ## 5. VaR and Expected Shortfall
# ## 6. Covered Call Backtest
# ## 7. Market Regime Analysis
# ## 8. Regression Analysis
# ## 9. Statistical Validation
# ## 10. Final Performance Comparison

# In[319]:


# FINAL PERFORMANCE COMPARISON

final_performance = pd.DataFrame({
    'Metric': [
        'Average Return',
        'Volatility',
        'Maximum Drawdown',
        'Sharpe Ratio',
        'Final Wealth'
    ],

    'NIFTY': [
        backtest_df['Long NIFTY Return'].mean(),
        backtest_df['Long NIFTY Return'].std(),
        (
            backtest_df['Long NIFTY Wealth']
            / backtest_df['Long NIFTY Wealth'].cummax()
            - 1
        ).min(),
        backtest_df['Long NIFTY Return'].mean()
        / backtest_df['Long NIFTY Return'].std(),
        backtest_df['Long NIFTY Wealth'].iloc[-1]
    ],

    'Covered Call': [
        backtest_df['Covered Call Return'].mean(),
        backtest_df['Covered Call Return'].std(),
        (
            backtest_df['Covered Call Wealth']
            / backtest_df['Covered Call Wealth'].cummax()
            - 1
        ).min(),
        backtest_df['Covered Call Return'].mean()
        / backtest_df['Covered Call Return'].std(),
        backtest_df['Covered Call Wealth'].iloc[-1]
    ]
})

final_performance


# In[320]:


# FINAL MARKET REGIME COMPARISON

regime_final = (
    backtest_df
    .groupby('Market Regime')
    .agg(
        Cycles=('Market Regime', 'count'),
        NIFTY_Return=('Long NIFTY Return', 'mean'),
        Covered_Call_Return=('Covered Call Return', 'mean'),
        Advantage=('Covered Call Advantage', 'mean'),
        Outperformance=('Covered Call Outperformed', 'mean')
    )
)

regime_final


# In[321]:


print("First Entry Date:", backtest_df["Entry Date"].min())
print("Last Entry Date:", backtest_df["Entry Date"].max())
print("First Expiry Date:", backtest_df["Expiry Date"].min())
print("Last Expiry Date:", backtest_df["Expiry Date"].max())


# In[322]:


print(backtest_df[["Entry Date", "Expiry Date"]].to_string(index=False))


# In[323]:


# Methodology Audit — Step 1
# Check the construction of all covered-call cycles

audit_cols = [
    'Entry Date',
    'Expiry Date',
    'Entry NIFTY',
    'Expiry NIFTY',
    'Strike',
    'Historical Volatility',
    'Call Premium',
    'Long NIFTY P&L',
    'Short Call P&L',
    'Covered Call P&L',
    'Long NIFTY Return',
    'Covered Call Return',
    'Premium Yield',
    'Covered Call Advantage',
    'Market Regime',
    'Long NIFTY Wealth',
    'Covered Call Wealth'
]

audit_df = backtest_df[audit_cols].copy()

print("===== BACKTEST METHODOLOGY AUDIT =====")
print(f"Number of cycles: {len(audit_df)}")
print(f"First entry: {audit_df['Entry Date'].min()}")
print(f"Last entry: {audit_df['Entry Date'].max()}")
print(f"First expiry: {audit_df['Expiry Date'].min()}")
print(f"Last expiry: {audit_df['Expiry Date'].max()}")

print("\n===== MISSING VALUES =====")
print(audit_df.isna().sum())

print("\n===== DUPLICATE CYCLES =====")
print(audit_df.duplicated().sum())

print("\n===== CYCLE SUMMARY =====")
print(audit_df[['Entry Date', 'Expiry Date', 'Strike']].to_string(index=False))


# In[324]:


# Methodology Audit — Step 2
# Verify Covered Call P&L identity

audit_df['P&L Check'] = (
    audit_df['Long NIFTY P&L']
    + audit_df['Short Call P&L']
    - audit_df['Covered Call P&L']
)

print("===== P&L IDENTITY CHECK =====")
print(audit_df['P&L Check'])

print("\nMaximum absolute error:")
print(audit_df['P&L Check'].abs().max())


# In[325]:


# Methodology Audit — Step 3
# Verify return calculations

audit_df['Calculated NIFTY Return'] = (
    audit_df['Long NIFTY P&L']
    / audit_df['Entry NIFTY']
)

audit_df['Calculated CC Return'] = (
    audit_df['Covered Call P&L']
    / audit_df['Entry NIFTY']
)

audit_df['NIFTY Return Error'] = (
    audit_df['Long NIFTY Return']
    - audit_df['Calculated NIFTY Return']
)

audit_df['CC Return Error'] = (
    audit_df['Covered Call Return']
    - audit_df['Calculated CC Return']
)

print("===== RETURN VALIDATION =====")

print(
    "Maximum NIFTY return error:",
    audit_df['NIFTY Return Error'].abs().max()
)

print(
    "Maximum Covered Call return error:",
    audit_df['CC Return Error'].abs().max()
)


# In[326]:


# Methodology Audit — Step 4

audit_df['Calculated Advantage'] = (
    audit_df['Covered Call Return']
    - audit_df['Long NIFTY Return']
)

audit_df['Advantage Error'] = (
    audit_df['Covered Call Advantage']
    - audit_df['Calculated Advantage']
)

print("===== COVERED CALL ADVANTAGE CHECK =====")
print(
    "Maximum absolute error:",
    audit_df['Advantage Error'].abs().max()
)


# In[327]:


# Methodology Audit — Step 5

audit_df['Calculated Premium Yield'] = (
    audit_df['Call Premium']
    / audit_df['Entry NIFTY']
)

audit_df['Premium Yield Error'] = (
    audit_df['Premium Yield']
    - audit_df['Calculated Premium Yield']
)

print("===== PREMIUM YIELD CHECK =====")
print(
    "Maximum absolute error:",
    audit_df['Premium Yield Error'].abs().max()
)


# In[328]:


# Methodology Audit — Step 6
# Verify short-call payoff construction

audit_df['Calculated Short Call P&L'] = (
    audit_df['Call Premium']
    - (
        audit_df['Expiry NIFTY']
        - audit_df['Strike']
    ).clip(lower=0)
)

audit_df['Short Call P&L Error'] = (
    audit_df['Short Call P&L']
    - audit_df['Calculated Short Call P&L']
)

print("===== SHORT CALL P&L CHECK =====")
print(
    "Maximum absolute error:",
    audit_df['Short Call P&L Error'].abs().max()
)


# In[329]:


# Methodology Audit — Step 7

calculated_nifty_wealth = (
    1 + audit_df['Long NIFTY Return']
).cumprod()

calculated_cc_wealth = (
    1 + audit_df['Covered Call Return']
).cumprod()

print("===== WEALTH VALIDATION =====")

print(
    "Calculated NIFTY final wealth:",
    calculated_nifty_wealth.iloc[-1]
)

print(
    "Reported NIFTY final wealth:",
    audit_df['Long NIFTY Wealth'].iloc[-1]
)

print(
    "\nCalculated Covered Call final wealth:",
    calculated_cc_wealth.iloc[-1]
)

print(
    "Reported Covered Call final wealth:",
    audit_df['Covered Call Wealth'].iloc[-1]
)


# In[330]:


# FINAL CHART 1 — Cumulative Wealth

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

plt.plot(
    backtest_df['Expiry Date'],
    backtest_df['Long NIFTY Wealth'],
    label='NIFTY Buy & Hold',
    linewidth=2
)

plt.plot(
    backtest_df['Expiry Date'],
    backtest_df['Covered Call Wealth'],
    label='Covered Call',
    linewidth=2
)

plt.axhline(
    1,
    linestyle='--',
    linewidth=1
)

plt.title('Cumulative Wealth: NIFTY vs Covered Call')
plt.xlabel('Expiry Date')
plt.ylabel('Wealth Index')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# In[331]:


# FINAL CHART 2 — Cycle-by-Cycle Returns

plt.figure(figsize=(13, 6))

x = range(len(backtest_df))

plt.plot(
    x,
    backtest_df['Long NIFTY Return'] * 100,
    marker='o',
    label='NIFTY Return'
)

plt.plot(
    x,
    backtest_df['Covered Call Return'] * 100,
    marker='o',
    label='Covered Call Return'
)

plt.axhline(
    0,
    linestyle='--',
    linewidth=1
)

plt.title('Cycle-by-Cycle Returns: NIFTY vs Covered Call')
plt.xlabel('Investment Cycle')
plt.ylabel('Return (%)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# In[332]:


# FINAL CHART 3 — Covered Call Advantage

plt.figure(figsize=(13, 6))

advantage = backtest_df['Covered Call Advantage'] * 100

plt.bar(
    range(len(backtest_df)),
    advantage
)

plt.axhline(
    0,
    linestyle='--',
    linewidth=1
)

plt.title('Covered Call Advantage by Investment Cycle')
plt.xlabel('Investment Cycle')
plt.ylabel('Covered Call Advantage (%)')

plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()


# In[333]:


# FINAL CHART 4 — Volatility Comparison

plt.figure(figsize=(8, 6))

strategies = ['NIFTY', 'Covered Call']
volatility = [
    backtest_df['Long NIFTY Return'].std() * 100,
    backtest_df['Covered Call Return'].std() * 100
]

plt.bar(
    strategies,
    volatility
)

plt.title('Volatility Comparison')
plt.ylabel('Volatility (%)')

plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()


# In[334]:


# FINAL CHART 5 — Drawdown

nifty_wealth = backtest_df['Long NIFTY Wealth']
cc_wealth = backtest_df['Covered Call Wealth']

nifty_drawdown = (
    nifty_wealth / nifty_wealth.cummax() - 1
) * 100

cc_drawdown = (
    cc_wealth / cc_wealth.cummax() - 1
) * 100

plt.figure(figsize=(12, 6))

plt.plot(
    backtest_df['Expiry Date'],
    nifty_drawdown,
    label='NIFTY'
)

plt.plot(
    backtest_df['Expiry Date'],
    cc_drawdown,
    label='Covered Call'
)

plt.axhline(
    0,
    linestyle='--',
    linewidth=1
)

plt.title('Drawdown Comparison')
plt.xlabel('Expiry Date')
plt.ylabel('Drawdown (%)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# In[335]:


# FINAL CHART 6 — Market Regime Performance

regime_summary = (
    backtest_df
    .groupby('Market Regime')
    [['Long NIFTY Return', 'Covered Call Return']]
    .mean()
)

regime_order = ['Bull', 'Neutral', 'Bear']

regime_summary = regime_summary.reindex(regime_order)

plt.figure(figsize=(10, 6))

x = range(len(regime_summary))
width = 0.35

plt.bar(
    [i - width/2 for i in x],
    regime_summary['Long NIFTY Return'] * 100,
    width=width,
    label='NIFTY'
)

plt.bar(
    [i + width/2 for i in x],
    regime_summary['Covered Call Return'] * 100,
    width=width,
    label='Covered Call'
)

plt.axhline(
    0,
    linestyle='--',
    linewidth=1
)

plt.xticks(x, regime_order)

plt.title('Strategy Performance Across Market Regimes')
plt.xlabel('Market Regime')
plt.ylabel('Average Return (%)')
plt.legend()
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()


# In[336]:


# FINAL CHART 7 — Monte Carlo P&L Distribution

import numpy as np
import matplotlib.pyplot as plt

pnl = np.asarray(simulated_pnl_1d)

var_95 = np.percentile(pnl, 5)
var_99 = np.percentile(pnl, 1)

es_95 = pnl[pnl <= var_95].mean()
es_99 = pnl[pnl <= var_99].mean()

plt.figure(figsize=(12, 6))

plt.hist(
    pnl,
    bins=50,
    edgecolor='black',
    alpha=0.75
)

plt.axvline(
    var_95,
    linestyle='--',
    linewidth=2,
    label=f'95% VaR = ₹{abs(var_95):.2f}'
)

plt.axvline(
    var_99,
    linestyle='--',
    linewidth=2,
    label=f'99% VaR = ₹{abs(var_99):.2f}'
)

plt.axvline(
    es_95,
    linestyle=':',
    linewidth=2,
    label=f'95% ES = ₹{abs(es_95):.2f}'
)

plt.axvline(
    es_99,
    linestyle=':',
    linewidth=2,
    label=f'99% ES = ₹{abs(es_99):.2f}'
)

plt.title('Monte Carlo Simulation: One-Day Covered Call P&L')
plt.xlabel('Simulated P&L (₹)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

print("===== MONTE CARLO RISK MEASURES =====")
print(f"95% VaR: ₹{abs(var_95):.2f}")
print(f"99% VaR: ₹{abs(var_99):.2f}")
print(f"95% ES:  ₹{abs(es_95):.2f}")
print(f"99% ES:  ₹{abs(es_99):.2f}")


# In[337]:


# FINAL CHART 8 — Regression Relationship

import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np

x = backtest_df['Long NIFTY Return']
y = backtest_df['Covered Call Advantage']

X = sm.add_constant(x)

reg_model = sm.OLS(y, X).fit()

x_line = np.linspace(
    x.min(),
    x.max(),
    100
)

y_line = (
    reg_model.params.iloc[0]
    + reg_model.params.iloc[1] * x_line
)

plt.figure(figsize=(10, 6))

plt.scatter(
    x * 100,
    y * 100,
    s=60,
    alpha=0.8,
    label='Observed cycles'
)

plt.plot(
    x_line * 100,
    y_line * 100,
    linewidth=2,
    label='OLS regression line'
)

plt.axhline(
    0,
    linestyle='--',
    linewidth=1
)

plt.axvline(
    0,
    linestyle='--',
    linewidth=1
)

plt.title('NIFTY Return vs Covered Call Advantage')
plt.xlabel('NIFTY Return (%)')
plt.ylabel('Covered Call Advantage (%)')

plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("===== REGRESSION SUMMARY =====")
print(f"R-squared: {reg_model.rsquared:.6f}")
print(f"NIFTY Return coefficient: {reg_model.params.iloc[1]:.6f}")
print(f"NIFTY Return p-value: {reg_model.pvalues.iloc[1]:.6f}")


# In[338]:


# FINAL MASTER RESULTS TABLE

final_results = pd.DataFrame({
    'Metric': [
        'Number of Cycles',
        'Average NIFTY Return',
        'Average Covered Call Return',
        'Average Covered Call Advantage',
        'NIFTY Volatility',
        'Covered Call Volatility',
        'Volatility Reduction',
        'NIFTY Maximum Drawdown',
        'Covered Call Maximum Drawdown',
        'NIFTY Sharpe Ratio',
        'Covered Call Sharpe Ratio',
        'Covered Call Outperformance Rate',
        'Final NIFTY Wealth',
        'Final Covered Call Wealth',
        'Paired t-test p-value',
        'Wilcoxon p-value',
        'Regression R-squared',
        'NIFTY Return Regression Coefficient',
        'NIFTY Return Regression p-value'
    ],

    'Value': [
        22,
        -0.1015,
        0.1569,
        0.2584,
        3.4948,
        1.9314,
        44.7350,
        -10.4505,
        -6.2165,
        -0.0291,
        0.0812,
        77.2727,
        0.965560,
        1.030959,
        0.587921,
        0.068444,
        0.753968,
        -0.546865,
        0.000000
    ]
})

print("===== FINAL MASTER RESULTS TABLE =====")
display(final_results)


# In[339]:


# FINAL MARKET REGIME TABLE

regime_results = pd.DataFrame({
    'Market Regime': [
        'Bull',
        'Bear',
        'Neutral'
    ],

    'Cycles': [
        3,
        6,
        13
    ],

    'NIFTY Return (%)': [
        5.6431,
        -4.0228,
        0.3826
    ],

    'Covered Call Return (%)': [
        1.6041,
        -2.3540,
        0.9818
    ],

    'Covered Call Advantage (%)': [
        -4.0391,
        1.6688,
        0.5992
    ],

    'Outperformance (%)': [
        0.00,
        100.00,
        84.62
    ]
})

print("===== FINAL MARKET REGIME RESULTS =====")
display(regime_results)


# In[340]:


# FINAL RISK TABLE

risk_results = pd.DataFrame({
    'Risk Measure': [
        '1-Day VaR (95%)',
        '1-Day VaR (99%)',
        '1-Day Expected Shortfall (95%)',
        '1-Day Expected Shortfall (99%)',
        'NIFTY Volatility',
        'Covered Call Volatility',
        'Volatility Reduction',
        'NIFTY Maximum Drawdown',
        'Covered Call Maximum Drawdown'
    ],

    'Value': [
        152.55,
        208.92,
        186.87,
        231.28,
        3.4948,
        1.9314,
        44.7350,
        -10.4505,
        -6.2165
    ]
})

print("===== FINAL RISK RESULTS =====")
display(risk_results)


# In[341]:


# FINAL CSV EXPORT

final_results.to_csv(
    'NIFTY_Covered_Call_Final_Results.csv',
    index=False
)

regime_results.to_csv(
    'NIFTY_Covered_Call_Regime_Results.csv',
    index=False
)

risk_results.to_csv(
    'NIFTY_Covered_Call_Risk_Results.csv',
    index=False
)

print("Final analytical tables exported successfully.")


# In[ ]:




