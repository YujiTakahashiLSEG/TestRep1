"""Streamlit app for Black-Scholes option pricing and Greeks calculation."""

import streamlit as st
from black_scholes import (
    black_scholes_call,
    black_scholes_put,
    delta,
    gamma,
    theta,
    vega,
    rho,
)

st.set_page_config(page_title="Black-Scholes Calculator", layout="wide")
st.title("Black-Scholes Option Pricing & Greeks")

# --- Sidebar inputs ---
st.sidebar.header("Option Parameters")
S = st.sidebar.number_input("Spot Price (S)", min_value=0.01, value=100.0, step=1.0, format="%.2f")
K = st.sidebar.number_input("Strike Price (K)", min_value=0.01, value=105.0, step=1.0, format="%.2f")
T = st.sidebar.number_input("Time to Maturity (years)", min_value=0.001, value=1.0, step=0.01, format="%.3f")
r = st.sidebar.number_input("Risk-Free Rate", min_value=0.0, value=0.05, step=0.001, format="%.4f")
sigma = st.sidebar.number_input("Volatility (sigma)", min_value=0.001, value=0.20, step=0.01, format="%.4f")

# --- Calculate prices ---
call_price = black_scholes_call(S, K, T, r, sigma)
put_price = black_scholes_put(S, K, T, r, sigma)

# --- Calculate Greeks ---
greeks = {
    "Delta": (delta(S, K, T, r, sigma, "call"), delta(S, K, T, r, sigma, "put")),
    "Gamma": (gamma(S, K, T, r, sigma), gamma(S, K, T, r, sigma)),
    "Theta (per day)": (theta(S, K, T, r, sigma, "call") / 365, theta(S, K, T, r, sigma, "put") / 365),
    "Vega": (vega(S, K, T, r, sigma), vega(S, K, T, r, sigma)),
    "Rho": (rho(S, K, T, r, sigma, "call"), rho(S, K, T, r, sigma, "put")),
}

# --- Display option prices ---
st.subheader("Option Prices")
col1, col2 = st.columns(2)
col1.metric("Call Price", f"${call_price:.4f}")
col2.metric("Put Price", f"${put_price:.4f}")

# --- Display Greeks table ---
st.subheader("Greeks")

header_cols = st.columns([2, 2, 2])
header_cols[0].markdown("**Greek**")
header_cols[1].markdown("**Call**")
header_cols[2].markdown("**Put**")

for name, (call_val, put_val) in greeks.items():
    row = st.columns([2, 2, 2])
    row[0].write(name)
    row[1].write(f"{call_val:.6f}")
    row[2].write(f"{put_val:.6f}")

# --- Put-Call Parity check ---
st.subheader("Put-Call Parity Check")
import math

lhs = call_price - put_price
rhs = S - K * math.exp(-r * T)
st.write(f"C - P = {lhs:.4f}")
st.write(f"S - K * e^(-rT) = {rhs:.4f}")
st.write(f"Difference = {abs(lhs - rhs):.2e}")
