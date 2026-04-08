from __future__ import annotations

import streamlit as st

from black_scholes import OptionInputs, greeks, price_call, price_put

st.set_page_config(page_title="Black-Scholes Pricer", layout="centered")
st.title("Black-Scholes Option Pricer")

with st.sidebar:
    st.header("Inputs")
    spot = st.number_input("Spot Price (S)", min_value=0.0001, value=100.0, step=1.0)
    strike = st.number_input("Strike (K)", min_value=0.0001, value=100.0, step=1.0)
    maturity = st.number_input("Maturity in Years (T)", min_value=0.0001, value=1.0, step=0.1)
    rate_pct = st.number_input("Risk-Free Rate (%)", value=5.0, step=0.1)
    vol_pct = st.number_input("Volatility (%)", min_value=0.0001, value=20.0, step=0.1)
    option_type = st.selectbox("Option Type", ["call", "put"])

params = OptionInputs(
    spot=spot,
    strike=strike,
    maturity=maturity,
    rate=rate_pct / 100.0,
    volatility=vol_pct / 100.0,
)

try:
    price = price_call(params) if option_type == "call" else price_put(params)
    g = greeks(params, option_type)

    st.subheader("Results")
    st.metric("Option Price", f"{price:,.4f}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Delta", f"{g['delta']:.6f}")
    c2.metric("Gamma", f"{g['gamma']:.6f}")
    c3.metric("Vega", f"{g['vega']:.6f}")

    c4, c5 = st.columns(2)
    c4.metric("Theta (per year)", f"{g['theta']:.6f}")
    c5.metric("Rho", f"{g['rho']:.6f}")
except ValueError as ex:
    st.error(str(ex))
