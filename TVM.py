import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image

#favicon = Image.open("C:/Users/KIIT/Documents/GitHub/TVM/dollar-removebg-preview.ico") #for custom icons

st.set_page_config(
    page_title="Time Value of Money Calculator", 
 #   page_icon=favicon, #add this if you have an icon of your own locally
    layout="wide"
)

# App title and description
st.title("Time Value of Money Calculator")
st.markdown("""
This app allows you to calculate and visualize different time value of money scenarios.
Choose a calculation type and input your parameters to see results.
""")

# Create tabs for different calculation types
tabs = st.tabs(["Single Payment", "Uniform Series (Annuity)", "Gradient Series", "Custom Cash Flow"])

# Define functions for TVM calculations
def pv_of_fv(fv, i, n):
    """Calculate present value of a future value"""
    return fv / ((1 + i) ** n)

def fv_of_pv(pv, i, n):
    """Calculate future value of a present value"""
    return pv * ((1 + i) ** n)

def pv_of_annuity(A, i, n):
    """Calculate present value of an annuity"""
    return A * ((1 - (1 / ((1 + i) ** n))) / i)

def fv_of_annuity(A, i, n):
    """Calculate future value of an annuity"""
    return A * (((1 + i) ** n - 1) / i)

def payment_from_pv(pv, i, n):
    """Calculate payment amount from a present value"""
    return pv * (i * (1 + i) ** n) / ((1 + i) ** n - 1)

def payment_from_fv(fv, i, n):
    """Calculate payment amount from a future value"""
    return fv * i / ((1 + i) ** n - 1)

def pv_of_gradient(G, i, n):
    """Calculate present value of a gradient series"""
    return G * (((1 + i) ** n - 1 - n * i) / (i ** 2 * (1 + i) ** n))

def annuity_of_gradient(G, i, n):
    """Calculate uniform series equivalent of a gradient series"""
    return G * ((1 / i) - (n / ((1 + i) ** n - 1)))

# Function to create cash flow diagrams
def plot_cash_flow(cash_flows, title):
    periods = len(cash_flows)
    
    # Create figure
    fig = go.Figure()
    
    # Add cash flow arrows
    for i, cf in enumerate(cash_flows):
        if cf != 0:
            arrow_dir = 1 if cf < 0 else -1  # Point down for outflows, up for inflows
            arrow_length = abs(cf) / max(max(abs(min(cash_flows)), abs(max(cash_flows))), 1) * 0.8
            
            # Arrow body
            fig.add_shape(
                type="line",
                x0=i, y0=0,
                x1=i, y1=arrow_dir * arrow_length,
                line=dict(color="blue", width=2),
            )
            
            # Arrow head
            fig.add_shape(
                type="line",
                x0=i-0.1, y0=arrow_dir * arrow_length - 0.05 * arrow_dir,
                x1=i, y1=arrow_dir * arrow_length,
                line=dict(color="blue", width=2),
            )
            fig.add_shape(
                type="line",
                x0=i+0.1, y0=arrow_dir * arrow_length - 0.05 * arrow_dir,
                x1=i, y1=arrow_dir * arrow_length,
                line=dict(color="blue", width=2),
            )
            
            # Add value label
            fig.add_annotation(
                x=i,
                y=arrow_dir * (arrow_length + 0.1),
                text=f"${abs(cf):.2f}",
                showarrow=False,
                font=dict(size=10)
            )
    
    # Add time axis
    fig.add_shape(
        type="line",
        x0=-0.5, y0=0,
        x1=periods-0.5, y1=0,
        line=dict(color="black", width=1),
    )
    
    # Add period markers
    for i in range(periods):
        fig.add_shape(
            type="line",
            x0=i, y0=-0.05,
            x1=i, y1=0.05,
            line=dict(color="black", width=1),
        )
        fig.add_annotation(
            x=i,
            y=-0.15,
            text=f"{i}",
            showarrow=False,
            font=dict(size=10)
        )
    
    # Set layout
    fig.update_layout(
        title=title,
        xaxis_title="Time Period",
        showlegend=False,
        xaxis=dict(range=[-0.5, periods-0.5], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

# Function to plot the value growth over time
def plot_value_growth(values, title):
    periods = len(values)
    
    fig = go.Figure()
    
    # Line graph for value growth
    fig.add_trace(go.Scatter(
        x=list(range(periods)),
        y=values,
        mode='lines+markers',
        line=dict(color='green', width=2),
        marker=dict(size=8),
        name='Value'
    ))
    
    # Add point annotations
    for i, val in enumerate(values):
        fig.add_annotation(
            x=i,
            y=val,
            text=f"${val:.2f}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1,
            arrowcolor="#636363",
            ax=0,
            ay=-30,
            font=dict(size=10)
        )
    
    # Set layout
    fig.update_layout(
        title=title,
        xaxis_title="Time Period",
        yaxis_title="Value",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

# Single Payment tab
with tabs[0]:
    st.header("Single Payment Calculations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        sp_calc_type = st.selectbox(
            "Calculation type",
            ["Present Value to Future Value", "Future Value to Present Value"],
            key="sp_calc_type"
        )
        
        if sp_calc_type == "Present Value to Future Value":
            pv = st.number_input("Present Value (P)", value=1000.0, key="sp_pv")
            i = st.number_input("Interest Rate (i) [decimal]", value=0.05, min_value=0.0, format="%.4f", key="sp_i")
            n = st.number_input("Number of Periods (n)", value=5, min_value=1, key="sp_n")
            
            # Calculate future value
            fv = fv_of_pv(pv, i, n)
            
            st.success(f"Future Value (F) = ${fv:.2f}")
            
            # Create cash flows for visualization
            cash_flows = [-pv] + [0] * (n-1) + [fv]
            value_growth = [pv * (1 + i) ** t for t in range(n+1)]
            
        else:  # Future Value to Present Value
            fv = st.number_input("Future Value (F)", value=1000.0, key="sp_fv")
            i = st.number_input("Interest Rate (i) [decimal]", value=0.05, min_value=0.0, format="%.4f", key="sp_i")
            n = st.number_input("Number of Periods (n)", value=5, min_value=1, key="sp_n")
            
            # Calculate present value
            pv = pv_of_fv(fv, i, n)
            
            st.success(f"Present Value (P) = ${pv:.2f}")
            
            # Create cash flows for visualization
            cash_flows = [-pv] + [0] * (n-1) + [fv]
            value_growth = [pv * (1 + i) ** t for t in range(n+1)]
    
    with col2:
        # Display formula
        st.subheader("Formula Used")
        if sp_calc_type == "Present Value to Future Value":
            st.latex("F = P(1+i)^n")
        else:
            st.latex("P = \\frac{F}{(1+i)^n}")
        
        # Display cash flow diagram
        st.subheader("Cash Flow Diagram")
        cf_fig = plot_cash_flow(cash_flows, "Single Payment Cash Flow")
        st.plotly_chart(cf_fig, use_container_width=True)
        
        # Display value growth
        st.subheader("Value Growth Over Time")
        growth_fig = plot_value_growth(value_growth, "Value Growth")
        st.plotly_chart(growth_fig, use_container_width=True)

# Uniform Series (Annuity) tab
with tabs[1]:
    st.header("Uniform Series (Annuity) Calculations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        annuity_calc_type = st.selectbox(
            "Calculation type",
            ["Annuity to Present Value", "Annuity to Future Value", 
             "Present Value to Annuity", "Future Value to Annuity"],
            key="annuity_calc_type"
        )
        
        i = st.number_input("Interest Rate (i) [decimal]", value=0.05, min_value=0.0, format="%.4f", key="annuity_i")
        n = st.number_input("Number of Periods (n)", value=5, min_value=1, key="annuity_n")
        
        if annuity_calc_type == "Annuity to Present Value":
            A = st.number_input("Periodic Payment (A)", value=100.0, key="annuity_A")
            pv = pv_of_annuity(A, i, n)
            st.success(f"Present Value (P) = ${pv:.2f}")
            cash_flows = [-pv] + [A] * n
            value_remaining = [pv_of_annuity(A, i, n-t) for t in range(n+1)]
            
        elif annuity_calc_type == "Annuity to Future Value":
            A = st.number_input("Periodic Payment (A)", value=100.0, key="annuity_A")
            fv = fv_of_annuity(A, i, n)
            st.success(f"Future Value (F) = ${fv:.2f}")
            cash_flows = [0] + [A] * n + [fv]
            value_accumulated = [fv_of_annuity(A, i, t) for t in range(n+1)]
            
        elif annuity_calc_type == "Present Value to Annuity":
            pv = st.number_input("Present Value (P)", value=1000.0, key="annuity_pv")
            A = payment_from_pv(pv, i, n)
            st.success(f"Periodic Payment (A) = ${A:.2f}")
            cash_flows = [-pv] + [A] * n
            value_remaining = [pv_of_annuity(A, i, n-t) for t in range(n+1)]
            
        else:  # Future Value to Annuity
            fv = st.number_input("Future Value (F)", value=1000.0, key="annuity_fv")
            A = payment_from_fv(fv, i, n)
            st.success(f"Periodic Payment (A) = ${A:.2f}")
            cash_flows = [0] + [A] * n + [fv]
            value_accumulated = [fv_of_annuity(A, i, t) for t in range(n+1)]
    
    with col2:
        # Display formula
        st.subheader("Formula Used")
        if annuity_calc_type == "Annuity to Present Value":
            st.latex("P = A\\frac{(1+i)^n - 1}{i(1+i)^n}")
        elif annuity_calc_type == "Annuity to Future Value":
            st.latex("F = A\\frac{(1+i)^n - 1}{i}")
        elif annuity_calc_type == "Present Value to Annuity":
            st.latex("A = P\\frac{i(1+i)^n}{(1+i)^n - 1}")
        else:  # Future Value to Annuity
            st.latex("A = F\\frac{i}{(1+i)^n - 1}")
        
        # Display cash flow diagram
        st.subheader("Cash Flow Diagram")
        cf_fig = plot_cash_flow(cash_flows[:-1] if annuity_calc_type.startswith("Annuity to Future") or annuity_calc_type.startswith("Future Value") else cash_flows, "Annuity Cash Flow")
        st.plotly_chart(cf_fig, use_container_width=True)
        
        # Display value growth/decline
        st.subheader("Value Over Time")
        if annuity_calc_type == "Annuity to Present Value" or annuity_calc_type == "Present Value to Annuity":
            growth_fig = plot_value_growth(value_remaining, "Outstanding Balance")
        else:
            growth_fig = plot_value_growth(value_accumulated, "Accumulated Value")
        st.plotly_chart(growth_fig, use_container_width=True)

# Gradient Series tab
with tabs[2]:
    st.header("Gradient Series Calculations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        gradient_calc_type = st.selectbox(
            "Calculation type",
            ["Gradient to Present Value", "Gradient to Equivalent Uniform Series"],
            key="gradient_calc_type"
        )
        
        i = st.number_input("Interest Rate (i) [decimal]", value=0.05, min_value=0.0, format="%.4f", key="gradient_i")
        n = st.number_input("Number of Periods (n)", value=5, min_value=1, key="gradient_n")
        G = st.number_input("Gradient Amount (G)", value=20.0, key="gradient_G")
        first_payment = st.number_input("First Payment (A₁)", value=100.0, key="gradient_A1")
        
        if gradient_calc_type == "Gradient to Present Value":
            # Calculate present value of base amount
            pv_base = pv_of_annuity(first_payment, i, n)
            # Calculate present value of gradient
            pv_gradient = pv_of_gradient(G, i, n)
            pv_total = pv_base + pv_gradient
            
            st.success(f"Present Value (P) = ${pv_total:.2f}")
            
            # Create cash flows
            cash_flows = [-pv_total] + [first_payment + G * t for t in range(n)]
            
        else:  # Gradient to Equivalent Uniform Series
            # Calculate equivalent annuity payment
            A_gradient = annuity_of_gradient(G, i, n)
            A_total = first_payment + A_gradient
            
            st.success(f"Equivalent Uniform Series (A) = ${A_total:.2f}")
            
            # Calculate PV for visualization
            pv_total = pv_of_annuity(A_total, i, n)
            
            # Create cash flows - original gradient payments
            gradient_flows = [first_payment + G * t for t in range(n)]
            # Create equivalent uniform flows
            uniform_flows = [A_total] * n
            
            # Create composite visualization
            cash_flows = [-pv_total] + gradient_flows
    
    with col2:
        # Display formula
        st.subheader("Formula Used")
        if gradient_calc_type == "Gradient to Present Value":
            st.latex("P = G\\frac{(1+i)^n - 1 - n \\times i}{i^2 \\times (1+i)^n}")
        else:
            st.latex("A = G\\left(\\frac{1}{i} - \\frac{n}{(1+i)^n - 1}\\right)")
        
        # Display cash flow diagram
        st.subheader("Cash Flow Diagram")
        cf_fig = plot_cash_flow(cash_flows, "Gradient Series Cash Flow")
        st.plotly_chart(cf_fig, use_container_width=True)
        
        # For equivalent uniform series, show comparison
        if gradient_calc_type == "Gradient to Equivalent Uniform Series":
            st.subheader("Gradient vs Equivalent Uniform Series")
            
            fig = go.Figure()
            
            # Add gradient series
            fig.add_trace(go.Bar(
                x=list(range(1, n+1)),
                y=gradient_flows,
                name='Gradient Series',
                marker_color='lightblue'
            ))
            
            # Add equivalent uniform series
            fig.add_trace(go.Bar(
                x=list(range(1, n+1)),
                y=uniform_flows,
                name='Equivalent Uniform Series',
                marker_color='lightgreen'
            ))
            
            # Set layout
            fig.update_layout(
                title="Comparison of Cash Flows",
                xaxis_title="Time Period",
                yaxis_title="Payment Amount",
                height=300,
                barmode='group',
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)

# Custom Cash Flow tab
with tabs[3]:
    st.header("Custom Cash Flow Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        n = st.number_input("Number of Periods", value=5, min_value=1, max_value=20, key="custom_n")
        i = st.number_input("Interest Rate (i) [decimal]", value=0.05, min_value=0.0, format="%.4f", key="custom_i")
        
        # Create input fields for each period's cash flow
        cash_flows = [0] * (n+1)
        st.subheader("Enter Cash Flows")
        st.markdown("(Positive for inflows, negative for outflows)")
        
        cash_flows[0] = st.number_input("Initial Cash Flow (t=0)", value=-1000.0, key="cf_0")
        
        # Create rows of 4 inputs each
        for row in range((n + 3) // 4):
            cols = st.columns(4)
            for col_idx, col in enumerate(cols):
                period = row * 4 + col_idx + 1
                if period <= n:
                    cash_flows[period] = col.number_input(f"t={period}", value=0.0, key=f"cf_{period}")
        
        # Calculate net present value
        npv = cash_flows[0]  # Initial cash flow
        for t in range(1, n+1):
            npv += cash_flows[t] / ((1 + i) ** t)
        
        # Calculate internal rate of return (IRR)
        try:
            irr = np.irr(cash_flows)
            irr_valid = True
        except:
            irr = None
            irr_valid = False
            
        # Calculate future value at the end
        fv = cash_flows[0] * ((1 + i) ** n)
        for t in range(1, n+1):
            fv += cash_flows[t] * ((1 + i) ** (n - t))
        
        st.success(f"Net Present Value (NPV) = ${npv:.2f}")
        if irr_valid:
            st.success(f"Internal Rate of Return (IRR) = {irr:.2%}")
        else:
            st.error("IRR calculation not possible with this cash flow pattern")
        st.success(f"Future Value (at t={n}) = ${fv:.2f}")
    
    with col2:
        # Display cash flow diagram
        st.subheader("Cash Flow Diagram")
        cf_fig = plot_cash_flow(cash_flows, "Custom Cash Flow")
        st.plotly_chart(cf_fig, use_container_width=True)
        
        # Calculate and display cumulative present value
        cumulative_pv = [0] * (n+1)
        cumulative_pv[0] = cash_flows[0]
        for t in range(1, n+1):
            present_value = cash_flows[t] / ((1 + i) ** t)
            cumulative_pv[t] = cumulative_pv[t-1] + present_value
        
        st.subheader("Cumulative Present Value")
        fig = go.Figure()
        
        # Add line for cumulative PV
        fig.add_trace(go.Scatter(
            x=list(range(n+1)),
            y=cumulative_pv,
            mode='lines+markers',
            name='Cumulative PV',
            line=dict(color='purple', width=2),
            marker=dict(size=8)
        ))
        
        # Add horizontal line at NPV
        fig.add_shape(
            type="line",
            x0=0, y0=npv,
            x1=n, y1=npv,
            line=dict(color="red", width=1, dash="dash"),
        )
        
        # Add annotation for NPV
        fig.add_annotation(
            x=n,
            y=npv,
            text=f"NPV = ${npv:.2f}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1,
            arrowcolor="#636363",
            ax=-50,
            ay=0,
            font=dict(size=10)
        )
        
        # Set layout
        fig.update_layout(
            title="Cumulative Present Value",
            xaxis_title="Time Period",
            yaxis_title="Value",
            height=300,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Add footer with formulas reference
st.markdown("---")
with st.expander("Reference: Time Value of Money Formulas"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Single Payment Formulas
        - **P to F**: F = P(1+i)ⁿ
        - **F to P**: P = F/(1+i)ⁿ
        
        ### Uniform Series Formulas
        - **A to F**: F = A[(1+i)ⁿ - 1]/i
        - **A to P**: P = A[(1+i)ⁿ - 1]/[i(1+i)ⁿ]
        - **P to A**: A = P[i(1+i)ⁿ]/[(1+i)ⁿ - 1]
        - **F to A**: A = F[i]/[(1+i)ⁿ - 1]
        """)
    
    with col2:
        st.markdown("""
        ### Gradient Series Formulas
        - **G to P**: P = G[[(1+i)ⁿ - 1 - n×i]/[i²×(1+i)ⁿ]]
        - **G to A**: A = G[(1/i) - (n/[(1+i)ⁿ - 1])]
        
        ### Perpetuity Formulas
        - **Perpetuity**: P = A/i
        - **Growing Perpetuity**: P = A/(i-g)
        """)
