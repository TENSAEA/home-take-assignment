#!/usr/bin/env python3
"""
Streamlit Demo - Visual Deep Learning Parallelization Dashboard
================================================================
Interactive web-based demo showing CNN training with parallel strategies.
"""

import streamlit as st
import numpy as np
import time
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os

# Page configuration
st.set_page_config(
    page_title="Deep Learning Parallelization Demo",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1E3A5F;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #5A6C7D;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stProgress > div > div > div > div {
        background-color: #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">🚀 Deep Learning Parallelization Demo</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">CNN Training on CIFAR-10 with Hybrid MPI+OpenMP</p>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center;"><b>Tensae Aschalew</b> | ID: GSR/3976/17</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("📋 Navigation")
page = st.sidebar.radio("Go to", [
    "🏠 Overview",
    "🧠 CNN Architecture",
    "⚡ Parallelization Strategies",
    "📊 Performance Results",
    "🎮 Interactive Training",
    "📈 Scalability Analysis"
])

# ============================================================================
# PAGE: Overview
# ============================================================================
if page == "🏠 Overview":
    st.header("Project Overview")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🎯 Max Speedup", "5.69×", "vs Serial")
    with col2:
        st.metric("⏱️ Training Time", "21s", "-96s from serial")
    with col3:
        st.metric("✅ Accuracy", "42.2%", "Maintained")
    
    st.markdown("---")
    
    st.subheader("🎯 Assignment Objectives")
    st.markdown("""
    1. ✅ **Serial Baseline**: Implemented CNN training from scratch using NumPy
    2. ✅ **MPI Parallelism**: Data-parallel training with gradient synchronization
    3. ✅ **Hybrid MPI+OpenMP**: Combined process and thread-level parallelism
    4. ✅ **Performance Analysis**: Speedup, efficiency, and scalability evaluation
    """)
    
    st.subheader("🔑 Key Design Decisions")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("""
        **Why CIFAR-10 over MNIST?**
        - 3× more data (RGB vs grayscale)
        - Higher compute per sample
        - Better demonstrates parallelization benefits
        """)
    with col2:
        st.info("""
        **Why Pure NumPy?**
        - Full algorithmic transparency
        - Control over parallelization points
        - Educational value
        """)

# ============================================================================
# PAGE: CNN Architecture
# ============================================================================
elif page == "🧠 CNN Architecture":
    st.header("Convolutional Neural Network Architecture")
    
    # Architecture diagram using Plotly
    fig = go.Figure()
    
    # Layer positions and sizes
    layers = [
        {"name": "Input\n32×32×3", "x": 0, "width": 0.8, "color": "#3498db"},
        {"name": "Conv2D\n32 filters", "x": 1.2, "width": 0.9, "color": "#2ecc71"},
        {"name": "MaxPool\n2×2", "x": 2.4, "width": 0.6, "color": "#f39c12"},
        {"name": "Conv2D\n64 filters", "x": 3.6, "width": 1.0, "color": "#2ecc71"},
        {"name": "MaxPool\n2×2", "x": 4.8, "width": 0.5, "color": "#f39c12"},
        {"name": "Flatten", "x": 6.0, "width": 0.4, "color": "#9b59b6"},
        {"name": "Dense\n256", "x": 7.2, "width": 0.7, "color": "#e74c3c"},
        {"name": "Dense\n10", "x": 8.4, "width": 0.5, "color": "#e74c3c"},
        {"name": "Softmax\nOutput", "x": 9.6, "width": 0.6, "color": "#1abc9c"},
    ]
    
    for i, layer in enumerate(layers):
        fig.add_trace(go.Scatter(
            x=[layer["x"], layer["x"], layer["x"] + layer["width"], layer["x"] + layer["width"], layer["x"]],
            y=[0, 1, 1, 0, 0],
            fill="toself",
            fillcolor=layer["color"],
            line=dict(color="white", width=2),
            name=layer["name"].split("\n")[0],
            text=layer["name"],
            hoverinfo="text",
            mode="lines"
        ))
        fig.add_annotation(
            x=layer["x"] + layer["width"]/2,
            y=0.5,
            text=layer["name"],
            showarrow=False,
            font=dict(size=10, color="white"),
            align="center"
        )
        
        # Add arrows between layers
        if i < len(layers) - 1:
            next_layer = layers[i + 1]
            fig.add_annotation(
                x=layer["x"] + layer["width"] + 0.1,
                y=0.5,
                ax=next_layer["x"] - 0.1,
                ay=0.5,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="#666"
            )
    
    fig.update_layout(
        title="CNN Architecture Flow",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Layer details
    st.subheader("📋 Layer Details")
    layer_data = {
        "Layer": ["Conv2D-1", "MaxPool-1", "Conv2D-2", "MaxPool-2", "Dense-1", "Dense-2"],
        "Input Shape": ["32×32×3", "32×32×32", "16×16×32", "16×16×64", "4096", "256"],
        "Output Shape": ["32×32×32", "16×16×32", "16×16×64", "8×8×64", "256", "10"],
        "Parameters": ["896", "0", "18,496", "0", "1,048,832", "2,570"],
        "Activation": ["ReLU", "-", "ReLU", "-", "ReLU", "Softmax"]
    }
    st.table(layer_data)
    
    st.metric("Total Parameters", "1,070,794", "~4 MB in float32")

# ============================================================================
# PAGE: Parallelization Strategies
# ============================================================================
elif page == "⚡ Parallelization Strategies":
    st.header("Parallelization Strategies")
    
    tab1, tab2, tab3 = st.tabs(["🔹 Serial Baseline", "🔸 MPI Data Parallelism", "🔶 Hybrid MPI+OpenMP"])
    
    with tab1:
        st.subheader("Serial Baseline")
        st.markdown("""
        The serial implementation processes all training data sequentially on a single CPU core.
        
        ```
        for epoch in range(epochs):
            for batch in batches:
                y_pred = model.forward(X_batch)
                loss = compute_loss(y_pred, y_true)
                gradients = model.backward(loss)
                optimizer.update(model, gradients)
        ```
        """)
        
        # Serial diagram
        fig = go.Figure()
        for i in range(4):
            fig.add_trace(go.Scatter(
                x=[i, i+0.8], y=[0, 0],
                mode="lines",
                line=dict(width=30, color="#3498db"),
                name=f"Batch {i+1}"
            ))
        fig.update_layout(
            title="Sequential Batch Processing",
            xaxis_title="Time →",
            showlegend=False,
            height=200
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("MPI Data Parallelism")
        st.markdown("""
        **Strategy**: Divide data across MPI processes, compute gradients locally, then synchronize.
        
        **Key Operation**: `MPI_Allreduce` for gradient averaging
        """)
        
        # MPI diagram
        fig = make_subplots(rows=4, cols=1, subplot_titles=["Process 0", "Process 1", "Process 2", "Process 3"])
        colors = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12"]
        for i in range(4):
            fig.add_trace(go.Scatter(
                x=[0, 0.6, 0.7, 1.3, 1.4, 2],
                y=[0, 0, 0, 0, 0, 0],
                mode="lines+markers",
                line=dict(width=20, color=colors[i]),
                marker=dict(size=15, symbol="diamond", color="white", line=dict(width=2, color=colors[i])),
                name=f"Process {i}"
            ), row=i+1, col=1)
        
        fig.add_vline(x=0.65, line_dash="dash", line_color="red", annotation_text="Allreduce")
        fig.add_vline(x=1.35, line_dash="dash", line_color="red")
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        st.success("**Speedup**: 3.17× with 4 MPI processes")
    
    with tab3:
        st.subheader("Hybrid MPI+OpenMP")
        st.markdown("""
        **Why Hybrid?**
        - Initially I planned to use only MPI
        - Decided to challenge myself with a more complex approach
        - Better hardware utilization on multi-core CPUs
        - Reduces memory overhead compared to pure MPI
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            st.info("""
            **MPI Level (Process)**
            - Data distribution
            - Gradient synchronization
            - Inter-node communication
            """)
        with col2:
            st.info("""
            **OpenMP Level (Thread)**
            - Parallel convolutions
            - `numba.prange` loops
            - Intra-node parallelism
            """)
        
        st.success("**Speedup**: 5.69× with 2 processes × 4 threads")

# ============================================================================
# PAGE: Performance Results
# ============================================================================
elif page == "📊 Performance Results":
    st.header("Performance Results")
    
    # Performance data
    configs = ["Serial", "MPI-2P", "MPI-4P", "Hybrid-2P×4T"]
    times = [117.58, 63.5, 37.15, 20.67]
    speedups = [1.0, 1.85, 3.17, 5.69]
    efficiencies = [100, 92.5, 79.2, 71.1]
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Speedup chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=configs,
            y=speedups,
            marker_color=["#3498db", "#2ecc71", "#f39c12", "#e74c3c"],
            text=[f"{s:.2f}×" for s in speedups],
            textposition="auto"
        ))
        fig.add_trace(go.Scatter(
            x=configs,
            y=[1, 2, 4, 8],
            mode="lines+markers",
            name="Ideal Linear",
            line=dict(dash="dash", color="gray")
        ))
        fig.update_layout(
            title="Speedup Comparison",
            yaxis_title="Speedup (×)",
            showlegend=True,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Training time chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=configs,
            y=times,
            marker_color=["#3498db", "#2ecc71", "#f39c12", "#e74c3c"],
            text=[f"{t:.1f}s" for t in times],
            textposition="auto"
        ))
        fig.update_layout(
            title="Training Time",
            yaxis_title="Time (seconds)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Efficiency pie chart
    st.subheader("Parallel Efficiency")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Serial", "100%", "Baseline")
    with col2:
        st.metric("MPI-2P", "92.5%", "-7.5%")
    with col3:
        st.metric("MPI-4P", "79.2%", "-20.8%")
    with col4:
        st.metric("Hybrid", "71.1%", "-28.9%")
    
    st.markdown("---")
    st.subheader("📋 Summary Table")
    results = {
        "Configuration": configs,
        "Time (s)": times,
        "Speedup": [f"{s:.2f}×" for s in speedups],
        "Efficiency": [f"{e:.1f}%" for e in efficiencies],
        "Test Accuracy": ["42.2%"] * 4
    }
    st.table(results)

# ============================================================================
# PAGE: Interactive Training
# ============================================================================
elif page == "🎮 Interactive Training":
    st.header("🎮 Interactive Training Simulation")
    
    st.markdown("""
    Click **Start Training** to see a simulated comparison of training approaches.
    This demonstrates how parallel training reduces total training time.
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        epochs = st.slider("Epochs", 1, 5, 3)
    with col2:
        batch_size = st.selectbox("Batch Size", [16, 32, 64], index=1)
    with col3:
        n_samples = st.selectbox("Training Samples", [500, 1000, 2000], index=1)
    
    if st.button("🚀 Start Training Simulation", type="primary"):
        
        # Create placeholders
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("🔵 Serial")
            serial_progress = st.progress(0)
            serial_status = st.empty()
            serial_chart = st.empty()
        
        with col2:
            st.subheader("🟢 MPI-4P")
            mpi_progress = st.progress(0)
            mpi_status = st.empty()
            mpi_chart = st.empty()
        
        with col3:
            st.subheader("🔴 Hybrid")
            hybrid_progress = st.progress(0)
            hybrid_status = st.empty()
            hybrid_chart = st.empty()
        
        # Simulated training data
        serial_losses = []
        mpi_losses = []
        hybrid_losses = []
        
        total_steps = epochs * 10
        
        for step in range(total_steps):
            progress = (step + 1) / total_steps
            epoch = step // 10 + 1
            
            # Simulated losses (decreasing)
            base_loss = 2.5 * (1 - progress) + 0.5 + np.random.uniform(-0.1, 0.1)
            serial_losses.append(base_loss)
            mpi_losses.append(base_loss + np.random.uniform(-0.05, 0.05))
            hybrid_losses.append(base_loss + np.random.uniform(-0.05, 0.05))
            
            # Update serial (slowest)
            serial_prog = min(progress, 1.0)
            serial_progress.progress(serial_prog)
            serial_status.markdown(f"**Epoch {epoch}/{epochs}** | Loss: {serial_losses[-1]:.4f}")
            
            # Update MPI (faster)
            mpi_prog = min(progress * 3.17, 1.0)
            mpi_progress.progress(mpi_prog)
            if mpi_prog < 1.0:
                mpi_status.markdown(f"**Epoch {epoch}/{epochs}** | Loss: {mpi_losses[-1]:.4f}")
            else:
                mpi_status.markdown("✅ **Complete!**")
            
            # Update Hybrid (fastest)
            hybrid_prog = min(progress * 5.69, 1.0)
            hybrid_progress.progress(hybrid_prog)
            if hybrid_prog < 1.0:
                hybrid_status.markdown(f"**Epoch {epoch}/{epochs}** | Loss: {hybrid_losses[-1]:.4f}")
            else:
                hybrid_status.markdown("✅ **Complete!**")
            
            # Update charts
            x_vals = list(range(len(serial_losses)))
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=serial_losses, name="Loss", line=dict(color="#3498db")))
            fig.update_layout(height=200, margin=dict(l=20, r=20, t=20, b=20), showlegend=False)
            serial_chart.plotly_chart(fig, use_container_width=True, key=f"serial_{step}")
            
            time.sleep(0.1)
        
        st.success("🎉 Training simulation complete!")
        st.balloons()

# ============================================================================
# PAGE: Scalability Analysis
# ============================================================================
elif page == "📈 Scalability Analysis":
    st.header("Scalability Analysis")
    
    st.subheader("Amdahl's Law")
    st.latex(r"S_{max} = \frac{1}{(1-p) + \frac{p}{N}}")
    st.markdown("Where *p* is the parallelizable fraction and *N* is the number of processors.")
    
    # Amdahl's Law visualization
    p = st.slider("Parallelizable Fraction (p)", 0.5, 0.99, 0.95, 0.01)
    
    processors = np.array([1, 2, 4, 8, 16, 32, 64])
    speedup_ideal = processors
    speedup_amdahl = 1 / ((1 - p) + p / processors)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=processors, y=speedup_ideal,
        mode="lines+markers",
        name="Ideal Linear",
        line=dict(dash="dash", color="gray")
    ))
    fig.add_trace(go.Scatter(
        x=processors, y=speedup_amdahl,
        mode="lines+markers",
        name=f"Amdahl's Law (p={p})",
        line=dict(color="#e74c3c")
    ))
    # Our actual data
    fig.add_trace(go.Scatter(
        x=[1, 2, 4, 8],
        y=[1, 1.85, 3.17, 5.69],
        mode="markers",
        name="Our Results",
        marker=dict(size=15, color="#2ecc71", symbol="star")
    ))
    
    fig.update_layout(
        title="Speedup vs Number of Processors",
        xaxis_title="Number of Processors",
        yaxis_title="Speedup",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.info(f"With p={p:.2f}, maximum theoretical speedup is **{1/(1-p):.1f}×** (infinite processors)")
    
    st.markdown("---")
    st.subheader("Communication Overhead Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Factors Affecting Scalability:**
        - Gradient synchronization (MPI_Allreduce)
        - Memory bandwidth limits
        - Cache coherency overhead
        - Load imbalance
        """)
    with col2:
        # Overhead pie chart
        fig = go.Figure(data=[go.Pie(
            labels=["Computation", "Gradient Sync", "Memory Transfer", "Other"],
            values=[70, 20, 7, 3],
            hole=0.4
        )])
        fig.update_layout(title="Time Distribution")
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>Deep Learning Parallelization Project | Tensae Aschalew (GSR/3976/17)</p>
    <p>Built with Streamlit • NumPy • MPI • OpenMP</p>
</div>
""", unsafe_allow_html=True)
