"""
POD-FCDNN Streamlit Web Application

Interactive dashboard for POD-based surrogate modeling of fluid dynamics.
Allows users to:
1. Configure dataset and hyperparameters
2. Train POD + Neural Network
3. Make predictions and compare with reference data
"""

import streamlit as st
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch

from engine import (
    discover_cases,
    load_training_snapshots,
    fit_pod,
    pod_project,
    PODFCDNNTrainer,
    predict_and_reconstruct,
    compute_errors,
    load_snapshot_uvp,
)


# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="POD-FCDNN Surrogate Model",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌊 POD-FCDNN Fluid Dynamics Surrogate Model")
st.markdown(
    """
    Train and deploy a neural network surrogate model for rapid CFD prediction.
    Combines Proper Orthogonal Decomposition with Deep Learning.
    """
)


# ============================================================================
# Session State Initialization
# ============================================================================

if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.X = None
    st.session_state.Re_values = None
    st.session_state.xy = None
    st.session_state.pod = None
    st.session_state.pod_coeffs = None

if "model_trained" not in st.session_state:
    st.session_state.model_trained = False
    st.session_state.trainer = None
    st.session_state.training_history = None

if "training_in_progress" not in st.session_state:
    st.session_state.training_in_progress = False


# ============================================================================
# Sidebar: Configuration
# ============================================================================

st.sidebar.markdown("## Configuration")

# Dataset Directory
st.sidebar.subheader("📂 Dataset")
data_dir_input = st.sidebar.text_input(
    "Dataset directory path:",
    value=r"D:\CAL Projects\Kainat\new_data\cavityflow\cavityflow training data files",
    help="Path to folder containing Re*.dat files"
)

# Load data button
if st.sidebar.button("🔍 Load Dataset", key="load_btn"):
    try:
        data_dir = Path(data_dir_input)
        if not data_dir.exists():
            st.sidebar.error(f"Directory not found: {data_dir}")
        else:
            with st.spinner("Loading dataset..."):
                X, Re_values, xy = load_training_snapshots(data_dir)
                st.session_state.X = X
                st.session_state.Re_values = Re_values
                st.session_state.xy = xy
                st.session_state.data_loaded = True
            st.sidebar.success("✅ Dataset loaded!")
    except Exception as e:
        st.sidebar.error(f"Error loading dataset: {str(e)}")

# Display dataset summary
if st.session_state.data_loaded:
    st.sidebar.markdown("### Dataset Summary")
    Re_vals = st.session_state.Re_values
    X = st.session_state.X
    N = X.shape[1] // 3
    
    col1, col2 = st.sidebar.columns(2)
    col1.metric("Snapshots", X.shape[0])
    col2.metric("Nodes", N)
    
    col1, col2 = st.sidebar.columns(2)
    col1.metric("Min Re", f"{Re_vals.min():.0f}")
    col2.metric("Max Re", f"{Re_vals.max():.0f}")
    
    st.sidebar.info(f"Re range: [{Re_vals.min():.1f}, {Re_vals.max():.1f}]")

st.sidebar.markdown("---")

# Hyperparameters
st.sidebar.subheader("⚙️ Hyperparameters")

pod_modes = st.sidebar.slider(
    "POD Modes (r)",
    min_value=5,
    max_value=100,
    value=30,
    step=5,
    help="Number of dominant modes to retain"
)

nn_width = st.sidebar.slider(
    "NN Width",
    min_value=32,
    max_value=512,
    value=128,
    step=32,
    help="Hidden layer width"
)

nn_depth = st.sidebar.slider(
    "NN Depth",
    min_value=2,
    max_value=8,
    value=4,
    step=1,
    help="Number of hidden layers"
)

learning_rate = st.sidebar.selectbox(
    "Learning Rate",
    [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
    index=2,
    help="AdamW optimizer learning rate"
)

epochs = st.sidebar.slider(
    "Training Epochs",
    min_value=100,
    max_value=10000,
    value=6000,
    step=100,
    help="Number of training epochs"
)

device = st.sidebar.radio(
    "Compute Device",
    ["cpu", "cuda" if torch.cuda.is_available() else "cpu"],
    help="PyTorch device for training"
)

st.sidebar.markdown("---")


# ============================================================================
# Main Content: Tabs
# ============================================================================

tab1, tab2 = st.tabs(["🔬 POD Training & Analysis", "🎯 Interactive Inference"])


# ============================================================================
# TAB 1: POD Training & Analysis
# ============================================================================

with tab1:
    st.header("POD Training & Analysis")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load a dataset first from the sidebar.")
    else:
        col_train, col_spacer = st.columns([1, 2])
        
        with col_train:
            st.subheader("Training")
            
            # Train button
            train_btn = st.button(
                "▶️ Run POD & Train Network",
                key="train_btn",
                disabled=st.session_state.training_in_progress
            )
            
            if train_btn:
                st.session_state.training_in_progress = True
                
                try:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # POD Decomposition
                    status_text.text("Computing POD...")
                    progress_bar.progress(10)
                    
                    pod = fit_pod(
                        st.session_state.X,
                        r=pod_modes,
                        xy=st.session_state.xy
                    )
                    st.session_state.pod = pod
                    
                    status_text.text("Projecting snapshots onto POD basis...")
                    progress_bar.progress(20)
                    
                    pod_coeffs = pod_project(pod, st.session_state.X)
                    st.session_state.pod_coeffs = pod_coeffs
                    
                    # Neural Network Training
                    status_text.text("Initializing neural network...")
                    progress_bar.progress(30)
                    
                    trainer = PODFCDNNTrainer(
                        pod=pod,
                        Re_values=st.session_state.Re_values,
                        pod_coeffs=pod_coeffs,
                        nn_width=nn_width,
                        nn_depth=nn_depth,
                        lr=learning_rate,
                        weight_decay=1e-4,
                        device=device
                    )
                    
                    # Custom callback for progress updates
                    def progress_callback(epoch, loss):
                        progress = min(30 + int(60 * epoch / epochs), 95)
                        progress_bar.progress(progress)
                        status_text.text(
                            f"Training NN... Epoch {epoch}/{epochs} | Loss: {loss:.6e}"
                        )
                    
                    status_text.text("Training neural network...")
                    training_history = trainer.train(
                        epochs=epochs,
                        batch_size=32,
                        verbose=False,
                        callback=progress_callback
                    )
                    
                    st.session_state.trainer = trainer
                    st.session_state.training_history = training_history
                    st.session_state.model_trained = True
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Training complete!")
                    
                    st.success("Model trained successfully!")
                    
                except Exception as e:
                    st.error(f"Training failed: {str(e)}")
                
                finally:
                    st.session_state.training_in_progress = False
        
        # Display results
        if st.session_state.model_trained and st.session_state.pod is not None:
            st.markdown("---")
            st.subheader("📊 POD Analysis")
            
            col_energy, col_loss = st.columns(2)
            
            # Energy plot
            with col_energy:
                st.markdown("#### Cumulative Energy vs. Modes")
                
                pod = st.session_state.pod
                energy_frac = pod.energy_fraction
                
                fig_energy = go.Figure()
                fig_energy.add_trace(go.Scatter(
                    x=np.arange(1, len(energy_frac) + 1),
                    y=energy_frac * 100,
                    mode="lines+markers",
                    name="Cumulative Energy",
                    line=dict(color="royalblue", width=2),
                    marker=dict(size=6)
                ))
                
                fig_energy.update_layout(
                    xaxis_title="Number of Modes",
                    yaxis_title="Cumulative Energy (%)",
                    hovermode="x unified",
                    height=400,
                    template="plotly_white"
                )
                
                st.plotly_chart(fig_energy, use_container_width=True)
                
                # Energy statistics
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Energy (5 modes)", f"{energy_frac[min(4, pod.r-1)]*100:.2f}%")
                col_b.metric("Energy (10 modes)", f"{energy_frac[min(9, pod.r-1)]*100:.2f}%")
                col_c.metric("Energy (all modes)", f"{energy_frac[-1]*100:.2f}%")
            
            # Training loss plot
            with col_loss:
                st.markdown("#### Training Loss Curve")
                
                history = st.session_state.training_history
                losses = history["loss"]
                
                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scatter(
                    y=losses,
                    mode="lines",
                    name="Training Loss",
                    line=dict(color="orangered", width=2)
                ))
                
                fig_loss.update_layout(
                    xaxis_title="Epoch",
                    yaxis_title="MSE Loss",
                    yaxis_type="log",
                    hovermode="x unified",
                    height=400,
                    template="plotly_white"
                )
                
                st.plotly_chart(fig_loss, use_container_width=True)
                
                # Loss statistics
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Initial Loss", f"{losses[0]:.6e}")
                col_b.metric("Final Loss", f"{losses[-1]:.6e}")
                col_c.metric("Improvement", f"{losses[0]/losses[-1]:.1f}x")


# ============================================================================
# TAB 2: Interactive Inference & Visualization
# ============================================================================

with tab2:
    st.header("Interactive Inference & Field Visualization")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train a model first in the 'POD Training & Analysis' tab.")
    else:
        trainer = st.session_state.trainer
        Re_min = st.session_state.Re_values.min()
        Re_max = st.session_state.Re_values.max()
        
        # Reynolds number selector
        st.subheader("🎮 Control Panel")
        
        col_re, col_pred = st.columns([2, 1])
        
        with col_re:
            Re_query = st.slider(
                "Reynolds Number (Re)",
                min_value=float(Re_min),
                max_value=float(Re_max),
                value=float((Re_min + Re_max) / 2),
                step=50.0,
                help=f"Query Reynolds number in range [{Re_min:.0f}, {Re_max:.0f}]"
            )
        
        with col_pred:
            predict_btn = st.button("🔮 Predict", key="predict_btn", use_container_width=True)
        
        # Inference
        if predict_btn or "last_prediction" not in st.session_state:
            try:
                with st.spinner("Running inference..."):
                    result = predict_and_reconstruct(trainer, Re_query)
                    st.session_state.last_prediction = result
                    st.success(f"✅ Prediction at Re = {Re_query:.1f}")
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
        
        # Display predictions
        if "last_prediction" in st.session_state:
            result = st.session_state.last_prediction
            
            # For comparison, load reference data if available
            ref_data_dir = Path(data_dir_input).parent / "cavityflow evaluation data"
            ref_available = False
            ref_result = None
            
            # Try to load reference data for comparison
            try:
                # Look for nearest training Re in evaluation data
                possible_Re_files = list(ref_data_dir.glob("Re *.dat"))
                if possible_Re_files:
                    # Find closest Re in training data for reference
                    closest_Re_idx = np.argmin(np.abs(st.session_state.Re_values - Re_query))
                    closest_Re = st.session_state.Re_values[closest_Re_idx]
                    
                    ref_file = ref_data_dir / f"Re {int(closest_Re)}.dat"
                    if ref_file.exists():
                        xvec_ref, xy_ref = load_snapshot_uvp(ref_file)
                        N_nodes = xy_ref.shape[0]
                        u_ref = xvec_ref[:N_nodes]
                        v_ref = xvec_ref[N_nodes:2*N_nodes]
                        p_ref = xvec_ref[2*N_nodes:3*N_nodes]
                        
                        ref_result = {
                            "u": u_ref,
                            "v": v_ref,
                            "p": p_ref,
                            "Re": closest_Re
                        }
                        ref_available = True
            except:
                pass
            
            st.markdown("---")
            st.subheader("📈 Field Visualization")
            
            # Velocity magnitude
            u_pred = result["u"]
            v_pred = result["v"]
            p_pred = result["p"]
            xy = result["xy"]
            
            vel_mag = np.sqrt(u_pred**2 + v_pred**2)
            
            # Create visualizations using plotly with subplots
            x_coords = xy[:, 0]
            y_coords = xy[:, 1]
            
            # Row 1: Pressure
            fig_p = go.Figure()
            
            fig_p.add_trace(go.Scatter(
                x=x_coords,
                y=y_coords,
                mode="markers",
                marker=dict(
                    size=4,
                    color=p_pred,
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Pressure")
                ),
                text=[f"p: {p:.4f}" for p in p_pred],
                hoverinfo="text",
                name="Pressure"
            ))
            
            fig_p.update_layout(
                title=f"Predicted Pressure at Re = {Re_query:.1f}",
                xaxis_title="x",
                yaxis_title="y",
                height=500,
                template="plotly_white",
                showlegend=False
            )
            fig_p.update_yaxes(scaleanchor="x", scaleratio=1)
            
            st.plotly_chart(fig_p, use_container_width=True)
            
            # Row 2: U-velocity
            fig_u = go.Figure()
            
            fig_u.add_trace(go.Scatter(
                x=x_coords,
                y=y_coords,
                mode="markers",
                marker=dict(
                    size=4,
                    color=u_pred,
                    colorscale="RdBu_r",
                    showscale=True,
                    colorbar=dict(title="u-velocity")
                ),
                text=[f"u: {u:.4f}" for u in u_pred],
                hoverinfo="text",
                name="U-Velocity"
            ))
            
            fig_u.update_layout(
                title=f"Predicted x-Velocity at Re = {Re_query:.1f}",
                xaxis_title="x",
                yaxis_title="y",
                height=500,
                template="plotly_white",
                showlegend=False
            )
            fig_u.update_yaxes(scaleanchor="x", scaleratio=1)
            
            st.plotly_chart(fig_u, use_container_width=True)
            
            # Row 3: V-velocity
            fig_v = go.Figure()
            
            fig_v.add_trace(go.Scatter(
                x=x_coords,
                y=y_coords,
                mode="markers",
                marker=dict(
                    size=4,
                    color=v_pred,
                    colorscale="RdBu_r",
                    showscale=True,
                    colorbar=dict(title="v-velocity")
                ),
                text=[f"v: {v:.4f}" for v in v_pred],
                hoverinfo="text",
                name="V-Velocity"
            ))
            
            fig_v.update_layout(
                title=f"Predicted y-Velocity at Re = {Re_query:.1f}",
                xaxis_title="x",
                yaxis_title="y",
                height=500,
                template="plotly_white",
                showlegend=False
            )
            fig_v.update_yaxes(scaleanchor="x", scaleratio=1)
            
            st.plotly_chart(fig_v, use_container_width=True)
            
            # Statistics
            st.markdown("---")
            st.subheader("📊 Field Statistics Summary")

            # 1. Create a compact dictionary of your results
            stats_dict = {
                "Metric": ["Min", "Max", "Mean"],
                "u-velocity": [f"{u_pred.min():.4f}", f"{u_pred.max():.4f}", f"{u_pred.mean():.4f}"],
                "v-velocity": [f"{v_pred.min():.4f}", f"{v_pred.max():.4f}", f"{v_pred.mean():.4f}"],
                "Pressure": [f"{p_pred.min():.4f}", f"{p_pred.max():.4f}", f"{p_pred.mean():.4f}"]
            }
            st.table(stats_dict)

# ============================================================================
# Footer
# ============================================================================

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 12px;'>
        POD-FCDNN Surrogate Model | Powered by Streamlit, PyTorch & Scikit-Learn
    </div>
    """,
    unsafe_allow_html=True
)
