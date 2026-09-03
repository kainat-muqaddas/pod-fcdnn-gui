"""
POD-FCDNN Streamlit Web Application (High-Performance Edition)
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from engine import load_checkpoint, predict_and_reconstruct

# ------------------------------------------------------------
# Page Configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="POD-FCDNN Surrogate Model",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌊 POD-FCDNN Fluid Dynamics Surrogate Model")
st.markdown("Train and deploy a neural network surrogate model for rapid CFD prediction.")

# ------------------------------------------------------------
# Cached Model Loader
# ------------------------------------------------------------
@st.cache_resource(show_spinner="Loading checkpoint...")
def get_cached_model(case_name: str):
    checkpoint_paths = {
        "Cavity": "checkpoints/cavity_checkpoint.pt",
        "Cylinder": "checkpoints/cylinder_checkpoint.pt",
        "Backward Facing Step": "checkpoints/bfs_checkpoint.pt",
        "NACA0012": "checkpoints/naca_checkpoint.pt"
    }
    return load_checkpoint(checkpoint_paths[case_name])

# ------------------------------------------------------------
# High-Performance Visualization Helper
# ------------------------------------------------------------
def create_fast_field_plot(x, y, values, title, colorscale="Viridis", max_display_nodes=30000):
    """
    Renders high-density spatial point data efficiently using WebGL and decimation.
    """
    n_nodes = len(x)
    
    # Subsample spatial points if grid density exceeds threshold
    if n_nodes > max_display_nodes:
        idx = np.random.choice(n_nodes, size=max_display_nodes, replace=False)
        x_plot, y_plot, values_plot = x[idx], y[idx], values[idx]
    else:
        x_plot, y_plot, values_plot = x, y, values

    fig = go.Figure()
    
    # Use Scattergl (WebGL) instead of standard Scatter (SVG)
    fig.add_trace(
        go.Scattergl(
            x=x_plot,
            y=y_plot,
            mode="markers",
            marker=dict(
                size=3,
                color=values_plot,
                colorscale=colorscale,
                showscale=True,
                colorbar=dict(thickness=15, len=0.8)
            )
        )
    )
    
    fig.update_layout(
        title=title,
        xaxis=dict(title="x", constrain="domain"),
        yaxis=dict(title="y", scaleanchor="x", scaleratio=1),
        height=550,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

# ------------------------------------------------------------
# Inputs & Case Selection
# ------------------------------------------------------------
st.header("Flow Field Prediction")

case = st.selectbox(
    "Select Case",
    ["Cavity", "Cylinder", "Backward Facing Step", "NACA0012"]
)

if case == "NACA0012":
    param = st.slider("Angle of Attack (α)", -5.0, 15.0, 0.0, 0.5)
else:
    param = st.slider("Reynolds Number", 100, 10000, 1000, 100)

predict_btn = st.button("Predict Flow Field", use_container_width=True)

# ------------------------------------------------------------
# Inference Run & State Cache
# ------------------------------------------------------------
if predict_btn:
    try:
        with st.spinner("Running POD-FCDNN inference..."):
            trainer = get_cached_model(case)
            res = predict_and_reconstruct(trainer, param)
            
            # Cache results in session state to prevent lost computations on render
            st.session_state["cached_prediction"] = res
            st.session_state["cached_case"] = case
            st.session_state["cached_param"] = param

    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

# ------------------------------------------------------------
# Display Output
# ------------------------------------------------------------
if "cached_prediction" in st.session_state:
    res = st.session_state["cached_prediction"]
    
    # Ensure cached prediction belongs to currently active case
    if st.session_state.get("cached_case") == case:
        u, v, p = res["u"], res["v"], res["p"]
        xy = res["xy"]
        x_coords, y_coords = xy[:, 0], xy[:, 1]

        st.success(f"Displaying prediction for {case} (Param: {st.session_state.get('cached_param')})")

        # Organize plots into tabs for better GPU memory management
        tab_p, tab_u, tab_v = st.tabs(["Pressure Field", "U Velocity", "V Velocity"])

        with tab_p:
            fig_p = create_fast_field_plot(x_coords, y_coords, p, f"{case} Pressure Field", "Viridis")
            st.plotly_chart(fig_p, use_container_width=True)

        with tab_u:
            fig_u = create_fast_field_plot(x_coords, y_coords, u, f"{case} U Velocity", "RdBu_r")
            st.plotly_chart(fig_u, use_container_width=True)

        with tab_v:
            fig_v = create_fast_field_plot(x_coords, y_coords, v, f"{case} V Velocity", "RdBu_r")
            st.plotly_chart(fig_v, use_container_width=True)
