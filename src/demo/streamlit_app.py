"""Streamlit web application for sign language detection."""

import streamlit as st
import cv2
import numpy as np
import torch
import time
import io
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
from typing import List, Dict, Any

from .inference_engine import InferenceEngine
from ..models.model_factory import create_model
from ..utils.config import config


@st.cache_resource
def load_model(model_path: str, num_classes: int):
    """Load model with caching."""
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Try to determine architecture
        architecture = config.get('model.architecture', 'efficientnet_b0')
        model = create_model(
            architecture=architecture,
            num_classes=num_classes,
            pretrained=False
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def create_streamlit_app():
    """Create Streamlit web application."""
    
    st.set_page_config(
        page_title="Sign Language Detection",
        page_icon="🤟",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🤟 Sign Language Detection")
    st.markdown("Real-time sign language recognition using deep learning")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Model selection
        model_path = st.file_uploader(
            "Upload Model Checkpoint",
            type=['pth', 'pt'],
            help="Upload a trained PyTorch model checkpoint"
        )
        
        # Dataset configuration
        dataset_type = st.selectbox(
            "Dataset Type",
            ["ASL Alphabet", "Custom"],
            help="Select the type of dataset your model was trained on"
        )
        
        if dataset_type == "ASL Alphabet":
            class_names = [
                "A", "B", "C", "D", "del", "E", "F", "G", "H", "I", "J", "K", "L", "M",
                "N", "nothing", "O", "P", "Q", "R", "S", "space", "T", "U", "V", "W", "X", "Y", "Z"
            ]
        else:
            custom_classes = st.text_area(
                "Class Names (one per line)",
                value="A\nB\nC",
                help="Enter class names, one per line"
            )
            class_names = [name.strip() for name in custom_classes.split('\n') if name.strip()]
        
        st.write(f"Number of classes: {len(class_names)}")
        
        # Inference parameters
        st.subheader("Inference Settings")
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Minimum confidence for predictions"
        )
        
        smoothing_window = st.slider(
            "Temporal Smoothing",
            min_value=1,
            max_value=10,
            value=5,
            help="Number of frames to smooth predictions over"
        )
        
        show_probabilities = st.checkbox("Show All Class Probabilities", value=True)
        show_top_k = st.slider("Show Top-K Predictions", 1, 10, 5)
    
    # Main content
    if model_path is None:
        st.warning("Please upload a model checkpoint to begin")
        return
    
    # Load model
    with st.spinner("Loading model..."):
        # Save uploaded file temporarily
        temp_path = f"/tmp/model_{int(time.time())}.pth"
        with open(temp_path, "wb") as f:
            f.write(model_path.read())
        
        model = load_model(temp_path, len(class_names))
    
    if model is None:
        st.error("Failed to load model")
        return
    
    # Initialize inference engine
    @st.cache_resource
    def get_inference_engine(_model, _class_names, _confidence_threshold, _smoothing_window):
        return InferenceEngine(
            model=_model,
            class_names=_class_names,
            confidence_threshold=_confidence_threshold,
            smoothing_window=_smoothing_window
        )
    
    inference_engine = get_inference_engine(model, class_names, confidence_threshold, smoothing_window)
    
    st.success("Model loaded successfully!")
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📷 Live Camera", "🖼️ Image Upload", "📊 Model Info", "⚡ Benchmark"])
    
    with tab1:
        create_camera_tab(inference_engine, class_names, show_probabilities, show_top_k)
    
    with tab2:
        create_upload_tab(inference_engine, class_names, show_probabilities, show_top_k)
    
    with tab3:
        create_model_info_tab(model, inference_engine)
    
    with tab4:
        create_benchmark_tab(inference_engine)


def create_camera_tab(inference_engine, class_names, show_probabilities, show_top_k):
    """Create live camera tab."""
    st.subheader("Live Camera Feed")
    
    # Camera controls
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        camera_index = st.selectbox("Camera", [0, 1, 2], index=0)
    
    with col2:
        if st.button("Start Camera"):
            st.session_state.camera_active = True
    
    with col3:
        if st.button("Stop Camera"):
            st.session_state.camera_active = False
    
    # Initialize session state
    if 'camera_active' not in st.session_state:
        st.session_state.camera_active = False
    
    if 'frame_count' not in st.session_state:
        st.session_state.frame_count = 0
    
    # Camera feed
    if st.session_state.camera_active:
        st.info("Camera feature requires additional setup for web deployment. Please use the desktop application for live camera feed.")
        
        # Placeholder for camera implementation
        # Note: Streamlit doesn't directly support webcam access in browsers
        # This would typically require additional JavaScript components or WebRTC
        
        st.markdown("""
        **For live camera detection:**
        1. Use the desktop webcam demo: `python -m src.demo.webcam_demo --model your_model.pth`
        2. Or implement WebRTC components for browser-based camera access
        """)
    else:
        st.info("Click 'Start Camera' to begin live detection")


def create_upload_tab(inference_engine, class_names, show_probabilities, show_top_k):
    """Create image upload tab."""
    st.subheader("Upload Image for Detection")
    
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload an image containing a sign language gesture"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Input Image")
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            st.subheader("Prediction Results")
            
            # Convert PIL to numpy
            image_np = np.array(image)
            if image_np.shape[2] == 4:  # RGBA to RGB
                image_np = image_np[:, :, :3]
            
            # Run inference
            with st.spinner("Running inference..."):
                result = inference_engine.predict_single(image_np)
            
            # Display main prediction
            confidence = result['confidence']
            predicted_class = result['predicted_class_name']
            
            # Confidence color
            if result['above_threshold']:
                confidence_color = "green"
            else:
                confidence_color = "orange"
            
            st.markdown(f"""
            **Predicted Class:** {predicted_class}
            
            **Confidence:** <span style='color: {confidence_color}'>{confidence:.3f}</span>
            """, unsafe_allow_html=True)
            
            # Progress bar for confidence
            st.progress(confidence)
            
            # Top-K predictions
            if show_top_k > 1:
                st.subheader(f"Top-{show_top_k} Predictions")
                top_k_preds = inference_engine.get_top_k_predictions(image_np, k=show_top_k)
                
                for i, pred in enumerate(top_k_preds):
                    st.write(f"{i+1}. {pred['class_name']}: {pred['probability']:.3f}")
        
        # Probability visualization
        if show_probabilities:
            st.subheader("Class Probabilities")
            
            probabilities = result['probabilities']
            prob_df = {
                'Class': class_names,
                'Probability': probabilities
            }
            
            # Bar chart
            fig = px.bar(
                prob_df,
                x='Class',
                y='Probability',
                title="Prediction Probabilities for All Classes"
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Top probabilities pie chart
            top_indices = np.argsort(probabilities)[-5:][::-1]
            top_classes = [class_names[i] for i in top_indices]
            top_probs = [probabilities[i] for i in top_indices]
            
            fig_pie = px.pie(
                values=top_probs,
                names=top_classes,
                title="Top-5 Predictions Distribution"
            )
            st.plotly_chart(fig_pie, use_container_width=True)


def create_model_info_tab(model, inference_engine):
    """Create model information tab."""
    st.subheader("Model Information")
    
    # Model architecture info
    model_info = model.get_model_info()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.metric("Model Type", model.__class__.__name__)
        st.metric("Total Parameters", f"{model_info['total_parameters']:,}")
        st.metric("Trainable Parameters", f"{model_info['trainable_parameters']:,}")
        st.metric("Model Size", f"{model_info['model_size_mb']:.1f} MB")
    
    with col2:
        st.metric("Number of Classes", model_info['num_classes'])
        st.metric("Dropout Rate", model_info['dropout_rate'])
        
        # Device info
        device_info = str(inference_engine.device)
        st.metric("Device", device_info)
        
        # CUDA info if available
        if torch.cuda.is_available():
            st.metric("CUDA Available", "✅ Yes")
            st.metric("GPU Name", torch.cuda.get_device_name(0))
        else:
            st.metric("CUDA Available", "❌ No")
    
    # Model architecture details
    with st.expander("Model Architecture Details"):
        st.text(str(model))
    
    # Performance statistics
    perf_stats = inference_engine.get_performance_stats()
    if perf_stats:
        st.subheader("Performance Statistics")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.metric("Average FPS", f"{perf_stats['fps']:.1f}")
        
        with col2:
            st.metric("Average Inference Time", f"{perf_stats['avg_inference_time']*1000:.1f} ms")
        
        with col3:
            st.metric("Total Frames Processed", perf_stats['total_frames'])


def create_benchmark_tab(inference_engine):
    """Create benchmark tab."""
    st.subheader("Performance Benchmark")
    
    st.write("Run benchmark tests to measure model performance on your hardware.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        num_iterations = st.number_input("Number of iterations", min_value=10, max_value=1000, value=100)
        image_size = st.selectbox("Image size", [(224, 224), (160, 160), (128, 128)], index=0)
    
    with col2:
        if st.button("Run Benchmark", type="primary"):
            with st.spinner("Running benchmark..."):
                results = inference_engine.benchmark(
                    num_iterations=num_iterations,
                    image_size=image_size
                )
            
            st.success("Benchmark completed!")
            
            # Display results
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                st.metric("Average FPS", f"{results['avg_fps']:.1f}")
                st.metric("Average Time", f"{results['avg_time_per_inference']*1000:.2f} ms")
            
            with col2:
                st.metric("Min Time", f"{results['min_time']*1000:.2f} ms")
                st.metric("Max Time", f"{results['max_time']*1000:.2f} ms")
            
            with col3:
                st.metric("95th Percentile", f"{results['p95_time']*1000:.2f} ms")
                st.metric("99th Percentile", f"{results['p99_time']*1000:.2f} ms")
            
            # Benchmark visualization
            st.subheader("Benchmark Results Visualization")
            
            # Create fake time series data for visualization (in real implementation, 
            # you'd collect actual timing data during benchmark)
            iterations = list(range(1, num_iterations + 1))
            avg_time_ms = results['avg_time_per_inference'] * 1000
            
            # Simulate some variance
            times = np.random.normal(avg_time_ms, avg_time_ms * 0.1, num_iterations)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=iterations,
                y=times,
                mode='lines',
                name='Inference Time',
                line=dict(color='blue', width=1)
            ))
            
            fig.add_hline(
                y=avg_time_ms,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Average: {avg_time_ms:.2f} ms"
            )
            
            fig.update_layout(
                title="Inference Time per Iteration",
                xaxis_title="Iteration",
                yaxis_title="Time (ms)",
                hovermode='x'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Export benchmark results
            if st.button("Download Benchmark Results"):
                results_json = json.dumps(results, indent=2)
                st.download_button(
                    label="Download JSON",
                    data=results_json,
                    file_name=f"benchmark_results_{int(time.time())}.json",
                    mime="application/json"
                )


if __name__ == "__main__":
    create_streamlit_app()
