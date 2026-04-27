
#python -m streamlit run app.py

# Remove 'libs' from sys.path and PYTHONPATH at startup — it contains
# packages compiled for Python 3.13 which are incompatible with this interpreter.
# ultralytics will be loaded from libs on-demand inside load_model().


import sys as _sys, os as _os
_libs = _os.path.normcase(_os.path.abspath('libs'))
if 'PYTHONPATH' in _os.environ:
    del _os.environ['PYTHONPATH']
_sys.path[:] = [p for p in _sys.path if _os.path.normcase(p) != _libs]



import streamlit as st
import cv2
import pandas as pd
import tempfile
import time
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Helper function to load model (Cached)
@st.cache_resource
# Helper function to load model (Cached)
@st.cache_resource
def load_model():
    from ultralytics import YOLO
    model = YOLO('yolov8n.pt')
    return model

# Custom CSS for styling
st.markdown("""
<style>
    /* -- Keyframe Animations -- */
    @keyframes fadeInUp {
        0%   { opacity: 0; transform: translateY(30px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    @keyframes pulseGlow {
        0%, 100% { text-shadow: 0 0 8px rgba(255,75,75,0.4); }
        50%      { text-shadow: 0 0 20px rgba(255,75,75,0.9), 0 0 40px rgba(255,75,75,0.3); }
    }
    @keyframes gradientBorder {
        0%   { border-color: #ff4b4b; }
        33%  { border-color: #0083b8; }
        66%  { border-color: #00c49a; }
        100% { border-color: #ff4b4b; }
    }
    @keyframes countUp {
        0%   { opacity: 0; transform: scale(0.5); }
        60%  { transform: scale(1.15); }
        100% { opacity: 1; transform: scale(1); }
    }
    @keyframes shimmer {
        0%   { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }

    /* -- Metric Cards -- */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 22px;
        border-radius: 16px;
        text-align: center;
        min-height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        animation: fadeInUp 0.6s ease-out both;
        border: 2px solid transparent;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(255,75,75,0.15);
    }
    .metric-value {
        font-size: 38px;
        font-weight: 800;
        color: #ff6b6b;
        animation: pulseGlow 2.5s ease-in-out infinite, countUp 0.8s ease-out both;
        letter-spacing: 1px;
    }
    .metric-label {
        font-size: 15px;
        color: #a0aec0;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-top: 6px;
    }

    /* -- Chart Container -- */
    .chart-container {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
        border-radius: 20px;
        padding: 24px;
        border: 2px solid #30363d;
        animation: fadeInUp 0.8s ease-out both, gradientBorder 4s ease infinite;
        box-shadow: 0 16px 48px rgba(0,0,0,0.4);
        margin-bottom: 20px;
    }
    .chart-title {
        color: #e6edf3;
        font-size: 20px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 12px;
        letter-spacing: 1px;
    }

    /* -- Stat Cards Row -- */
    .stat-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 16px;
        padding: 20px 16px;
        text-align: center;
        border: 1px solid #30363d;
        animation: fadeInUp 0.6s ease-out both;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        transition: transform 0.3s ease;
    }
    .stat-card:hover { transform: translateY(-3px); }
    .stat-card:nth-child(1) { animation-delay: 0.1s; }
    .stat-card:nth-child(2) { animation-delay: 0.3s; }
    .stat-card:nth-child(3) { animation-delay: 0.5s; }
    .stat-number {
        font-size: 36px;
        font-weight: 800;
        background: linear-gradient(135deg, #ff6b6b, #ffa500);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: countUp 1s ease-out both;
    }
    .stat-label {
        font-size: 13px;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-top: 4px;
    }
    .stat-bar {
        height: 6px;
        border-radius: 3px;
        background: #21262d;
        margin-top: 12px;
        overflow: hidden;
    }
    .stat-bar-fill {
        height: 100%;
        border-radius: 3px;
        background: linear-gradient(90deg, #ff6b6b, #ffa500);
        animation: shimmer 2s linear infinite;
        background-size: 200% 100%;
    }
    .stat-bar-fill-blue {
        height: 100%;
        border-radius: 3px;
        background: linear-gradient(90deg, #0083b8, #00c49a);
        animation: shimmer 2s linear infinite;
        background-size: 200% 100%;
    }

    /* -- Environment Badge -- */
    .environment-badge {
        padding: 10px;
        border-radius: 8px;
        color: white;
        font-weight: bold;
        text-align: center;
        margin-bottom: 10px;
        animation: fadeInUp 0.5s ease-out both;
    }
    .env-day { background: linear-gradient(135deg, #FFD700, #FFA500); color: #000; }
    .env-night { background: linear-gradient(135deg, #191970, #4B0082); color: #fff; }
</style>
""", unsafe_allow_html=True)

def detect_environment(frame):
    """
    Detects if the environment is Day or Night based on average brightness of the frame.
    Returns: 'Day' or 'Night'
    """
    # Convert to HSV (Hue, Saturation, Value)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Calculate average Value (brightness)
    brightness = np.mean(hsv[:, :, 2])
    
    # Threshold for Day vs Night (Adjustable, typically ~80-100 for video)
    if brightness > 90:
        return 'Day', brightness
    else:
        return 'Night', brightness

# Helper function to process video
def process_video(video_path, line_position, confidence, placeholders, limit_count=0):
    model = load_model()
    class_names = {2: 'Car', 7: 'Truck'}
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    counters = {'Car': set(), 'Truck': set()}
    data_log = []
    
    offset = 10 
    start_time = time.time()
    frame_count = 0
    
    st_frame = st.empty()
    progress_bar = st.empty()
    status_text = st.empty()

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            line_crossed = False
            frame_count += 1
            
            # STOP LIMIT CHECK (Based on Count)
            current_total_count = sum(len(ids) for ids in counters.values())
            if limit_count > 0 and current_total_count >= limit_count:
                break # Stop early
                
            elapsed_time = time.time() - start_time
            
            original_width = frame.shape[1]
            target_width = 640
            aspect_ratio = target_width / original_width
            if original_width > target_width:
                original_height = frame.shape[0]
                target_height = int(original_height * aspect_ratio)
                frame = cv2.resize(frame, (target_width, target_height))
                current_line_pos = int(line_position * aspect_ratio)
                current_offset = int(offset * aspect_ratio)
            else:
                current_line_pos = line_position
                current_offset = offset
                
            # YOLO Tracking
            results = model.track(frame, persist=True, conf=confidence, verbose=False)
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                class_ids = results[0].boxes.cls.int().cpu().tolist()
                
                for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                    if class_id in class_names:
                        vehicle_type = class_names[class_id]
                        x, y, w, h = box
                        cx, cy = int(x), int(y)
                        
                        # Counting Logic (Left side only)
                        limit_x = int(frame.shape[1] / 2)
                        if cx < limit_x and current_line_pos - current_offset < cy < current_line_pos + current_offset:
                            if track_id not in counters[vehicle_type]:
                                counters[vehicle_type].add(track_id)
                                line_crossed = True
                                
                                data_log.append({
                                    'Timestamp': f"{elapsed_time:.2f}s",
                                    'Vehicle ID': track_id,
                                    'Type': vehicle_type
                                })
                        
            if frame_count % 10 == 0:
                car_count = len(counters['Car'])
                truck_count = len(counters['Truck'])
                
                placeholders[0].markdown(f'<div class="metric-card"><div class="metric-label">Cars 🚗</div><div class="metric-value">{car_count}</div></div>', unsafe_allow_html=True)
                placeholders[1].markdown(f'<div class="metric-card"><div class="metric-label">Trucks 🚛</div><div class="metric-value">{truck_count}</div></div>', unsafe_allow_html=True)

            frame_w = frame.shape[1]
            limit_x = int(frame_w / 2) 
            line_color = (0, 127, 255) if line_crossed else (255, 127, 0)
            cv2.line(frame, (25, current_line_pos), (limit_x, current_line_pos), line_color, 3)
            
            if frame_count % 4 == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                st_frame.image(frame_rgb, channels="RGB", width='stretch')
        
        cap.release()
        
        # Final update of metrics to ensure count is correct on UI
        car_count = len(counters['Car'])
        truck_count = len(counters['Truck'])
        

            
        return counters, data_log

    except Exception as e:
        st.error(f"❌ Processing error: {str(e)}")
        return None, []

@st.cache_data
def get_video_preview(uploaded_file):
    """
    Extracts the first frame of the uploaded video for preview.
    Cached to prevent re-reading the file on every slider change.
    """
    try:
        # Create a temp file to read the video (OpenCV requires a path)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
            uploaded_file.seek(0)
            tfile.write(uploaded_file.read())
            temp_path = tfile.name
        
        cap = cv2.VideoCapture(temp_path)
        ret, frame = cap.read()
        cap.release()
        os.unlink(temp_path) # Cleanup
        
        uploaded_file.seek(0) # Reset pointer for other operations
        
        if ret:
            # Convert to RGB here to save processing later, or keep BGR?
            # Keeping BGR for consistency with cv2 drawing, but streamlit needs RGB.
            # Let's return BGR and convert when displaying.
            return frame
        return None
    except Exception as e:
        return None

# Sidebar
st.sidebar.title("🔧 Settings")
uploaded_file = st.sidebar.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])

st.sidebar.divider()
st.sidebar.subheader("⚙️ Configuration")
# Hardcoded settings
confidence = 0.3
line_pos = 550

vehicle_limit = st.sidebar.number_input("🛑 Stop after X Vehicles", min_value=0, value=0, help="Set to 0 for no limit. Set 50 to stop after counting 50 vehicles.")

# Main Page
st.title("🚦 AI Traffic Analytics Dashboard")
st.markdown("### Final Year Project - Vehicle Detection & Counting System")

if uploaded_file is not None:
    # Reset file pointer to beginning (important because sidebar might have read it)
    uploaded_file.seek(0)
    
    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:  # Auto suffix for video
        tfile.write(uploaded_file.read())
        video_path = tfile.name
    
    col1, col2 = st.columns(2)
    with col1:
        ph_car = st.empty()
        ph_car.markdown('<div class="metric-card"><div class="metric-label">Cars 🚗</div><div class="metric-value">0</div></div>', unsafe_allow_html=True)
    with col2:
        ph_truck = st.empty()
        ph_truck.markdown('<div class="metric-card"><div class="metric-label">Trucks 🚛</div><div class="metric-value">0</div></div>', unsafe_allow_html=True)

    # Initial Environment Detection (Preview first frame)
    if 'video_path' in locals():
        cap_env = cv2.VideoCapture(video_path)
        ret, first_frame = cap_env.read()
        if ret:
            env_type, brightness_val = detect_environment(first_frame)
            env_class = "env-day" if env_type == "Day" else "env-night"
            st.markdown(f'<div class="environment-badge {env_class}">Analysis Mode: {env_type} (Brightness: {brightness_val:.1f})</div>', unsafe_allow_html=True)
        cap_env.release()

    if st.sidebar.button("▶️ Start Analysis"):
        with st.spinner('Processing Video...'):
            final_counts, logs = process_video(video_path, line_pos, confidence, [ph_car, ph_truck], limit_count=vehicle_limit)
            
        if final_counts:
            st.success("Analysis Complete! 🎉")
            
            # Full final metrics
            col_a, col_b = st.columns(2)
            with col_a: st.metric("Total Cars", len(final_counts['Car']))
            with col_b: st.metric("Total Trucks", len(final_counts['Truck']))
            
            total_vehicles = sum(len(v) for v in final_counts.values())
            st.metric("Grand Total", total_vehicles)
            
            # --- Visualizations ---
            st.divider()
            st.markdown('<div class="chart-title">📊 Traffic Distribution Analytics</div>', unsafe_allow_html=True)
            
            # Pie Chart Data
            labels = ['Cars 🚗', 'Trucks 🚛']
            sizes = [len(final_counts['Car']), len(final_counts['Truck'])]
            
            if sum(sizes) > 0:
                chart_col1, chart_col2 = st.columns(2)
                
                # -- LEFT: Animated Donut Chart --
                with chart_col1:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    
                    car_pct = sizes[0] / sum(sizes) * 100 if sum(sizes) > 0 else 0
                    truck_pct = sizes[1] / sum(sizes) * 100 if sum(sizes) > 0 else 0
                    
                    # Determine which slice to pull out (the larger one)
                    pull_vals = [0.08, 0.02] if sizes[0] >= sizes[1] else [0.02, 0.08]
                    
                    fig_donut = go.Figure(data=[go.Pie(
                        labels=labels,
                        values=sizes,
                        hole=0.5,
                        pull=pull_vals,
                        marker=dict(
                            colors=['#ff6b6b', '#0083b8'],
                            line=dict(color='#0d1117', width=3),
                            pattern=dict(shape=['', ''])
                        ),
                        textinfo='percent+label',
                        textfont=dict(size=14, color='#e6edf3', family='Inter, sans-serif'),
                        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>',
                        rotation=45,
                        direction='clockwise',
                        opacity=0.95
                    )])
                    
                    # Center annotation
                    fig_donut.update_layout(
                        annotations=[
                            dict(
                                text=f'<b>{total_vehicles}</b><br><span style="font-size:12px;color:#8b949e">Total</span>',
                                x=0.5, y=0.5, font=dict(size=32, color='#ff6b6b', family='Inter, sans-serif'),
                                showarrow=False
                            )
                        ],
                        showlegend=True,
                        legend=dict(
                            orientation='h', yanchor='bottom', y=-0.15, xanchor='center', x=0.5,
                            font=dict(size=13, color='#e6edf3'),
                            bgcolor='rgba(0,0,0,0)'
                        ),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        margin=dict(t=30, b=40, l=20, r=20),
                        height=400,
                        font=dict(family='Inter, sans-serif')
                    )
                    
                    st.plotly_chart(fig_donut, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # -- RIGHT: Animated Horizontal Bar Chart --
                with chart_col2:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    
                    fig_bar = go.Figure()
                    
                    bar_colors = ['#ff6b6b', '#0083b8']
                    bar_labels = ['Cars 🚗', 'Trucks 🚛']
                    
                    for i, (label, size, color) in enumerate(zip(bar_labels, sizes, bar_colors)):
                        fig_bar.add_trace(go.Bar(
                            y=[label],
                            x=[size],
                            orientation='h',
                            marker=dict(
                                color=color,
                                line=dict(color='#e6edf3', width=1),
                                opacity=0.9
                            ),
                            text=f'  {size}',
                            textposition='outside',
                            textfont=dict(size=18, color='#e6edf3', family='Inter, sans-serif'),
                            hovertemplate=f'<b>{label}</b><br>Count: {size}<extra></extra>',
                            showlegend=False
                        ))
                    
                    max_val = max(sizes) if sizes else 1
                    fig_bar.update_layout(
                        xaxis=dict(
                            title=dict(text='Vehicle Count', font=dict(color='#8b949e', size=13)),
                            range=[0, max_val * 1.35],
                            gridcolor='#21262d',
                            tickfont=dict(color='#8b949e', size=12),
                            showline=False
                        ),
                        yaxis=dict(
                            tickfont=dict(color='#e6edf3', size=15),
                            showgrid=False
                        ),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        margin=dict(t=30, b=40, l=10, r=60),
                        height=400,
                        bargap=0.35,
                        font=dict(family='Inter, sans-serif'),
                        # Animation on initial load
                        transition=dict(duration=800, easing='cubic-in-out')
                    )
                    
                    st.plotly_chart(fig_bar, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                # -- Animated Summary Stat Cards --
                st.markdown('<br>', unsafe_allow_html=True)
                stat1, stat2, stat3 = st.columns(3)
                
                car_pct_val = round(sizes[0] / sum(sizes) * 100, 1) if sum(sizes) > 0 else 0
                truck_pct_val = round(sizes[1] / sum(sizes) * 100, 1) if sum(sizes) > 0 else 0
                
                with stat1:
                    st.markdown(f'''
                    <div class="stat-card" style="animation-delay:0.1s">
                        <div class="stat-number">{total_vehicles}</div>
                        <div class="stat-label">Total Vehicles</div>
                        <div class="stat-bar"><div class="stat-bar-fill" style="width:100%"></div></div>
                    </div>
                    ''', unsafe_allow_html=True)
                
                with stat2:
                    st.markdown(f'''
                    <div class="stat-card" style="animation-delay:0.3s">
                        <div class="stat-number">{car_pct_val}%</div>
                        <div class="stat-label">Cars Share</div>
                        <div class="stat-bar"><div class="stat-bar-fill" style="width:{car_pct_val}%"></div></div>
                    </div>
                    ''', unsafe_allow_html=True)
                
                with stat3:
                    st.markdown(f'''
                    <div class="stat-card" style="animation-delay:0.5s">
                        <div class="stat-number">{truck_pct_val}%</div>
                        <div class="stat-label">Trucks Share</div>
                        <div class="stat-bar"><div class="stat-bar-fill-blue" style="width:{truck_pct_val}%"></div></div>
                    </div>
                    ''', unsafe_allow_html=True)

            else:
                st.info("No vehicles detected for chart.")
            
            # Data Export (always show, add summary)
            df = pd.DataFrame(logs)
            
            # Convert Vehicle ID to string to avoid Arrow mixing types error with 'N/A'
            if not df.empty:
                df['Vehicle ID'] = df['Vehicle ID'].astype(str)
            
            summary_row = pd.DataFrame([{
                'Timestamp': 'SUMMARY',
                'Vehicle ID': 'N/A',
                'Type': f'Total: {total_vehicles}'
            }])
            
            df = pd.concat([df, summary_row], ignore_index=True)
            st.dataframe(df)
            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Report (CSV)",
                csv,
                "traffic_report.csv",
                "text/csv",
                key='download-csv'
            )
        else:
            st.warning("No results generated. Try lowering confidence or check video.")
    
    # Cleanup
    if 'video_path' in locals():
        os.unlink(video_path)
else:
    st.info("👈 Please upload a video file to begin analysis.")