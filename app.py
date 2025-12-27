import streamlit as st
import cv2
import pandas as pd
import tempfile
import time
import os
import sys

# Helper function to load model (Cached)
@st.cache_resource
def load_model():
    # Add local libs to path (for ultralytics)
    libs_path = os.path.abspath('libs')
    if libs_path not in sys.path:
        sys.path.append(libs_path)
    
    from ultralytics import YOLO
    model = YOLO('yolov8n.pt')
    return model

# Custom CSS for styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .metric-value {
        font-size: 32px;
        font-weight: bold;
        color: #ff4b4b;
    }
    .metric-label {
        font-size: 16px;
        color: #333;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to process video
def process_video(video_path, line_position, confidence, placeholders):
    cap = cv2.VideoCapture(video_path)
    # Load model from cache
    model = load_model()
    
    # Session state for counters
    if 'counters' not in st.session_state:
        st.session_state.counters = {
            'Car': set(),
            'Truck': set(),
            'Bus': set(),
            'Motorcycle': set()
        }
    
    # Reset counters for new run
    counters = {
        'Car': set(),
        'Truck': set(),
        'Bus': set(),
        'Motorcycle': set()
    }
    
    # Mapping class IDs to names
    class_names = {2: 'Car', 3: 'Motorcycle', 5: 'Bus', 7: 'Truck'}
    
    # Layout placeholders
    col1, col2 = st.columns([2, 1])
    with col1:
        st_frame = st.empty()
    with col2:
        # Real-time data table
        st_data_table = st.empty()
        
    data_log = []
    
    offset = 6
    
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Resize frame for speed (640px width - Standard limit for YOLOv8n)
        original_width = frame.shape[1]
        target_width = 640
        Aspect_Ratio = target_width / original_width
        # Only resize if necessary
        if original_width > target_width:
            original_height = frame.shape[0]
            target_height = int(original_height * Aspect_Ratio)
            frame = cv2.resize(frame, (target_width, target_height))
            
            # Scale the line position to match new size
            # We use a distinct variable so we don't mess up the slider logic
            current_line_pos = int(line_position * Aspect_Ratio)
            current_offset = int(offset * Aspect_Ratio) 
        else:
            current_line_pos = line_position
            current_offset = offset
            
        # YOLO Tracking
        # imgsz=640 is standard for YOLOv8n
        results = model.track(frame, persist=True, conf=confidence, verbose=False)
        
        # Draw line
        # Line width needs to match the frame width
        frame_w = frame.shape[1]
        cv2.line(frame, (25, current_line_pos), (frame_w - 25, current_line_pos), (255, 127, 0), 3)
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            
            for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                if class_id in class_names:
                    vehicle_type = class_names[class_id]
                    x, y, w, h = box
                    cx, cy = int(x), int(y)
                    
                    # Draw
                    x1, y1 = int(x - w/2), int(y - h/2)
                    cv2.rectangle(frame, (x1, y1), (int(x+w/2), int(y+h/2)), (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                    cv2.putText(frame, f"{vehicle_type} ID:{track_id}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Counting Logic
                    if current_line_pos - current_offset < cy < current_line_pos + current_offset:
                        if track_id not in counters[vehicle_type]:
                            counters[vehicle_type].add(track_id)
                            cv2.line(frame, (25, current_line_pos), (frame_w - 25, current_line_pos), (0, 127, 255), 3)
                            
                            # Log data
                            data_log.append({
                                'Timestamp': time.strftime("%H:%M:%S"),
                                'Vehicle ID': track_id,
                                'Type': vehicle_type
                            })
                            
        # Update metrics (Every processed frame)
        car_count = len(counters['Car'])
        truck_count = len(counters['Truck'])
        bus_count = len(counters['Bus'])
        moto_count = len(counters['Motorcycle'])
        
        placeholders[0].markdown(f'<div class="metric-card"><div class="metric-label">Cars 🚗</div><div class="metric-value">{car_count}</div></div>', unsafe_allow_html=True)
        placeholders[1].markdown(f'<div class="metric-card"><div class="metric-label">Trucks 🚛</div><div class="metric-value">{truck_count}</div></div>', unsafe_allow_html=True)
        placeholders[2].markdown(f'<div class="metric-card"><div class="metric-label">Buses 🚌</div><div class="metric-value">{bus_count}</div></div>', unsafe_allow_html=True)
        placeholders[3].markdown(f'<div class="metric-card"><div class="metric-label">Motorcycles 🏍️</div><div class="metric-value">{moto_count}</div></div>', unsafe_allow_html=True)

        # Skip display Update to every 4th frame to improve speed
        # We still process tracking every frame, but only send image to browser occasionally
        if frame_count % 4 == 0:
            # Convert BGR to RGB for Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st_frame.image(frame_rgb, channels="RGB", use_container_width=True)
        
    cap.release()
    return counters, data_log

# Sidebar
st.sidebar.title("🔧 Settings")
uploaded_file = st.sidebar.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])
confidence = st.sidebar.slider("Model Confidence", 0.1, 1.0, 0.4)
line_pos = st.sidebar.slider("Line Position", 100, 1000, 550)

# Main Page
st.title("🚦 AI Traffic Analytics Dashboard")
st.markdown("### Final Year Project - Vehicle Detection & Counting System")

if uploaded_file is not None:
    # Save uploaded file momentarily
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        ph_car = st.empty()
        ph_car.markdown('<div class="metric-card"><div class="metric-label">Cars 🚗</div><div class="metric-value" id="count-car">0</div></div>', unsafe_allow_html=True)
    with col2:
        ph_truck = st.empty()
        ph_truck.markdown('<div class="metric-card"><div class="metric-label">Trucks 🚛</div><div class="metric-value" id="count-truck">0</div></div>', unsafe_allow_html=True)
    with col3:
        ph_bus = st.empty()
        ph_bus.markdown('<div class="metric-card"><div class="metric-label">Buses 🚌</div><div class="metric-value" id="count-bus">0</div></div>', unsafe_allow_html=True)
    with col4:
        ph_moto = st.empty()
        ph_moto.markdown('<div class="metric-card"><div class="metric-label">Motorcycles 🏍️</div><div class="metric-value" id="count-moto">0</div></div>', unsafe_allow_html=True)

    if st.sidebar.button("▶️ Start Analysis"):
        with st.spinner('Processing Video...'):
            final_counts, logs = process_video(tfile.name, line_pos, confidence, [ph_car, ph_truck, ph_bus, ph_moto])
            
        st.success("Analysis Complete!")
        
        # Metrics Update (Final)
        st.metric("Total Cars", len(final_counts['Car']))
        st.metric("Total Trucks", len(final_counts['Truck']))
        
        # Data Export
        if logs:
            df = pd.DataFrame(logs)
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
    st.info("👈 Please upload a video file to begin analysis.")
