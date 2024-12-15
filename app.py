import streamlit as st
import cv2
import torch
import numpy as np
from PIL import Image

# Load YOLO model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Process each image to count vehicles and calculate density
def process_image(img_path):
    image = cv2.imread(img_path)
    image = cv2.resize(image, (640, 480))
    results = model(image)
    detections = results.pandas().xyxy[0]
    
    # Filter for general vehicle types
    vehicle_detections = detections[detections['name'].isin(['car', 'truck', 'bus'])]
    vehicle_count = len(vehicle_detections)
    density = vehicle_count / (640 * 480)  # Vehicle density (vehicles per pixel)

    # Remove confidence score by editing the label text
    results.render()  # Render once to update labels
    for i in range(len(results.imgs[0])):
        # Remove the confidence part from each label
        results.pandas().xyxy[0].loc[i, 'class'] = vehicle_detections['name'].iloc[i]

    return vehicle_count, density, results

# Calculate open times based on vehicle counts
def calculate_open_times(vehicle_counts):
    total_count = sum(vehicle_counts)
    open_times = [(count / total_count) * 60 for count in vehicle_counts]  # Scale to 60 seconds total
    return open_times

# Streamlit UI code
st.title("Enhanced Traffic Signal Management System")
st.write("Upload 4 traffic images to analyze vehicle density, set signal open times, and determine opening order.")

# Upload images
uploaded_files = st.file_uploader("Choose 4 traffic images", accept_multiple_files=True, type=["jpg", "jpeg", "png"])

if uploaded_files and len(uploaded_files) == 4:
    vehicle_counts = []
    densities = []
    results_list = []

    # Process each image and display vehicle count and density
    for i, uploaded_file in enumerate(uploaded_files):
        img = Image.open(uploaded_file)
        img_array = np.array(img)  # Convert to NumPy array for OpenCV
        cv2.imwrite(f'temp_image_{i}.jpg', img_array)  # Save temporarily for OpenCV processing

        # Process image and get results
        vehicle_count, density, results = process_image(f'temp_image_{i}.jpg')
        
        # Append to lists
        vehicle_counts.append(vehicle_count)
        densities.append(density)
        results_list.append(results)

        # Display results with labels without confidence scores
        st.image(results.render()[0], caption=f"Signal {i+1}: {vehicle_count} vehicles")

    # Calculate open times and priority order based on density
    open_times = calculate_open_times(vehicle_counts)
    priority_order = sorted(range(len(densities)), key=lambda k: densities[k], reverse=True)

    # Display recommended order and open times
    st.write("**Recommended Signal Open Times and Order**")
    for i, idx in enumerate(priority_order):
        st.write(f"Open Signal {idx+1} first for {open_times[idx]:.2f} seconds")

else:
    st.warning("Please upload exactly 4 images.")
