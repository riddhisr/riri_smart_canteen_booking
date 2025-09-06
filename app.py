import streamlit as st
import tempfile
import os
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
from datetime import datetime, timedelta
import threading
from twilio.rest import Client
import qrcode
from io import BytesIO

# --- Load YOLOv8 Model ---
model = YOLO("runs/obb/train/weights/best.pt")  # Update with your .pt path

# --- Session State ---
if 'booking_status' not in st.session_state:
    st.session_state.booking_status = {}

booking_status = st.session_state.booking_status
lock = threading.Lock()

# --- Timer Utility ---
def get_time_left(expiry_time):
    current_time = datetime.now()
    time_left = expiry_time - current_time
    return str(time_left).split('.')[0] if time_left.total_seconds() > 0 else "Expired"

# --- Generate QR Code with expiry ---
def generate_qr_code(content):
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(content)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    return img

# --- Convert PIL Image to Bytes ---
def pil_image_to_bytes(img):
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr

# --- Send SMS ---
def send_sms_to_user(phone_number, table_id):
    account_sid =  '' # Replace with your Twilio Account SID
    auth_token =   ''  # Replace with your Twilio Auth Token
    from_phone_number = ''  # Replace with your Twilio phone number

    client = Client(account_sid, auth_token)
    message = client.messages.create(
        body=f"✅ Table {table_id} booked successfully! Show your QR at the canteen.",
        from_=from_phone_number,
        to=phone_number
    )
    return message.sid

# --- YOLOv8 Prediction ---
def predict_and_show(file_path, is_image=True):
    results = model.predict(source=file_path, conf=0.4, save=True, project="runs", name="streamlit_output", exist_ok=True)
    result = results[0]
    out_path = os.path.join(result.save_dir, os.path.basename(file_path))
    return out_path, result

# --- Parse Results ---
def get_table_counts(result):
    empty, occupied = [], []
    if result.obb is not None:
        for i, box in enumerate(result.obb):
            label = model.names[int(box.cls)].lower()
            table_id = f"Table {i+1}"
            if "empty" in label:
                empty.append(table_id)
            else:
                occupied.append(table_id)
    return empty, occupied

# --- Heatmap Simulation (color overlay on bounding boxes) ---
def draw_heatmap_overlay(image_path, result):
    img = cv2.imread(image_path)
    if result.obb is not None:
        for i, box in enumerate(result.obb):
            coords = box.xyxy[0].cpu().numpy().astype(int)
            color = (0, 255, 0) if 'empty' in model.names[int(box.cls)].lower() else (0, 0, 255)  # Green for empty, Red for occupied
            overlay = img.copy()
            cv2.rectangle(overlay, coords[:2], coords[2:], color, -1)
            alpha = 0.3  # Transparency
            img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
            # Label only with the table number (1, 2, 3, 4...)
            cv2.putText(img, f"{i+1}", coords[:2], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img

# --- Streamlit UI ---
st.set_page_config(page_title="Smart Canteen", layout="wide")
st.title("🍽 Smart Canteen Table Booking")
st.markdown("Upload an image of the canteen to detect empty/occupied tables and book dynamically.")

# Sidebar Inputs
st.sidebar.title("Reservation Panel")
group_size = st.sidebar.number_input("Number of People", min_value=1, max_value=20, value=2)
phone_number = st.sidebar.text_input("📱 Phone Number")
uploaded_file = st.file_uploader("Upload Canteen Image", type=['jpg', 'jpeg', 'png'])

# --- Main Logic ---
if uploaded_file:
    is_image = uploaded_file.type.startswith("image")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    st.info("🚀 Running detection model...")
    output_path, result = predict_and_show(temp_path, is_image=True)

    empty_tables, occupied_tables = get_table_counts(result)

    st.success(f"✅ {len(empty_tables)} empty | ❌ {len(occupied_tables)} occupied")

    # Show Heatmap
    heatmap_img = draw_heatmap_overlay(temp_path, result)
    st.image(heatmap_img, caption="🧠 Detection Heatmap Overlay", use_column_width=True)

    # Show Booking Options
    st.subheader("🪑 Book a Table")

    available = []
    for table in empty_tables:
        status = booking_status.get(table)
        if not status or get_time_left(status['expires']) == "Expired":
            available.append(table)

    selected_table = st.selectbox("Available Tables", available if available else ["No tables available"])

    if selected_table != "No tables available" and st.button("✅ Claim This Table"):
        if not phone_number:
            st.warning("⚠ Please enter your phone number.")
        else:
            expiry_time = datetime.now() + timedelta(minutes=5)
            qr_data = f"Booking Confirmation: {selected_table}, Group Size: {group_size}, Expires at: {expiry_time.strftime('%H:%M:%S')}"
            qr_img = generate_qr_code(qr_data)

            # Convert the QR code image to bytes
            qr_img_bytes = pil_image_to_bytes(qr_img)

            # Save booking
            booking_status[selected_table] = {
                'status': 'booked',
                'expires': expiry_time,
                'phone': phone_number,
                'group': group_size
            }

            # SMS & QR Display
            sid = send_sms_to_user(phone_number, selected_table)
            if sid:
                st.success(f"📨 SMS Sent for {selected_table}")
                st.image(qr_img_bytes, caption="📎 Your Booking QR (Valid for 5 minutes)", use_column_width=False)

    # Booking Summary
    st.subheader("📋 Booking Status Summary")
    booked, expired, available_now = [], [], []

    for table in empty_tables:
        status = booking_status.get(table)
        if not status or get_time_left(status['expires']) == "Expired":
            available_now.append(table)
        else:
            time_left = get_time_left(status['expires'])
            if status['status'] == 'booked' and time_left != "Expired":
                booked.append(f"{table} (expires in {time_left})")

    if booked:
        st.success("📅 Booked tables: " + ", ".join(booked))
    if available_now:
        st.info("🟩 Available tables: " + ", ".join(available_now))
    if occupied_tables:
        st.error("❌ Occupied tables: " + ", ".join(occupied_tables))