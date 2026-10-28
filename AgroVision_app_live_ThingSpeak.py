from pathlib import Path
from io import BytesIO
from datetime import datetime
import json
import sqlite3
import base64
import hashlib
import requests

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageOps

# -------------------- CONFIG --------------------
st.set_page_config(page_title="AgroVision", page_icon="🌱", layout="wide")
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "agrovision_tomato.keras"
DB_PATH = BASE_DIR / "agrovision_batches.db"
CLASS_NAMES = ["Green", "Turning", "Red_Ripe"]


# -------------------- ESP32 LOCAL WI-FI LINK --------------------
ESP32_DEFAULT_URL = "http://10.239.231.42"
REQUEST_TIMEOUT_S = 2.5

def esp32_get_status(base_url):
    try:
        response = requests.get(f"{base_url.rstrip('/')}/api/status", timeout=REQUEST_TIMEOUT_S)
        response.raise_for_status()
        return response.json(), None
    except Exception as exc:
        return None, str(exc)

def esp32_set_crop(base_url, crop_name):
    """Match the current DevKit firmware: GET /api/crop?name=Tomato."""
    try:
        response = requests.get(
            f"{base_url.rstrip('/')}/api/crop",
            params={"name": crop_name},
            timeout=REQUEST_TIMEOUT_S,
        )
        response.raise_for_status()
        return response.json(), None
    except Exception as exc:
        return None, str(exc)


# -------------------- THINGSPEAK CLOUD TELEMETRY --------------------
# Configure these in Streamlit Cloud -> App settings -> Secrets.
# Keep the channel private and use a READ API key here.
def thingspeak_get_latest():
    try:
        channel_id = str(st.secrets["THINGSPEAK_CHANNEL_ID"]).strip()
        read_key = str(st.secrets.get("THINGSPEAK_READ_API_KEY", "")).strip()
        params = {"results": 1}
        if read_key:
            params["api_key"] = read_key
        response = requests.get(
            f"https://api.thingspeak.com/channels/{channel_id}/feeds.json",
            params=params,
            timeout=6,
        )
        response.raise_for_status()
        payload = response.json()
        feeds = payload.get("feeds") or []
        if not feeds:
            return None, "ThingSpeak has no readings yet."
        feed = feeds[-1]
        def number(field):
            value = feed.get(f"field{field}")
            return float(value) if value not in (None, "") else None
        status = {
            "temperature_c": number(1),
            "humidity_pct": number(2),
            "temperature_alert": bool(number(3)),
            "sensor_fault": bool(number(4)),
            "buzzer_on": bool(number(5)),
            "created_at": feed.get("created_at"),
            "source": "ThingSpeak cloud",
        }
        return status, None
    except KeyError:
        return None, "Configure THINGSPEAK_CHANNEL_ID in Streamlit Secrets."
    except Exception as exc:
        return None, f"ThingSpeak read failed: {exc}"

CROP_PROFILES = {
    "Tomato": (8.0, 10.0),
    "Leafy Greens": (2.0, 5.0),
    "Cucumber": (10.0, 12.0),
    "Chilli": (7.0, 10.0),
}


# -------------------- DATABASE --------------------
def connect_db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def init_db():
    with connect_db() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS batches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_code TEXT UNIQUE NOT NULL,
                crop TEXT NOT NULL,
                quantity REAL NOT NULL,
                unit TEXT NOT NULL,
                notes TEXT,
                created_at TEXT NOT NULL,
                latest_maturity TEXT,
                latest_ai_score REAL
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS batch_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_code TEXT NOT NULL,
                event_time TEXT NOT NULL,
                event_type TEXT NOT NULL,
                details TEXT NOT NULL
            )
        """)


init_db()

# Store each automatic ESP32 capture and its AI result separately.
with connect_db() as con:
    con.execute("""
        CREATE TABLE IF NOT EXISTS esp32_captures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_code TEXT NOT NULL,
            captured_at TEXT NOT NULL,
            image_path TEXT NOT NULL,
            maturity TEXT NOT NULL,
            ai_score REAL NOT NULL,
            class_scores TEXT NOT NULL
        )
    """)


def add_event(batch_code, event_type, details):
    with connect_db() as con:
        con.execute(
            "INSERT INTO batch_events (batch_code, event_time, event_type, details) VALUES (?, ?, ?, ?)",
            (batch_code, datetime.now().astimezone().isoformat(timespec="seconds"), event_type, details),
        )


def get_batches():
    with connect_db() as con:
        return con.execute("SELECT * FROM batches ORDER BY id DESC").fetchall()


def get_batch(batch_code):
    with connect_db() as con:
        return con.execute("SELECT * FROM batches WHERE batch_code = ?", (batch_code,)).fetchone()


def get_events(batch_code):
    with connect_db() as con:
        return con.execute(
            "SELECT * FROM batch_events WHERE batch_code = ? ORDER BY id DESC", (batch_code,)
        ).fetchall()


def delete_batch(batch_code):
    """Delete one batch and its related event history in a single transaction."""
    with connect_db() as con:
        con.execute("DELETE FROM batch_events WHERE batch_code = ?", (batch_code,))
        con.execute("DELETE FROM batches WHERE batch_code = ?", (batch_code,))


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    import tensorflow as tf
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


def make_qr_png(payload):
    import qrcode
    qr = qrcode.QRCode(box_size=8, border=3)
    qr.add_data(payload)
    qr.make(fit=True)
    image = qr.make_image(fill_color="black", back_color="white")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()

# -------------------- HEADER --------------------
st.title("🌱 AgroVision")
st.caption("Solar-powered smart mini cold storage • Student prototype")
# Read the current DevKit status for the summary cards. The detailed panel refreshes independently.
summary_esp32, summary_esp32_error = esp32_get_status(ESP32_DEFAULT_URL)
if summary_esp32_error:
    summary_esp32, cloud_error = thingspeak_get_latest()
    if summary_esp32:
        st.success("Cloud telemetry connected · readings received from ThingSpeak")
    else:
        st.info("Tomato AI and batch records are available. Local ESP32 is not reachable and cloud telemetry is not available yet.")
else:
    st.success(f"ESP32 connected · {summary_esp32.get('ip', ESP32_DEFAULT_URL)} · live telemetry available")

# -------------------- DASHBOARD SUMMARY --------------------
batches = get_batches()
col1, col2, col3, col4 = st.columns(4)
col1.metric("Saved batches", len(batches))
col2.metric("AI model file", "Found" if MODEL_PATH.exists() else "Not found")
if summary_esp32 and isinstance(summary_esp32.get("temperature_c"), (int, float)):
    col3.metric("ESP32 temperature", f"{summary_esp32['temperature_c']:.1f} °C")
else:
    col3.metric("ESP32 temperature", "Not connected" if summary_esp32_error else "Sensor fault")
if summary_esp32_error:
    col4.metric("Hardware alerts", "Not connected")
elif summary_esp32.get("sensor_fault", False):
    col4.metric("Hardware alerts", "Sensor fault")
elif summary_esp32.get("temperature_alert", False):
    col4.metric("Hardware alerts", "Temperature alert")
else:
    col4.metric("Hardware alerts", "Normal")

# -------------------- TABS --------------------
tab_overview, tab_ai, tab_batch, tab_history, tab_hardware, tab_live = st.tabs([
    "🏠 Overview", "🍅 Tomato AI", "📦 Batch Passport", "🗂️ Batch History", "🌡️ ESP32 Controls", "📷 ESP32 Live AI"
])

with tab_overview:
    st.subheader("Project modules")
    a, b = st.columns(2)
    with a:
        st.markdown("### 🍅 Tomato Maturity AI")
        st.write("Upload a tomato photo. Preview and prediction run automatically.")
        st.markdown("### 📦 QR Batch Passport")
        st.write("Create batch records, generate QR codes, and keep records in a local SQLite database.")
    with b:
        st.markdown("### 🌡️ Temperature & Crop Mode")
        st.write("Live crop profiles, temperature readings and hardware alerts are connected to the ESP32 DevKit.")
        st.markdown("### 🚨 Fault Monitoring")
        st.write("DHT11 sensor faults and temperature deviations are reported by the connected ESP32.")

with tab_ai:
    st.subheader("🍅 Automatic Tomato Maturity Estimator")
    st.write("Choose a JPG or PNG photo; the model analyzes it automatically.")
    uploaded_file = st.file_uploader("Choose tomato photo", type=["jpg", "jpeg", "png"], key="ai_upload")
    if uploaded_file is not None:
        try:
            raw = uploaded_file.getvalue()
            image = ImageOps.exif_transpose(Image.open(BytesIO(raw))).convert("RGB")
            preview_buffer = BytesIO()
            image.save(preview_buffer, format="PNG")
            preview_b64 = base64.b64encode(preview_buffer.getvalue()).decode("ascii")
            preview_html = f'''<!DOCTYPE html><html><head><meta name="viewport" content="width=device-width, initial-scale=1"><style>body{{margin:0;padding:8px;background:transparent;text-align:center}}img{{display:block;max-width:100%;max-height:420px;width:auto;height:auto;object-fit:contain;margin:auto;border-radius:10px}}</style></head><body><img src="data:image/png;base64,{preview_b64}" alt="Uploaded tomato"></body></html>'''
            components.html(preview_html, height=440, scrolling=False)
            st.caption(f"Image loaded: {image.width} × {image.height} pixels")
            if not MODEL_PATH.exists():
                st.error(f"Model not found. Put {MODEL_PATH.name} beside this Python file.")
            else:
                with st.spinner("Analyzing tomato automatically..."):
                    model = load_model()
                    arr = np.asarray(image.resize((224, 224)), dtype=np.float32)
                    arr = np.expand_dims(arr, axis=0)
                    # The trained model contains its own Rescaling layer.
                    predictions = model.predict(arr, verbose=0)[0]
                if len(predictions) != len(CLASS_NAMES):
                    st.error("Model output does not match the three configured class labels.")
                else:
                    idx = int(np.argmax(predictions))
                    label = CLASS_NAMES[idx]
                    score = float(predictions[idx]) * 100
                    left, right = st.columns([2, 1])
                    with left:
                        st.success(f"Estimated maturity: {label}")
                    with right:
                        st.metric("Model score", f"{score:.1f}%")
                    st.write("Class scores")
                    for name, value in zip(CLASS_NAMES, predictions):
                        st.write(f"**{name}: {float(value)*100:.1f}%")
                        st.progress(float(np.clip(value, 0, 1)))
                    st.warning("Experimental estimate only. The initial test set was small; confirm results using more independent images.")
                    latest = {"label": label, "score": score, "filename": uploaded_file.name}
                    st.session_state["latest_ai"] = latest

                    # Auto-save once for each unique image and selected batch.
                    selected_batch_for_ai = st.session_state.get("selected_batch")
                    image_key = hashlib.sha256(raw).hexdigest()
                    save_key = f"{selected_batch_for_ai}:{image_key}"
                    if selected_batch_for_ai and st.session_state.get("last_saved_ai_key") != save_key:
                        batch_row = get_batch(selected_batch_for_ai)
                        if batch_row:
                            with connect_db() as con:
                                con.execute(
                                    "UPDATE batches SET latest_maturity = ?, latest_ai_score = ? WHERE batch_code = ?",
                                    (label, score, selected_batch_for_ai),
                                )
                            add_event(
                                selected_batch_for_ai,
                                "AI maturity result",
                                f"{label} ({score:.1f}%), file={uploaded_file.name}",
                            )
                            st.session_state["last_saved_ai_key"] = save_key
                            st.success(f"Automatically saved {label} ({score:.1f}%) to batch {selected_batch_for_ai}.")
                    elif not selected_batch_for_ai:
                        st.info("Prediction ready. Create/select a batch in Batch Passport; its next analysis will save automatically.")
        except Exception as exc:
            st.error(f"Could not display or analyze this image: {exc}")

with tab_batch:
    st.subheader("📦 Create QR Batch Passport")
    st.write("Batch records are saved locally in agrovision_batches.db and remain after app restarts on this computer.")
    with st.form("create_batch_form", clear_on_submit=True):
        crop = st.selectbox("Crop", list(CROP_PROFILES.keys()))
        c1, c2 = st.columns(2)
        with c1:
            quantity = st.number_input("Quantity", min_value=0.1, value=10.0, step=0.5)
        with c2:
            unit = st.selectbox("Unit", ["kg", "crates", "boxes", "pieces"])
        notes = st.text_area("Notes (optional)", placeholder="Farm, harvest date, visual notes...")
        submitted = st.form_submit_button("Create Batch & QR", type="primary")
    if submitted:
        now = datetime.now().astimezone()
        batch_code = "AV-" + now.strftime("%Y%m%d-%H%M%S")
        try:
            with connect_db() as con:
                con.execute(
                    "INSERT INTO batches (batch_code, crop, quantity, unit, notes, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (batch_code, crop, float(quantity), unit, notes.strip(), now.isoformat(timespec="seconds")),
                )
            add_event(batch_code, "Batch created", f"Crop={crop}; quantity={quantity} {unit}; notes={notes.strip() or 'None'}")
            st.session_state["selected_batch"] = batch_code
            st.success(f"Batch created: {batch_code}")
        except sqlite3.IntegrityError:
            st.error("Batch ID collision. Please create the batch again.")

    batches = get_batches()
    if batches:
        codes = [row["batch_code"] for row in batches]
        default_code = st.session_state.get("selected_batch", codes[0])
        if default_code not in codes:
            default_code = codes[0]
        selected_code = st.selectbox("Select batch to view / update", codes, index=codes.index(default_code), key="batch_selector")
        st.session_state["selected_batch"] = selected_code
        row = get_batch(selected_code)
        st.markdown(f"### Batch `{row['batch_code']}`")
        d1, d2, d3 = st.columns(3)
        d1.metric("Crop", row["crop"])
        d2.metric("Quantity", f"{row['quantity']:g} {row['unit']}")
        d3.metric("Latest AI result", row["latest_maturity"] or "Not recorded")
        st.write("**Created:**", row["created_at"])
        st.write("**Notes:**", row["notes"] or "—")

        try:
            payload = json.dumps({
                "project": "AgroVision",
                "batch_id": row["batch_code"],
                "crop": row["crop"],
                "quantity": row["quantity"],
                "unit": row["unit"],
                "created_at": row["created_at"],
                "notes": row["notes"] or "",
                "latest_maturity": row["latest_maturity"] or "Not recorded",
            }, ensure_ascii=False)
            qr_bytes = make_qr_png(payload)
            qcol1, qcol2 = st.columns([1, 2])
            with qcol1:
                st.image(qr_bytes, caption="Batch QR Code", width=220)
                st.download_button("Download QR PNG", data=qr_bytes, file_name=f"{row['batch_code']}_QR.png", mime="image/png")
            with qcol2:
                st.caption("This QR currently contains the batch details as data. Scanning it shows the encoded details; a public web page for remote lookup requires hosting/backend later.")
                st.caption("AI results are saved automatically to the selected batch when a new image is analyzed.")
        except ImportError:
            st.error("QR library is missing. In your activated environment run: python -m pip install qrcode[pil]")
        except Exception as exc:
            st.error(f"Could not create QR code: {exc}")
    else:
        st.info("No batches yet. Fill the form above and click Create Batch & QR.")

with tab_history:
    st.subheader("🗂️ Batch History")
    batches = get_batches()
    if not batches:
        st.info("No saved batches. Create your first batch in the Batch Passport tab.")
    else:
        import pandas as pd

        # Convert only explicit kg/g entries to kilograms. Crates/boxes are
        # not converted because their weights are unknown.
        total_kg = 0.0
        non_weight_units = 0
        records = []
        for row in batches:
            unit = str(row["unit"]).strip().lower()
            quantity = float(row["quantity"])
            if unit == "kg":
                total_kg += quantity
            elif unit == "g":
                total_kg += quantity / 1000.0
            else:
                non_weight_units += 1

            records.append({
                "Batch ID": row["batch_code"],
                "Crop": row["crop"],
                "Weight / Quantity": f"{quantity:g} {row['unit']}",
                "AI Result": row["latest_maturity"] or "Not analyzed",
                "Confidence": f"{row['latest_ai_score']:.1f}%" if row["latest_ai_score"] is not None else "—",
                "Created": row["created_at"],
            })

        st.dataframe(pd.DataFrame(records), use_container_width=True, hide_index=True)

        total_col, count_col = st.columns(2)
        total_col.metric("Total stored weight (kg)", f"{total_kg:,.3f} kg")
        count_col.metric("Total batches", len(batches))
        if non_weight_units:
            st.caption(
                f"Note: {non_weight_units} batch(es) use crates/boxes or other units, "
                "so they are not included in the kg total because their individual weights are unknown."
            )

        st.markdown("### Manage batches")
        st.caption("Select a batch to delete it. A confirmation is required; its event/AI history will be deleted too.")
        batch_codes = [row["batch_code"] for row in batches]
        selected_delete = st.selectbox("Choose batch", batch_codes, key="delete_batch_selector")
        selected_row = get_batch(selected_delete)
        if selected_row:
            st.write(
                f"**Selected:** {selected_row['batch_code']} · {selected_row['crop']} · "
                f"{selected_row['quantity']:g} {selected_row['unit']}"
            )
        if st.button("🗑️ Delete selected batch", key="request_delete_batch"):
            st.session_state["pending_delete_batch"] = selected_delete

        if st.session_state.get("pending_delete_batch") in batch_codes:
            pending = st.session_state["pending_delete_batch"]
            st.warning(f"Permanently delete batch {pending} and all its associated event history?")
            confirm_col, cancel_col = st.columns(2)
            with confirm_col:
                if st.button("Yes, delete permanently", key="confirm_delete_selected", type="primary"):
                    delete_batch(pending)
                    if st.session_state.get("selected_batch") == pending:
                        st.session_state.pop("selected_batch", None)
                    st.session_state.pop("pending_delete_batch", None)
                    st.success(f"Batch {pending} and its event history were deleted.")
                    st.rerun()
            with cancel_col:
                if st.button("Cancel", key="cancel_delete_selected"):
                    st.session_state.pop("pending_delete_batch", None)
                    st.rerun()

        history_code = st.selectbox(
            "View detailed event history", batch_codes, key="history_selector"
        )
        events = get_events(history_code)
        if events:
            st.markdown(f"#### Events for {history_code}")
            for event in events:
                with st.expander(f"{event['event_time']} · {event['event_type']}"):
                    st.write(event["details"])
        else:
            st.caption("No events recorded for this batch yet.")

        st.markdown("### 📷 ESP32 camera captures for selected batch")
        with connect_db() as con:
            capture_rows = con.execute(
                """SELECT captured_at, image_path, maturity, ai_score, class_scores
                   FROM esp32_captures WHERE batch_code = ?
                   ORDER BY id DESC LIMIT 50""",
                (history_code,),
            ).fetchall()
        if capture_rows:
            capture_records = [
                {
                    "Captured at": row["captured_at"],
                    "Image": Path(row["image_path"]).name,
                    "AI result": row["maturity"],
                    "Model score": f"{float(row['ai_score']):.1f}%",
                }
                for row in capture_rows
            ]
            st.dataframe(pd.DataFrame(capture_records), use_container_width=True, hide_index=True)
            latest_capture = capture_rows[0]
            latest_path = Path(latest_capture["image_path"])
            if latest_path.exists():
                st.image(str(latest_path), caption=f"Latest capture · {latest_capture['maturity']} ({float(latest_capture['ai_score']):.1f}%)", width=420)
        else:
            st.caption("No automatic ESP32 captures have been saved for this batch yet.")

        csv_text = pd.DataFrame(records).to_csv(index=False)
        st.download_button(
            "Export batch history (CSV)",
            data=csv_text,
            file_name="agrovision_batch_history.csv",
            mime="text/csv",
        )

with tab_hardware:
    st.subheader("🌡️ Temperature, Crop Mode & Fault Monitoring")
    st.write("Local controls require the same Wi-Fi as the ESP32. ThingSpeak cloud readings can be viewed remotely.")
    esp32_url = st.text_input(
        "ESP32 DevKit URL",
        value=st.session_state.get("esp32_url", ESP32_DEFAULT_URL),
        key="esp32_url",
        help="Use the ESP32 IP shown in Serial Monitor, without /api/status.",
    ).strip().rstrip("/")

    h1, h2 = st.columns(2)
    with h1:
        st.markdown("### Smart Crop Mode")
        selected_crop = st.selectbox("Crop profile", list(CROP_PROFILES.keys()), key="hardware_crop")
        target_min, target_max = CROP_PROFILES[selected_crop]
        st.metric("Configured target range", f"{target_min:.1f}–{target_max:.1f} °C")
        st.caption("The ESP32 firmware uses fixed crop ranges. This dashboard sends the crop name; it does not set an arbitrary temperature target.")

        command_key = f"{esp32_url}:{selected_crop}"
        if st.session_state.get("last_auto_command_key") != command_key:
            result, error = esp32_set_crop(esp32_url, selected_crop)
            if error:
                st.warning("Crop selection has not reached the ESP32 yet. Check the IP and local Wi-Fi.")
                st.caption(f"Connection detail: {error}")
            else:
                st.session_state["last_auto_command_key"] = command_key
                st.success(f"Crop mode sent: {result.get('crop', selected_crop)}")

    with h2:
        st.markdown("### Live Sensor Status")
        if st.button("Refresh ESP32 status", key="refresh_esp32_status"):
            st.session_state["esp32_manual_refresh"] = datetime.now().isoformat()

        if hasattr(st, "fragment"):
            @st.fragment(run_every=5)
            def render_esp32_status():
                status, status_error = esp32_get_status(esp32_url)
                if status_error:
                    status, cloud_error = thingspeak_get_latest()
                    if not status:
                        st.error("Neither local ESP32 nor ThingSpeak cloud telemetry is available.")
                        st.caption(f"Local connection: {status_error}")
                        st.caption(f"Cloud connection: {cloud_error}")
                        st.metric("Current temperature", "—")
                        st.metric("Humidity", "—")
                        st.metric("Sensor / alert status", "Not connected")
                        return
                    st.success("Cloud telemetry connected · ThingSpeak")
                    if status.get("created_at"):
                        st.caption(f"Last cloud reading: {status['created_at']} (UTC)")
                else:
                    st.success(f"ESP32 connected · {status.get('ip', 'local network')}")
                temp_value = status.get("temperature_c")
                humidity_value = status.get("humidity_pct")
                m1, m2 = st.columns(2)
                m1.metric("Current temperature", f"{float(temp_value):.1f} °C" if isinstance(temp_value, (int, float)) else "—")
                m2.metric("Humidity", f"{float(humidity_value):.1f}%" if isinstance(humidity_value, (int, float)) else "—")
                st.write(f"**Selected crop:** {status.get('crop', '—')}")
                low = status.get("target_min_c")
                high = status.get("target_max_c")
                if isinstance(low, (int, float)) and isinstance(high, (int, float)):
                    st.write(f"**Configured range:** {low:.1f}–{high:.1f} °C")
                if status.get("sensor_fault", False):
                    st.error("SENSOR FAULT: Check DHT11 wiring / sensor. Buzzer should alert.")
                elif status.get("temperature_alert", False):
                    st.warning("TEMPERATURE ALERT: Reading is outside the configured range plus/minus the firmware margin.")
                else:
                    st.success("Sensor status OK · Temperature within alert limits.")
                st.write(f"**Buzzer status:** {'ON' if status.get('buzzer_on', False) else 'OFF'}")
                st.caption("Values refresh every 5 seconds while this tab is open.")
            render_esp32_status()
        else:
            status, status_error = esp32_get_status(esp32_url)
            if status_error:
                st.error(f"ESP32 not connected: {status_error}")
            else:
                st.json(status)

    st.info("ThingSpeak is used for remote sensor readings only. Crop-mode commands still require direct local Wi-Fi access to the ESP32. This prototype does not control a compressor or relay.")

# -------------------- ESP32 LIVE CAMERA + AI --------------------
with tab_live:
    st.subheader("📷 ESP32 Live Camera → Tomato AI → Batch")
    st.write(
        "When automatic capture is enabled, the dashboard fetches a fresh frame from the ESP32, "
        "runs the tomato model, and saves the image + result against the selected batch."
    )
    live_camera_url = st.text_input(
        "ESP32 capture URL",
        value=st.session_state.get("live_camera_url", "http://10.239.231.189/capture"),
        key="live_camera_url",
        help="Use the CURRENT camera IP shown by the ESP32-S3-CAM Serial Monitor, ending in /capture. This is not the normal DevKit IP.",
    )
    live_batches = get_batches()
    if not live_batches:
        st.warning("Create a batch first in the Batch Passport tab, then return here and select it.")
        st.session_state["live_batch_code"] = None
    else:
        live_codes = [row["batch_code"] for row in live_batches]
        default_live = st.session_state.get("live_batch_code") or st.session_state.get("selected_batch")
        if default_live not in live_codes:
            default_live = live_codes[0]
        st.selectbox(
            "Save camera images and predictions to this batch",
            live_codes,
            index=live_codes.index(default_live),
            key="live_batch_code",
        )

    live_interval = st.selectbox(
        "Capture interval",
        options=[5, 10, 15, 30, 60],
        index=1,
        format_func=lambda value: f"Every {value} seconds",
        key="live_capture_interval",
    )
    st.checkbox(
        "Enable automatic capture and AI prediction",
        key="live_capture_enabled",
        help="Keep this dashboard tab open. Stop the capture by unchecking this box.",
    )

    if not hasattr(st, "fragment"):
        st.error(
            "Your installed Streamlit version does not support automatic fragment refresh. "
            "Please update Streamlit in the same environment used to run this dashboard."
        )
    else:
        @st.fragment(run_every=live_interval if st.session_state.get("live_capture_enabled", False) else None)
        def render_live_camera():
            if not st.session_state.get("live_capture_enabled", False):
                st.info("Automatic capture is OFF. Enable the checkbox above to start.")
                return

            batch_code = st.session_state.get("live_batch_code")
            if not batch_code:
                st.warning("Create/select a batch before starting camera capture.")
                return

            try:
                response = requests.get(
                    st.session_state.get("live_camera_url", "http://10.239.231.189/capture"),
                    timeout=10,
                )
                response.raise_for_status()
                image = Image.open(BytesIO(response.content)).convert("RGB")
                image.load()

                image_dir = BASE_DIR / "esp32_images"
                image_dir.mkdir(parents=True, exist_ok=True)
                captured_dt = datetime.now().astimezone()
                timestamp = captured_dt.strftime("%Y%m%d_%H%M%S_%f")
                image_path = image_dir / f"tomato_live_{timestamp}.jpg"
                image.save(image_path, format="JPEG", quality=92)

                model = load_model()
                if model is None:
                    st.error(f"Model file not found: {MODEL_PATH}")
                    return
                array = np.asarray(image.resize((224, 224)), dtype=np.float32)
                array = np.expand_dims(array, axis=0)
                predictions = model.predict(array, verbose=0)[0]
                if len(predictions) != len(CLASS_NAMES):
                    st.error("Model output does not match Green / Turning / Red_Ripe labels.")
                    return

                index = int(np.argmax(predictions))
                label = CLASS_NAMES[index]
                score = float(predictions[index]) * 100
                scores_json = json.dumps(
                    {name: float(value) * 100 for name, value in zip(CLASS_NAMES, predictions)}
                )
                captured_text = captured_dt.isoformat(timespec="seconds")

                with connect_db() as con:
                    con.execute(
                        """INSERT INTO esp32_captures
                           (batch_code, captured_at, image_path, maturity, ai_score, class_scores)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (batch_code, captured_text, str(image_path), label, score, scores_json),
                    )
                    con.execute(
                        "UPDATE batches SET latest_maturity = ?, latest_ai_score = ? WHERE batch_code = ?",
                        (label, score, batch_code),
                    )
                add_event(
                    batch_code,
                    "ESP32 camera AI capture",
                    f"{image_path.name} · {label} ({score:.1f}%) · {captured_text}",
                )

                left, right = st.columns([3, 2])
                with left:
                    st.image(image, caption=f"Latest ESP32 capture · {captured_text}", use_container_width=True)
                with right:
                    st.success(f"Batch: {batch_code}")
                    st.metric("Predicted maturity", label)
                    st.metric("Model score", f"{score:.1f}%")
                    st.write("Class scores")
                    for name, value in zip(CLASS_NAMES, predictions):
                        st.write(f"**{name}: {float(value) * 100:.1f}%**")
                        st.progress(float(np.clip(value, 0, 1)))
                st.caption(f"Saved image: {image_path.name}")
                st.warning(
                    "Experimental AI estimate, not a guaranteed accuracy percentage. "
                    "Verify predictions against actual tomatoes."
                )
            except Exception as exc:
                st.error(f"ESP32 capture or AI prediction failed: {exc}")
                st.caption("Check that the ESP32 is powered on, reachable on the same Wi-Fi, and the capture URL is correct.")

        render_live_camera()

st.divider()
st.caption("AgroVision student prototype • Local batch database • AI estimates require validation")
