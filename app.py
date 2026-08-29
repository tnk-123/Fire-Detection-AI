import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import time
from datetime import datetime
from PIL import Image
from ultralytics import YOLO

# 1. ตั้งค่าหน้าเพจ
st.set_page_config(page_title="AI Fire Detect", layout="wide")

# รายการตัวเลือกเสียงเตือนภัย (Web Audio Synthesizer Presets)
# noinspection PyPackageRequirements
SOUND_OPTIONS = {
    "🚨 Siren Emergency (ไซเรนฉุกเฉิน)": "siren",
    "⏰ Rapid Beep (เสียงบี๊บถี่)": "beep",
    "☢️ Low Horn (หวอเตือนภัยระดับสูง)": "nuclear",
    "🔔 High Pulse (สัญญาณชีพจรความถี่สูง)": "high_pulse"
}

# 2. ปรับแต่ง CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700;800;900&family=Prompt:wght@300;400;600;700;800;900&display=swap');

header, footer {visibility: hidden;}

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', 'Prompt', sans-serif;
}

.stApp {
    background-color: #0b0c10;
    color: #e2e8f0;
}

.brand-logo {
    display: flex;
    align-items: center;
    gap: 12px;
}
.brand-title {
    font-size: 24px;
    font-weight: 800;
    color: #ffffff;
    line-height: 1.1;
}
.brand-title span {
    color: #ef4444;
}
.brand-subtitle {
    font-size: 12px;
    color: #64748b;
}

.clock-badge {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    padding: 6px 16px;
    border-radius: 20px;
    font-size: 14px;
    color: #94a3b8;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    width: fit-content;
}

.hero-title {
    font-size: 42px;
    font-weight: 800;
    line-height: 1.1;
    color: #ffffff;
    margin-bottom: 10px;
}
.hero-title-red {
    color: #ef4444;
}
.hero-desc {
    color: #94a3b8;
    font-size: 15px;
    margin-bottom: 25px;
}

div[data-testid="stRadio"] div[role="radiogroup"] {
    gap: 12px;
}
div[data-testid="stRadio"] div[role="radiogroup"] > label {
    background: #161821;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 16px 20px;
    color: #ffffff;
    transition: all 0.3s ease;
    cursor: pointer;
}
div[data-testid="stRadio"] div[role="radiogroup"] > label:hover {
    border-color: rgba(239, 68, 68, 0.5);
    background: #1c1f2c;
}
div[data-testid="stRadio"] div[role="radiogroup"] > label[data-checked="true"] {
    background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%);
    border: none;
    box-shadow: 0 8px 20px rgba(220, 38, 38, 0.3);
}

.how-title {
    font-size: 14px;
    font-weight: 700;
    color: #ef4444;
    margin-top: 25px;
    margin-bottom: 10px;
}
.step-item {
    display: flex;
    align-items: flex-start;
    gap: 15px;
    margin-bottom: 18px;
}
.step-icon {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid rgba(239, 68, 68, 0.2);
    display: flex;
    align-items: center;
    justify-content: center;
    color: #ef4444;
    font-size: 16px;
    flex-shrink: 0;
}
.step-content {
    display: flex;
    flex-direction: column;
}
.step-header {
    font-size: 14px;
    font-weight: 700;
    color: #f1f5f9;
}
.step-text {
    font-size: 12px;
    color: #64748b;
}

.panel-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
}
.live-badge {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 14px;
    font-weight: 600;
    color: #f8fafc;
}
.live-dot {
    width: 8px;
    height: 8px;
    background-color: #ef4444;
    border-radius: 50%;
    box-shadow: 0 0 10px #ef4444;
}
.system-active {
    background: rgba(34, 197, 94, 0.1);
    border: 1px solid rgba(34, 197, 94, 0.2);
    color: #4ade80;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    display: flex;
    align-items: center;
    gap: 6px;
}
.system-dot {
    width: 6px;
    height: 6px;
    background-color: #22c55e;
    border-radius: 50%;
}

[data-testid="stImage"] img {
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    background-color: #000;
}

.metric-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 12px;
    padding: 8px 10px;
    text-align: center;
}
.metric-title {
    font-size: 11px;
    font-weight: 700;
    color: #94a3b8;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 4px;
    margin-bottom: 2px;
}
.metric-value {
    font-size: 20px;
    font-weight: 800;
    color: #ffffff;
    line-height: 1.1;
    margin-bottom: 2px;
}
.metric-sub {
    font-size: 10px;
    color: #64748b;
}

.display-card {
    background: rgba(0, 0, 0, 0.3);
    border: 2px dashed rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    min-height: 380px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    color: #64748b;
    font-size: 18px;
    gap: 12px;
}

.tip-box {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    padding: 10px 16px;
    font-size: 12px;
    color: #64748b;
    margin-top: 15px;
    display: flex;
    align-items: center;
    gap: 8px;
}
</style>
""", unsafe_allow_html=True)


# 3. โหลดโมเดล YOLO
@st.cache_resource
def load_model():
    return YOLO("model/best.pt")


model = load_model()


# ฟังก์ชันเล่นเสียงทดสอบ
def play_test_audio(sound_type="siren"):
    html_code = f"""
    <script>
    (function() {{
        var AudioContext = window.AudioContext || window.webkitAudioContext;
        var ctx = new AudioContext();
        if (ctx.state === 'suspended') {{ ctx.resume(); }}
        var osc = ctx.createOscillator();
        var gain = ctx.createGain();
        osc.type = 'sawtooth';
        osc.frequency.setValueAtTime(750, ctx.currentTime);
        osc.connect(gain);
        gain.connect(ctx.destination);
        gain.gain.setValueAtTime(0.2, ctx.currentTime);
        osc.start();
        setTimeout(function() {{ osc.stop(); }}, 1000);
    }})();
    </script>
    """
    components.html(html_code, height=0, width=0)


# ฟังก์ชันสร้าง Alert Banner พร้อมปุ่มปิดเสียงแบบ Dynamic (JavaScript Interactive Component)
def render_alert_with_audio(alert_title, alert_sub, sound_type):
    html_code = f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Prompt:wght@600;800;900&display=swap');
        body {{
            margin: 0;
            padding: 0;
            font-family: 'Prompt', sans-serif;
            background: transparent;
        }}
        .alert-banner-danger {{
            background: linear-gradient(135deg, #ff0000 0%, #b91c1c 50%, #7f1d1d 100%);
            border: 2px solid #ffe600;
            border-radius: 14px;
            padding: 10px 18px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 0 15px rgba(255, 0, 0, 0.8);
            animation: super-flash 0.6s infinite alternate;
        }}
        @keyframes super-flash {{
            0% {{ box-shadow: 0 0 12px rgba(255, 0, 0, 0.8); border-color: #ff0000; transform: scale(1); }}
            100% {{ box-shadow: 0 0 25px rgba(255, 0, 0, 1), 0 0 35px rgba(255, 230, 0, 0.9); border-color: #ffe600; transform: scale(1.002); }}
        }}
        .alert-left {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        .alert-icon-big {{ font-size: 26px; }}
        .alert-title-big {{
            font-size: 16px;
            font-weight: 900;
            color: #ffffff;
            letter-spacing: 0.5px;
        }}
        .alert-sub-big {{
            font-size: 11px;
            color: #fff1f1;
            font-weight: 600;
        }}
        .stop-btn {{
            background: #161821;
            color: #ffffff;
            border: 2px solid #ef4444;
            border-radius: 10px;
            padding: 8px 16px;
            font-family: 'Prompt', sans-serif;
            font-size: 13px;
            font-weight: 800;
            cursor: pointer;
            transition: all 0.2s ease;
            white-space: nowrap;
            box-shadow: 0 4px 10px rgba(0,0,0,0.3);
        }}
        .stop-btn:hover {{
            background: #ef4444;
            color: #ffffff;
            border-color: #ffffff;
        }}
    </style>

    <div class="alert-banner-danger">
        <div class="alert-left">
            <div class="alert-icon-big">🚨</div>
            <div>
                <div class="alert-title-big">{alert_title}</div>
                <div class="alert-sub-big">{alert_sub}</div>
            </div>
        </div>
        <button class="stop-btn" onclick="stopAlarmSound(this)">🛑 🔇 STOP ALARM / ปิดเสียง</button>
    </div>

    <script>
    var AudioContext = window.AudioContext || window.webkitAudioContext;
    var ctx = new AudioContext();

    function startSound() {{
        if (ctx.state === 'suspended') {{ ctx.resume(); }}
        var soundType = "{sound_type}";
        if (soundType === "siren") {{
            var osc = ctx.createOscillator();
            var gain = ctx.createGain();
            var lfo = ctx.createOscillator();
            var lfoGain = ctx.createGain();
            osc.type = 'sawtooth';
            lfo.type = 'sine';
            lfo.frequency.setValueAtTime(2.5, ctx.currentTime);
            lfoGain.gain.setValueAtTime(350, ctx.currentTime);
            osc.frequency.setValueAtTime(750, ctx.currentTime);
            lfo.connect(osc.frequency);
            osc.connect(gain);
            gain.connect(ctx.destination);
            gain.gain.setValueAtTime(0.25, ctx.currentTime);
            osc.start(); lfo.start();
        }} else if (soundType === "beep") {{
            var osc = ctx.createOscillator();
            var gain = ctx.createGain();
            osc.type = 'sine';
            osc.frequency.setValueAtTime(950, ctx.currentTime);
            var now = ctx.currentTime;
            for (var i = 0; i < 200; i++) {{
                gain.gain.setValueAtTime(0.3, now + i*0.35);
                gain.gain.setValueAtTime(0.0, now + i*0.35 + 0.18);
            }}
            osc.connect(gain);
            gain.connect(ctx.destination);
            osc.start();
        }} else if (soundType === "nuclear") {{
            var osc1 = ctx.createOscillator();
            var osc2 = ctx.createOscillator();
            var gain = ctx.createGain();
            osc1.type = 'sawtooth';
            osc2.type = 'square';
            osc1.frequency.setValueAtTime(220, ctx.currentTime);
            osc2.frequency.setValueAtTime(224, ctx.currentTime);
            osc1.connect(gain);
            osc2.connect(gain);
            gain.connect(ctx.destination);
            gain.gain.setValueAtTime(0.25, ctx.currentTime);
            osc1.start(); osc2.start();
        }} else if (soundType === "high_pulse") {{
            var osc = ctx.createOscillator();
            var gain = ctx.createGain();
            var lfo = ctx.createOscillator();
            var lfoGain = ctx.createGain();
            osc.type = 'square';
            lfo.type = 'square';
            lfo.frequency.setValueAtTime(7, ctx.currentTime);
            lfoGain.gain.setValueAtTime(250, ctx.currentTime);
            osc.frequency.setValueAtTime(1300, ctx.currentTime);
            lfo.connect(osc.frequency);
            osc.connect(gain);
            gain.connect(ctx.destination);
            gain.gain.setValueAtTime(0.18, ctx.currentTime);
            osc.start(); lfo.start();
        }}
    }}

    function stopAlarmSound(btn) {{
        if (ctx) {{
            ctx.suspend();
        }}
        btn.innerText = "🔇 Muted / ปิดเสียงแล้ว";
        btn.style.background = "#334155";
        btn.style.borderColor = "#64748b";
        btn.style.color = "#94a3b8";
        btn.disabled = true;
    }}

    startSound();
    </script>
    """
    return components.html(html_code, height=85)


# Header ด้านบน
col_nav1, col_nav2 = st.columns([3, 1])
with col_nav1:
    st.markdown("""
<div class="brand-logo" style="margin-top: 5px;">
    <span style="font-size:28px;">🔥</span>
    <div>
        <div class="brand-title"><span>AI</span> Fire Detect</div>
        <div class="brand-subtitle">Smart Fire Detection System</div>
    </div>
</div>
""", unsafe_allow_html=True)

with col_nav2:
    clock_placeholder = st.empty()
    clock_placeholder.markdown(
        f'<div style="display: flex; justify-content: flex-end;"><div class="clock-badge">🕒 {datetime.now().strftime("%H:%M")}</div></div>',
        unsafe_allow_html=True,
    )

col_left, col_right = st.columns([1, 2.2])

with col_left:
    st.markdown("""
<div class="hero-title">AI-Powered<br><span class="hero-title-red">Fire Detection</span></div>
<div class="hero-desc">Detect fire and smoke in real-time to keep your environment safe.</div>
""", unsafe_allow_html=True)

    mode = st.radio(
        label="",
        options=["📷  Start Webcam\n\nReal-time detection", "📁  Upload Photo\n\nDetect from image"],
        label_visibility="collapsed",
        key="mode_radio_main"
    )

    st.markdown('<div class="how-title">🔊 Select Alert Sound</div>', unsafe_allow_html=True)
    selected_sound_name = st.selectbox(
        "Select Sound",
        options=list(SOUND_OPTIONS.keys()),
        label_visibility="collapsed",
        key="sound_selectbox"
    )
    selected_sound_code = SOUND_OPTIONS[selected_sound_name]

    if st.button("🔊 Test Sound / ทดสอบเสียง", use_container_width=True, key="btn_test_sound"):
        play_test_audio(selected_sound_code)
        st.toast(f"✅ ทดสอบเสียง: {selected_sound_name}", icon="🔔")

    st.markdown("""
<div class="how-title">How it works</div>
<div class="step-item">
    <div class="step-icon">📹</div>
    <div class="step-content">
        <div class="step-header">1. Choose a mode</div>
        <div class="step-text">Select webcam or upload an image</div>
    </div>
</div>
<div class="step-item">
    <div class="step-icon">🧠</div>
    <div class="step-content">
        <div class="step-header">2. AI Analysis</div>
        <div class="step-text">Our AI analyzes fire (>90% confidence)</div>
    </div>
</div>
<div class="step-item">
    <div class="step-icon">🔔</div>
    <div class="step-content">
        <div class="step-header">3. Get Alert</div>
        <div class="step-text">Receive instant alert if fire lasts > 15 sec</div>
    </div>
</div>
""", unsafe_allow_html=True)

with col_right:
    st.markdown("""
<div class="panel-header">
    <div class="live-badge">
        <div class="live-dot"></div> Live Detection
    </div>
    <div class="system-active">
        <div class="system-dot"></div> System Active
    </div>
</div>
""", unsafe_allow_html=True)

    alert_placeholder = st.empty()
    metrics_placeholder = st.empty()

    is_upload_mode = "Upload" in mode

    if is_upload_mode:
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], key="file_uploader_main")

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            results = model(image)
            res_plotted = results[0].plot()

            fire_conf = 0.0
            smoke_conf = 0.0

            for box in results[0].boxes:
                conf = box.conf[0].item()
                cls_id = int(box.cls[0].item())
                cls_name = model.names[cls_id].lower() if hasattr(model, 'names') else ''

                if 'smoke' in cls_name:
                    if conf > smoke_conf: smoke_conf = conf
                else:
                    if conf > fire_conf: fire_conf = conf

            if fire_conf > 0.90:
                with alert_placeholder:
                    render_alert_with_audio(
                        "⚠️ FIRE DETECTED!",
                        "Fire detected with high confidence (>90%).",
                        selected_sound_code
                    )
            else:
                alert_placeholder.empty()

            status_txt = "FIRE DETECTED!" if fire_conf > 0.90 else "Monitoring"
            status_clr = "#ef4444" if fire_conf > 0.90 else "#22c55e"
            metrics_placeholder.markdown(f"""
<div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 12px;">
    <div class="metric-card">
        <div class="metric-title" style="color:#ef4444;">🔥 FIRE</div>
        <div class="metric-value">{fire_conf * 100:.1f}%</div>
        <div class="metric-sub">Confidence</div>
    </div>
    <div class="metric-card">
        <div class="metric-title" style="color:#94a3b8;">💨 SMOKE</div>
        <div class="metric-value">{smoke_conf * 100:.1f}%</div>
        <div class="metric-sub">Confidence</div>
    </div>
    <div class="metric-card">
        <div class="metric-title" style="color:#f59e0b;">⏱️ DURATION</div>
        <div class="metric-value">0.0s</div>
        <div class="metric-sub">Static Image</div>
    </div>
    <div class="metric-card">
        <div class="metric-title" style="color:#22c55e;">🛡️ STATUS</div>
        <div class="metric-value" style="color:{status_clr}; font-size:14px; margin-top:3px;">{status_txt}</div>
        <div class="metric-sub">System status</div>
    </div>
</div>""", unsafe_allow_html=True)

            st.image(res_plotted, use_container_width=True)

        else:
            alert_placeholder.empty()
            st.markdown("""
<div class="display-card">
    <div style="font-size:40px;">📷</div>
    <div>Upload an image to start AI detection</div>
</div>""", unsafe_allow_html=True)

    else:
        run_webcam = st.checkbox("🟢 Enable Webcam Stream", key="webcam_checkbox_main")
        video_placeholder = st.empty()

        if run_webcam:
            cap = cv2.VideoCapture(0)
            fire_start_time = None
            alert_active = False

            while run_webcam:
                ret, frame = cap.read()
                if not ret:
                    st.error("❌ Cannot access webcam.")
                    break

                clock_placeholder.markdown(
                    f'<div style="display: flex; justify-content: flex-end;"><div class="clock-badge">🕒 {datetime.now().strftime("%H:%M")}</div></div>',
                    unsafe_allow_html=True,
                )

                results = model(frame)
                res_plotted = results[0].plot()

                fire_conf = 0.0
                smoke_conf = 0.0

                for box in results[0].boxes:
                    conf = box.conf[0].item()
                    cls_id = int(box.cls[0].item())
                    cls_name = model.names[cls_id].lower() if hasattr(model, 'names') else ''

                    if 'smoke' in cls_name:
                        if conf > smoke_conf: smoke_conf = conf
                    else:
                        if conf > fire_conf: fire_conf = conf

                elapsed_time = 0.0
                if fire_conf > 0.90:
                    if fire_start_time is None:
                        fire_start_time = time.time()
                    elapsed_time = time.time() - fire_start_time
                else:
                    fire_start_time = None

                is_emergency = (elapsed_time >= 15.0)

                # จัดการแสดงผล Alert Banner และปุ่มกดปิดเสียง
                if is_emergency:
                    if not alert_active:
                        with alert_placeholder:
                            render_alert_with_audio(
                                "⚠️ EMERGENCY: FIRE DETECTED!",
                                "Continuous fire detected over 15 seconds!",
                                selected_sound_code
                            )
                        alert_active = True
                else:
                    if alert_active:
                        alert_placeholder.empty()
                        alert_active = False

                if is_emergency:
                    status_txt = "EMERGENCY!"
                    status_clr = "#ef4444"
                elif fire_conf > 0.90:
                    status_txt = "Warning..."
                    status_clr = "#f59e0b"
                else:
                    status_txt = "Monitoring"
                    status_clr = "#22c55e"

                metrics_placeholder.markdown(f"""
<div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 12px;">
    <div class="metric-card">
        <div class="metric-title" style="color:#ef4444;">🔥 FIRE</div>
        <div class="metric-value">{fire_conf * 100:.1f}%</div>
        <div class="metric-sub">Confidence</div>
    </div>
    <div class="metric-card">
        <div class="metric-title" style="color:#94a3b8;">💨 SMOKE</div>
        <div class="metric-value">{smoke_conf * 100:.1f}%</div>
        <div class="metric-sub">Confidence</div>
    </div>
    <div class="metric-card">
        <div class="metric-title" style="color:#f59e0b;">⏱️ DURATION</div>
        <div class="metric-value">{elapsed_time:.1f}s</div>
        <div class="metric-sub">Target: >15.0s</div>
    </div>
    <div class="metric-card">
        <div class="metric-title" style="color:#22c55e;">🛡️ STATUS</div>
        <div class="metric-value" style="color:{status_clr}; font-size:14px; margin-top:3px;">{status_txt}</div>
        <div class="metric-sub">System status</div>
    </div>
</div>""", unsafe_allow_html=True)

                res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                video_placeholder.image(res_plotted_rgb, channels="RGB", use_container_width=True)

            if cap is not None:
                cap.release()
        else:
            alert_placeholder.empty()
            video_placeholder.markdown("""
<div class="display-card">
    <div style="font-size:40px;">📹</div>
    <div>Check the box above to activate Webcam</div>
</div>""", unsafe_allow_html=True)

    st.markdown("""
<div class="tip-box">
    🛡️ <b>Tip:</b> Detection threshold is set to <b>90% confidence</b>. When an alert triggers, click the red <b>STOP ALARM</b> button on the alert banner to silence the audio immediately.
</div>""", unsafe_allow_html=True)