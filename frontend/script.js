const webcam = document.getElementById('webcam');
const statusBadge = document.getElementById('status-badge');
const eyeStatusText = document.getElementById('eye-status');
const scoreText = document.getElementById('score-text');
const scoreFill = document.getElementById('score-fill');
const confidenceText = document.getElementById('confidence');
const connectionStatus = document.getElementById('connection-status');
const startBtn = document.getElementById('start-btn');
const alarmSound = document.getElementById('alarm-sound');

let ws = null;
let stream = null;
let intervalId = null;
const FRAME_RATE = 10; // FPS
const WS_URL = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1"
    ? "ws://localhost:8000/ws"
    : "wss://driver-drowsiness-detection-1snh.onrender.com/ws";

async function startCamera() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
        webcam.srcObject = stream;
        return true;
    } catch (err) {
        console.error("Camera error:", err);
        alert("Could not access webcam. Please ensure permissions are granted.");
        return false;
    }
}

function initWebSocket() {
    ws = new WebSocket(WS_URL);

    ws.onopen = () => {
        connectionStatus.innerHTML = '<span class="dot"></span> WebSocket Connected';
        connectionStatus.classList.add('status-connected');
        statusBadge.textContent = "ACTIVE";
        console.log("Connected to Backend");
    };

    ws.onclose = () => {
        connectionStatus.innerHTML = '<span class="dot"></span> WebSocket Disconnected';
        connectionStatus.classList.remove('status-connected');
        statusBadge.textContent = "OFFLINE";
        stopMonitoring();
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        updateUI(data);

        // Request-Response Pattern: Send next frame only AFTER receiving the previous result
        // This prevents buffer bloat and ensures the AI is always processing the most RECENT image
        if (intervalId) { // intervalId is now used as a "is monitoring" flag
            setTimeout(sendFrame, 33); // Small delay to avoid hammering the CPU (approx 30 FPS cap)
        }
    };

    ws.onerror = (err) => {
        console.error("WebSocket Error:", err);
    };
}

function updateUI(data) {
    // Eye Status
    eyeStatusText.textContent = data.status || "---";

    // Score
    const maxScore = 15;
    const percentage = (data.score / maxScore) * 100;
    scoreFill.style.width = `${Math.min(100, percentage)}%`;
    scoreText.textContent = `${data.score} / ${maxScore}`;

    // Confidence
    confidenceText.textContent = `${Math.round(data.confidence * 100)}%`;

    // Drowsiness State Visuals
    if (data.is_drowsy) {
        document.body.classList.add('state-drowsy');
        document.body.classList.remove('state-awake');
        statusBadge.textContent = "DROWSY DETECTED";

        // Play Alarm
        if (alarmSound.paused) {
            alarmSound.play().catch(e => console.error("Audio playback stalled:", e));
        }
    } else {
        document.body.classList.remove('state-drowsy');
        document.body.classList.add('state-awake');
        statusBadge.textContent = "MONITORING";

        // Stop Alarm
        alarmSound.pause();
        alarmSound.currentTime = 0;
    }
}

function sendFrame() {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;

    const canvas = document.createElement('canvas');
    canvas.width = webcam.videoWidth;
    canvas.height = webcam.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(webcam, 0, 0, canvas.width, canvas.height);

    // Convert to base64
    const dataUrl = canvas.toDataURL('image/jpeg', 0.6); // Compress slightly for latency
    ws.send(dataUrl);
}

async function startMonitoring() {
    const cameraOk = await startCamera();
    if (!cameraOk) return;

    initWebSocket();

    startBtn.textContent = "Stop Monitoring";
    startBtn.classList.add('danger-btn');

    // Set flag and send the first frame to start the loop
    intervalId = true;
    setTimeout(sendFrame, 1000); // Give the WebSocket a second to connect
}

function stopMonitoring() {
    if (intervalId) {
        intervalId = null;
    }

    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        webcam.srcObject = null;
    }

    if (ws) {
        ws.close();
        ws = null;
    }

    startBtn.textContent = "Start Monitoring";
    startBtn.classList.remove('danger-btn');
    statusBadge.textContent = "INITIALIZING";
}

startBtn.addEventListener('click', () => {
    if (intervalId) {
        stopMonitoring();
    } else {
        startMonitoring();
    }
});
