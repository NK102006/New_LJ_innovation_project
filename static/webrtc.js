// WebRTC/MediaPipe.js - Client-side camera and ML processing
// Replaces server-side OpenCV/MediaPipe for cloud deployment

let faceMesh, hands;
let isInitialized = false;
let currentFilter = 'normal';
let socket;

// MediaPipe configuration
const FACE_MESH_CONFIG = {
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
};

const HANDS_CONFIG = {
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
};

// MediaPipe callbacks
function onFaceMeshResults(results) {
    if (!results.multiFaceLandmarks || results.multiFaceLandmarks.length === 0) {
        sendDetectionData({ face_detected: false, expression: 'None' });
        return;
    }

    const landmarks = results.multiFaceLandmarks[0];
    const expression = detectExpression(landmarks);

    sendDetectionData({
        face_detected: true,
        expression: expression
    });

    // Draw face mesh on canvas
    drawFaceMesh(results);
}

function onHandsResults(results) {
    if (!results.multiHandLandmarks || results.multiHandLandmarks.length === 0) {
        sendDetectionData({ gesture: 'none' });
        return;
    }

    for (let i = 0; i < results.multiHandLandmarks.length; i++) {
        const landmarks = results.multiHandLandmarks[i];
        const handedness = results.multiHandedness[i].label;
        const gesture = detectGesture(landmarks, handedness);

        sendDetectionData({ gesture: gesture });
    }

    // Draw hands on canvas
    drawHands(results);
}

// Detection functions (mirrors server-side logic)
function detectExpression(landmarks) {
    // Landmark IDs: 61 (Left Mouth), 291 (Right Mouth), 13 (Top Lip), 14 (Bottom Lip)
    const mouthWidth = Math.hypot(
        landmarks[61].x - landmarks[291].x,
        landmarks[61].y - landmarks[291].y
    );
    const mouthHeight = Math.hypot(
        landmarks[13].x - landmarks[14].x,
        landmarks[13].y - landmarks[14].y
    );

    if (mouthWidth > 0.08) return 'Happy';
    if (mouthHeight > 0.03) return 'Surprised';
    return 'Neutral';
}

function fingersUp(landmarks, handLabel) {
    const tips = [4, 8, 12, 16, 20];
    const fingers = [];

    if (handLabel === 'Right') {
        fingers.push(landmarks[tips[0]].x < landmarks[tips[0] - 1].x);
    } else {
        fingers.push(landmarks[tips[0]].x > landmarks[tips[0] - 1].x);
    }

    for (let i = 1; i < 5; i++) {
        fingers.push(landmarks[tips[i]].y < landmarks[tips[i] - 2].y);
    }

    return fingers;
}

function detectGesture(landmarks, handLabel) {
    const f = fingersUp(landmarks, handLabel);

    if (f.every(x => x === false)) return '✊';
    if (f.every(x => x === true)) return '🤚';
    if (f[0] === true && f[1] === true && f[2] === false && f[3] === false && f[4] === false) return '👍';
    if (f[0] === false && f[1] === true && f[2] === true && f[3] === false && f[4] === false) return '✌️';
    if (f[0] === false && f[1] === true && f[2] === false && f[3] === false && f[4] === false) return '☝️';
    if (f[0] === false && f[1] === true && f[2] === true && f[3] === true && f[4] === false) return '🤟';

    // Thumb-Index pinch
    const thumb = landmarks[4];
    const index = landmarks[8];
    const dist = Math.hypot(thumb.x - index.x, thumb.y - index.y);
    if (dist < 0.04) return '👌';

    return 'none';
}

// Web Speech API for client-side speech recognition
let recognition;
let isListening = false;

function initSpeechRecognition() {
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
        console.log('Speech recognition not supported');
        return;
    }

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;

    recognition.onresult = (event) => {
        let transcript = '';
        for (let i = event.resultIndex; i < event.results.length; i++) {
            transcript += event.results[i][0].transcript;
        }
        if (transcript.trim()) {
            socket.emit('speech_result', { text: transcript.trim() });
        }
    };

    recognition.onerror = (event) => {
        console.log('Speech recognition error:', event.error);
    };
}

function toggleSpeech() {
    if (!recognition) {
        initSpeechRecognition();
    }

    if (isListening) {
        recognition.stop();
        isListening = false;
        socket.emit('speech_toggle', { listening: false });
    } else {
        recognition.start();
        isListening = true;
        socket.emit('speech_toggle', { listening: true });
    }
    updateSpeechButton();
}

function updateSpeechButton() {
    const speechBtn = document.getElementById('speechToggle');
    if (isListening) {
        speechBtn.innerHTML = '⏹️ STOP Listening';
        speechBtn.className = 'speech-btn stop-btn';
    } else {
        speechBtn.innerHTML = '▶️ START Listening';
        speechBtn.className = 'speech-btn start-btn';
    }
}

// Send detection data to server via Socket.IO
function sendDetectionData(data) {
    if (socket && socket.connected) {
        socket.emit('detection_data', data);
    }
}

// Canvas drawing functions
let canvas, ctx;
let videoElement;

function drawFaceMesh(results) {
    if (!canvas || !ctx) return;

    ctx.save();
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw video frame first
    if (videoElement && videoElement.readyState >= 2) {
        ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
    }

    // Apply current filter
    applyFilter(ctx, videoElement);

    // Draw face mesh
    if (results.multiFaceLandmarks) {
        for (const landmarks of results.multiFaceLandmarks) {
            drawConnectors(ctx, landmarks, faceMesh.FACEMESH_CONTOURS,
                { color: '#00FF00', lineWidth: 1 });
            drawLandmarks(ctx, landmarks, { color: '#FF0000', lineWidth: 0.5 });
        }
    }

    ctx.restore();
}

function drawHands(results) {
    if (!canvas || !ctx || !results.multiHandLandmarks) return;

    ctx.save();

    for (const landmarks of results.multiHandLandmarks) {
        drawConnectors(ctx, landmarks, HANDS_CONNECTIONS,
            { color: '#00FFFF', lineWidth: 2 });
        drawLandmarks(ctx, landmarks, { color: '#FF00FF', lineWidth: 1 });
    }

    ctx.restore();
}

function applyFilter(ctx, video) {
    // Filter application is handled server-side for the actual frame
    // Client-side canvas just shows the raw + MediaPipe overlay
}

// Canvas filter presets (for display only - processing still server-side)
const filters = {
    normal: () => {},
    bw: (ctx, w, h) => {
        const imageData = ctx.getImageData(0, 0, w, h);
        const data = imageData.data;
        for (let i = 0; i < data.length; i += 4) {
            const avg = (data[i] + data[i + 1] + data[i + 2]) / 3;
            data[i] = data[i + 1] = data[i + 2] = avg;
        }
        ctx.putImageData(imageData, 0, 0);
    },
    red: (ctx, w, h) => {
        const imageData = ctx.getImageData(0, 0, w, h);
        const data = imageData.data;
        for (let i = 0; i < data.length; i += 4) {
            data[i + 2] = Math.min(255, data[i + 2] + 60);
        }
        ctx.putImageData(imageData, 0, 0);
    },
    blur: () => {
        ctx.filter = 'blur(10px)';
    },
    cartoon: () => {
        ctx.filter = 'saturate(1.5) contrast(1.2)';
    }
};

// Initialize everything
async function initCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480, facingMode: 'user' }
        });

        videoElement.srcObject = stream;
        await videoElement.play();

        canvas = document.getElementById('outputCanvas');
        ctx = canvas.getContext('2d');

        // Set canvas size
        canvas.width = videoElement.videoWidth || 640;
        canvas.height = videoElement.videoHeight || 480;

        await initMediaPipe();

        requestAnimationFrame(processFrame);

        console.log('✅ Camera initialized successfully');
        updateLiveStatus('🟢 LIVE', true);

    } catch (error) {
        console.error('❌ Camera initialization failed:', error);
        updateLiveStatus('🔴 Camera Error', false);
    }
}

async function initMediaPipe() {
    try {
        // Initialize Face Mesh
        faceMesh = new FaceMesh(FACE_MESH_CONFIG);
        faceMesh.setOptions({
            maxNumFaces: 1,
            refineLandmarks: true,
            minDetectionConfidence: 0.5,
            minTrackingConfidence: 0.5
        });
        faceMesh.onResults(onFaceMeshResults);

        // Initialize Hands
        hands = new Hands(HANDS_CONFIG);
        hands.setOptions({
            maxNumHands: 2,
            minDetectionConfidence: 0.7,
            minTrackingConfidence: 0.5
        });
        hands.onResults(onHandsResults);

        isInitialized = true;
        console.log('✅ MediaPipe initialized');

    } catch (error) {
        console.error('❌ MediaPipe init failed:', error);
    }
}

async function processFrame() {
    if (!isInitialized || !videoElement) {
        requestAnimationFrame(processFrame);
        return;
    }

    // Send frame to MediaPipe for processing
    await faceMesh.send({ image: videoElement });
    await hands.send({ image: videoElement });

    requestAnimationFrame(processFrame);
}

// External API called by HTML
window.setFilter = function(name) {
    currentFilter = name;
    document.querySelectorAll('.filter-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    event.target.classList.add('active');
    fetch(`/filter/${name}`);
};

window.toggleSpeechWebRTC = toggleSpeech;

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    videoElement = document.getElementById('videoFeed');

    // Socket.IO is already initialized in index.html
    socket = window.socket;

    initCamera();
});
