from flask import Flask, render_template_string, request, jsonify, send_from_directory, send_file
import os
import cv2
import numpy as np
import re
from werkzeug.utils import secure_filename
from scipy.signal import savgol_filter
import subprocess
import requests
import ffmpeg

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['FRAME_FOLDER'] = 'frames'
app.config['PROCESSED_FOLDER'] = 'processed'
app.config['PROCESSED_VIDEO'] = 'output.mp4'
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # Limit upload size to 200MB
app.config['AUDIO_FOLDER'] = 'static_audio'

pixels_per_meter = 50

roi_coords = None
video_path = None
frame_count = 0
frame_map = {}
trajectory = []
accumulated_trajectory = []

for folder in [app.config['UPLOAD_FOLDER'], app.config['FRAME_FOLDER'], app.config['PROCESSED_FOLDER'], app.config['AUDIO_FOLDER']]:
    os.makedirs(folder, exist_ok=True)

def extract_frames(video_path):
    global frame_count
    # Clear frames folder
    for f in os.listdir(app.config['FRAME_FOLDER']):
        os.remove(os.path.join(app.config['FRAME_FOLDER'], f))
    cap = cv2.VideoCapture(video_path)
    cnt = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        roi = frame[:min(800, frame.shape[0]), :]
        if roi.shape[1] > 2400:
            roi = roi[:, 1200:roi.shape[1]-1200]
        cv2.imwrite(f"{app.config['FRAME_FOLDER']}/{cnt}.png", roi)
        cnt += 1
    cap.release()
    # Update frame_count to actual number of frames saved
    frame_count = len([f for f in os.listdir(app.config['FRAME_FOLDER']) if f.endswith('.png')])

def generate_processed_video():
    frame_files = sorted([f for f in os.listdir(app.config['PROCESSED_FOLDER']) if f.endswith('.jpg')],
                         key=lambda x: int(re.sub(r'\D', '', x)))
    if not frame_files:
        return
    sample_frame = cv2.imread(os.path.join(app.config['PROCESSED_FOLDER'], frame_files[0]))
    height, width = sample_frame.shape[:2]
    out_path = os.path.join(app.config['PROCESSED_FOLDER'], app.config['PROCESSED_VIDEO'])
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
    for f in frame_files:
        img = cv2.imread(os.path.join(app.config['PROCESSED_FOLDER'], f))
        out.write(img)
    out.release()

def extract_audio(video_path, output_audio_path):
    command = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-vn',  # No video
        '-acodec', 'libmp3lame',
        '-ar', '44100',
        '-ac', '2',
        '-b:a', '192k',
        output_audio_path
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def download_video_from_url(url):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        filename = secure_filename(url.split('/')[-1])
        if not filename or '.' not in filename:
            filename = 'downloaded_video.mp4'
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        with open(video_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return video_path, filename
    else:
        raise Exception("Failed to download video from URL.")

@app.route('/download')
def download_video():
    generate_processed_video()
    return send_file(os.path.join(app.config['PROCESSED_FOLDER'], app.config['PROCESSED_VIDEO']),
                     as_attachment=True, download_name="Processed_Trajectory.mp4")

@app.route('/static_audio/<path:filename>')
def serve_audio(filename):
    return send_from_directory('static_audio', filename)

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/')
def index():
    return render_template_string("""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Ball Tracker Web</title>
<style>
    body {
        background-color: #1c1c1c;
        color: white;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        margin: 0; padding: 0;
        display: flex;
        flex-direction: column;
        align-items: center;
        min-height: 100vh;
        height: 100vh;
    }
    header {
        padding: 20px;
        font-size: 2rem;
        font-weight: bold;
        color: #00d1b2;
        letter-spacing: 2px;
        text-shadow: 0 0 5px #00d1b2;
        user-select: none;
        flex-shrink: 0;
    }
    #uploadSection, #videoSection, #roiSection, #processedSection {
        background: #2a2a2a;
        margin: 10px;
        padding: 15px 20px;
        border-radius: 10px;
        box-shadow: 0 0 15px #00555588;
        width: 90%;
        max-width: 900px;
    }
    #urlSection {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
    }
    #videoSection {
        display: none;
        flex-direction: column;
        align-items: center;
        margin-top: 20px;
    }
    #roiSection {
        display: none;
        margin-top: 20px;
        flex-direction: column;
        align-items: center;
    }
    #processedSection {
        display: none;
        flex-direction: column;
        align-items: center;
    }
    label {
        margin-right: 10px;
        font-weight: 600;
    }
    button {
        padding: 8px 16px;
        border-radius: 6px;
        border: none;
        font-weight: 600;
        font-size: 1rem;
        margin: 10px 10px 10px 0;
        background: #015151;
        color: #80fff7;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    button:disabled {
        opacity: 0.4;
        cursor: not-allowed;
    }
    button:hover:not(:disabled) {
        background: #00e5ca;
        color: #003330;
    }
    #videoPlayer {
        width: 100%;
        max-width: 900px;
        border-radius: 10px;
        margin-top: 10px;
        outline: none;
    }
    #message {
        margin-top: 10px;
        height: 20px;
        font-weight: bold;
        color: #ffbaba;
        font-size: 1.1rem;
        letter-spacing: 0.5px;
        text-align: center;
        user-select: none;
    }
    #metrics {
        margin-top: 20px;
        font-size: 1.1rem;
        color: #80fff7;
        font-weight: 600;
        user-select: none;
        text-align: center;
    }
    #frameInfo {
        margin-top: 10px;
        font-size: 1.2rem;
        color: #00d1b2;
        user-select: none;
    }
        #urlSection label {
        font-weight: 700;
        font-size: 1.2rem;
        user-select: none;
        color: #00d1b2;
    }
    #videoUrl {
        flex-grow: 1;
        padding: 8px;
        border-radius: 6px;
        border: 1px solid #ccc;
        font-size: 1rem;
        background: #444;
        color: white;
        outline: none;
        transition: border-color 0.3s ease;
    }
    #videoUrl:focus {
        border-color: #00e5ca;
        background: #333;
    }
    #canvasContainer {
        position: relative;
        margin-top: 10px;
        border: 3px solid #00d1b2cc;
        border-radius: 10px;
        display: inline-block;
        cursor: crosshair;
    }
    canvas {
        border-radius: 10px;
        max-width: 100%;
        height: auto;
        display: block;
    }
    #roiCanvas {
        position: absolute;
        top: 0; left: 0;
        user-select: none;
    }
    #processedFrame {
        max-width: 100%;
        border-radius: 10px;
        margin-top: 10px;
        border: 3px solid #00d1b2cc;
    }
    #processedControls {
        margin-top: 10px;
    }
    #frameSlider {
        width: 80%;
        margin-top: 15px;
        -webkit-appearance: none;
        height: 8px;
        background: #044;
        border-radius: 4px;
        outline: none;
    }
    #frameSlider::-webkit-slider-thumb {
        -webkit-appearance: none;
        appearance: none;
        width: 20px;
        height: 20px;
        border-radius: 50%;
        background: #00e5ca;
        cursor: pointer;
        box-shadow: 0 0 2px #00e5ca;
        transition: background 0.3s ease;
    }
    #frameSlider::-webkit-slider-thumb:hover {
        background: #057a6b;
    }
    #snickometerContainer {
        width: 30%; /* Reduced width */
        margin-top: 10px;
        background: #222;
        border-radius: 10px;
        padding: 5px 0;
        border: 2px solid #00d1b2cc;
    }
    #snickometerCanvas {
        width: 100%;
        height: 100px;
        display: block;
        background: black;
        border-radius: 8px;
    }
    #videoOverlayContainer {
        position: relative;
        display: inline-block;
    }
    #snickometerCanvasOverlay {
        position: absolute;
        bottom: 0;
        left: 0;
        width: 100%;
        height: 80px;
        background: transparent;
        pointer-events: none;
    }
</style>
</head>
<body>
<header>Ball Tracker Web</header>

<section id="uploadSection">
    <label for="videoFile">Select Video (MP4):</label>
    <input type="file" id="videoFile" accept="video/mp4" />
    <button id="uploadBtn" disabled>Upload &amp; Extract Frames</button>
</section>
                                  
<section id="urlSection">
    <div style="background: #2a2a2a; padding: 15px 20px; border-radius: 10px; box-shadow: 0 0 15px #00555588; width: 98%; max-width: 1300px;">
        <label style="font-weight: 600; color: white;">OR</label>
        <div style="display: flex; gap: 10px; margin-top: 10px; align-items: center;">
            <input type="text" id="videoUrl" placeholder="Paste Video URL" 
                style="flex-grow: 1; min-width: 625px; height: 40px; padding: 0 12px; border-radius: 6px; border: 1px solid #ccc; font-size: 1rem; background: #444; color: white; outline: none; transition: border-color 0.3s ease;" />
            <button id="urlUploadBtn" 
                style="height: 40px; padding: 0 20px; border-radius: 6px; border: none; font-weight: 600; font-size: 1rem; background: #015151; color: #80fff7; cursor: pointer; transition: all 0.3s ease;">
                Upload &amp; Extract Frames
            </button>
        </div>
    </div>
</section>

<div id="message"></div>

<section id="videoSection">
  <h3>Video Playback</h3>
  <video id="videoPlayer" controls></video>

  <div style="display: flex; justify-content: space-between; width: 100%; margin-top: 15px; flex-wrap: wrap;">
    <!-- Left: Frame Info + Buttons (moved more right) -->
    <div style="display: flex; flex-direction: column; align-items: flex-start; gap: 10px; margin-left: 120px;">
      <div id="frameInfo" style="font-size: 1.1rem;">Current Frame: <span id="currentFrame">0</span></div>
      <button id="setStartFrameBtn" disabled>Set Start Frame</button>
      <button id="setEndFrameBtn" disabled>Set End Frame</button>
    </div>

    <!-- Right: Snickometer (moved more left) -->
    <div id="snickometerContainer" style="width: 30%; height: 62%; background: #222; border-radius: 10px; padding: 5px 0; border: 2px solid #00d1b2cc; margin-right: 100px; margin-top: 40px;">
      <canvas id="snickometerCanvas" style="margin-top: 1px;"></canvas>
    </div>
  </div>
</section>



<section id="roiSection">
    <p><strong>Step 3:</strong> Draw ROI on the selected frame (click and drag)</p>
    <div id="canvasContainer">
        <canvas id="frameCanvas"></canvas>
        <canvas id="roiCanvas"></canvas>
    </div>
    <p>ROI coords: <span id="roiCoords">Not set</span></p>
    <button id="setRoiBtn" disabled>Set ROI</button>
</section>

<section id="processedSection">
    <h3>Processed Frame Output</h3>
    <div id="videoOverlayContainer">
    <img id="processedFrame" alt="Processed frame" />
    <canvas id="snickometerCanvasOverlay"></canvas>
    </div>

    <div id="processedControls">
        <button id="rewindBtn" disabled>⏮ Rewind</button>
        <button id="pauseBtn" disabled>⏸ Pause</button>
        <button id="playBtn" disabled>▶ Play Processed Video</button>
        <button id="forwardBtn" disabled>⏭ Forward</button>
        <button id="downloadBtn">⬇ Download Video</button>
    </div>
    <div id="metrics"></div>
    <input type="range" id="frameSlider" min="0" max="0" value="0" style="margin-top:10px;" />
</section>

<div id="message"></div>

<script>
const videoInput = document.getElementById('videoFile');
const uploadBtn = document.getElementById('uploadBtn');
const videoUrlInput = document.getElementById('videoUrl');
const urlUploadBtn = document.getElementById('urlUploadBtn');
const videoPlayer = document.getElementById('videoPlayer');
const currentFrameSpan = document.getElementById('currentFrame');
const setStartFrameBtn = document.getElementById('setStartFrameBtn');
const setEndFrameBtn = document.getElementById('setEndFrameBtn');
const videoSection = document.getElementById('videoSection');
const roiSection = document.getElementById('roiSection');
const processedSection = document.getElementById('processedSection');
const frameCanvas = document.getElementById('frameCanvas');
const roiCanvas = document.getElementById('roiCanvas');
const roiCoordsSpan = document.getElementById('roiCoords');
const setRoiBtn = document.getElementById('setRoiBtn');
const messageDiv = document.getElementById('message');
const processedFrame = document.getElementById('processedFrame');
const rewindBtn = document.getElementById('rewindBtn');
const pauseBtn = document.getElementById('pauseBtn');
const playBtn = document.getElementById('playBtn');
const forwardBtn = document.getElementById('forwardBtn');
const downloadBtn = document.getElementById('downloadBtn');
const metricsDiv = document.getElementById('metrics');
const frameSlider = document.getElementById('frameSlider');

let videoUploaded = false;
let framesCount = 0;
let startFrame = null;
let endFrame = null;
let currentProcessedFrame = 0;
let playingProcessed = false;
let playInterval = null;
let processedStartFrame = 0;
let processedEndFrame = 0;
const fps = 60; // original video fps

// Upload controls
videoInput.addEventListener('change', () => {
  uploadBtn.disabled = videoInput.files.length === 0;
  messageDiv.textContent = "";
});

uploadBtn.addEventListener('click', () => {
  if (videoInput.files.length === 0) return;
  uploadBtn.disabled = true;
  messageDiv.textContent = "Uploading and extracting frames... Please wait.";
  const formData = new FormData();
  formData.append('video', videoInput.files[0]);

  fetch('/upload', {
    method: 'POST',
    body: formData,
  }).then(res => res.json())
    .then(data => {
      if(data.success){
        framesCount = data.frame_count;
        messageDiv.textContent = "✅ Video uploaded and frames extracted: " + framesCount + " frames.";
        videoUploaded = true;

        const fileURL = URL.createObjectURL(videoInput.files[0]);
        videoPlayer.src = fileURL;
        videoPlayer.load();
        videoSection.style.display = 'flex';
        // Enable start and end frame buttons after upload
        setStartFrameBtn.disabled = false;
        setEndFrameBtn.disabled = false;
        uploadBtn.disabled = true;
        videoInput.disabled = true;

        roiSection.style.display = 'none';
        processedSection.style.display = 'none';
      } else {
        messageDiv.textContent = 'Error: ' + data.message;
        uploadBtn.disabled = false;
      }
  }).catch((error) => {
  console.warn('Upload failed silently:', error);
  uploadBtn.disabled = false;
});
});
urlUploadBtn.addEventListener('click', () => {
  const videoUrl = videoUrlInput.value.trim();
  if (!videoUrl) {
    alert('Please enter a valid video URL.');
    return;
  }
  urlUploadBtn.disabled = true;
  videoUrlInput.disabled = true;
  messageDiv.textContent = "Fetching video from URL... Please wait.";

  fetch('/fetch_video', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({url: videoUrl})
  }).then(res => res.json())
    .then(data => {
      if(data.success){
        framesCount = data.frame_count;
        messageDiv.textContent = "✅ Video fetched and frames extracted: " + framesCount + " frames.";
        videoUploaded = true;

        videoPlayer.src = data.video_url;
        videoPlayer.load();
        videoSection.style.display = 'flex';
        setStartFrameBtn.disabled = false;
        setEndFrameBtn.disabled = false;

        uploadBtn.disabled = true;
        videoInput.disabled = true;
      } else {
        messageDiv.textContent = 'Error: ' + data.message;
        urlUploadBtn.disabled = false;
        videoUrlInput.disabled = false;
      }
  }).catch(() => {
    messageDiv.textContent = 'Failed to fetch video from URL. Please try again.';
    urlUploadBtn.disabled = false;
    videoUrlInput.disabled = false;
  });
});

// Video playback tracking
function getCurrentVideoFrame(){
  if (!videoPlayer.duration || !framesCount) return 0;
  return Math.min(framesCount - 1, Math.floor(videoPlayer.currentTime * fps));
}
videoPlayer.addEventListener('timeupdate', () => {
  const frame = getCurrentVideoFrame();
  currentFrameSpan.textContent = frame;
});

setStartFrameBtn.addEventListener('click', () => {
  const frame = getCurrentVideoFrame();
  startFrame = frame;
  messageDiv.textContent = 'Start frame set to ' + frame;
  checkFramesSet();
});
setEndFrameBtn.addEventListener('click', () => {
  const frame = getCurrentVideoFrame();
  endFrame = frame;
  messageDiv.textContent = 'End frame set to ' + frame;
  checkFramesSet();
});

function checkFramesSet(){
  if(startFrame !== null && endFrame !== null){
    if(startFrame > endFrame){
      messageDiv.textContent = 'Start frame should be less than or equal to end frame.';
      return;
    }
    messageDiv.textContent = 'Start and End frames set. Please draw ROI on the next step.';
    videoSection.style.display = 'none';
    roiSection.style.display = 'flex';
    loadFrameForROI(startFrame);
    setStartFrameBtn.disabled = true;
    setEndFrameBtn.disabled = true;
    setRoiBtn.disabled = false;
  }
}

function loadFrameForROI(frameNumber){
  fetch('/get_frame/' + frameNumber).then(res => {
    if(res.ok) return res.blob();
    throw new Error("Failed to load frame");
  }).then(blob => {
    const imgURL = URL.createObjectURL(blob);
    let img = new Image();
    img.onload = function(){
      frameCanvas.width = roiCanvas.width = img.width;
      frameCanvas.height = roiCanvas.height = img.height;
      const ctx = frameCanvas.getContext('2d');
      ctx.drawImage(img, 0, 0);
      roi = null;
      drawRoiRect();
      URL.revokeObjectURL(imgURL);
    }
    img.src = imgURL;
  }).catch(() => {
    messageDiv.textContent = "Error loading frame for ROI selection.";
  });
}

let roi = null;
const roiCtx = roiCanvas.getContext('2d');

function drawRoiRect() {
  roiCtx.clearRect(0, 0, roiCanvas.width, roiCanvas.height);
  if (roi) {
    roiCtx.strokeStyle = 'lime';
    roiCtx.lineWidth = 3;
    roiCtx.setLineDash([6]);
    roiCtx.strokeRect(roi.x, roi.y, roi.width, roi.height);
    roiCtx.setLineDash([]);
  }
}

let isDrawing = false;
let startX, startY;

// Helper to get scaled coordinates
function getRelativeCoords(event, canvas) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  return {
    x: (event.clientX - rect.left) * scaleX,
    y: (event.clientY - rect.top) * scaleY
  };
}

roiCanvas.addEventListener('mousedown', (e) => {
  const pos = getRelativeCoords(e, roiCanvas);
  startX = pos.x;
  startY = pos.y;
  isDrawing = true;
});

roiCanvas.addEventListener('mousemove', (e) => {
  if (!isDrawing) return;
  const pos = getRelativeCoords(e, roiCanvas);
  const width = pos.x - startX;
  const height = pos.y - startY;
  roi = {
    x: Math.min(startX, pos.x),
    y: Math.min(startY, pos.y),
    width: Math.abs(width),
    height: Math.abs(height)
  };
  drawRoiRect();
});

roiCanvas.addEventListener('mouseup', () => {
  isDrawing = false;
});

roiCanvas.addEventListener('mouseleave', () => {
  isDrawing = false;
});

setRoiBtn.addEventListener('click', () => {
  if (!roi) {
    alert('Please draw ROI first.');
    return;
  }
  fetch('/set_roi', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(roi)
  }).then(res => res.json()).then(data => {
    if (data.success) {
      roiCoordsSpan.textContent = 'x=${roi.x.toFixed(0)}, y=${roi.y.toFixed(0)}, w=${roi.width.toFixed(0)}, h=${roi.height.toFixed(0)}';
      messageDiv.textContent = '✅ ROI set successfully. Running analysis...';
      runAnalysis();
    } else {
      messageDiv.textContent = 'Failed to set ROI: ' + data.message;
    }
  });
});

// Analysis and display functions with slider
function runAnalysis(){
  fetch('/run_analysis',{
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({start_frame: startFrame, end_frame: endFrame})
  }).then(res => res.json()).then(data => {
    if(data.success){
      messageDiv.textContent = '✅ Analysis complete. Playing processed video.';
      roiSection.style.display = 'none';
      processedSection.style.display = 'flex';
      rewindBtn.disabled = false;
      pauseBtn.disabled = false;
      playBtn.disabled = false;
      forwardBtn.disabled = false;
      processedStartFrame = 0;
      processedEndFrame = data.processed_frame_count ? data.processed_frame_count - 1 : endFrame;
      displayMetrics(data.metrics);
      currentProcessedFrame = processedStartFrame;
      displayProcessedFrame(currentProcessedFrame);
      frameSlider.max = processedEndFrame;  // Set slider max
      frameSlider.value = currentProcessedFrame;  // Reset slider value
    } else {
      messageDiv.textContent = 'Analysis failed: ' + data.message;
    }
  }).catch(() => {
    messageDiv.textContent = 'Analysis request failed.';
  });
}

function displayProcessedFrame(frameNum){
  processedFrame.src = '/processed_frame/' + frameNum;
  frameSlider.value = frameNum;  // Sync slider position
}

playBtn.addEventListener('click', () => {
  if(playingProcessed){
    pauseProcessedPlayback();
  } else {
    startProcessedPlayback();
  }
});

rewindBtn.addEventListener('click', () => {
  pauseProcessedPlayback();
  currentProcessedFrame = processedStartFrame;
  displayProcessedFrame(currentProcessedFrame);
});

pauseBtn.addEventListener('click', () => {
  pauseProcessedPlayback();
});

forwardBtn.addEventListener('click', () => {
  pauseProcessedPlayback();
  if (currentProcessedFrame < processedEndFrame) {
    currentProcessedFrame++;
    displayProcessedFrame(currentProcessedFrame);
  }
});

downloadBtn.addEventListener('click', () => {
  window.location.href = '/download';
});

// Slider event for frame navigation
frameSlider.addEventListener('input', (e) => {
  pauseProcessedPlayback();
  currentProcessedFrame = parseInt(e.target.value);
  displayProcessedFrame(currentProcessedFrame);
});

function startProcessedPlayback(){
  if(!audioContext) {
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
  }
  audioContext.resume();
  if(processedStartFrame === null || processedEndFrame === null){
    alert('Frame range must be set before playing processed video.');
    return;
  }
  playingProcessed = true;
  playBtn.textContent = '⏸ Pause';
  rewindBtn.disabled = false;
  pauseBtn.disabled = false;
  forwardBtn.disabled = false;
  messageDiv.textContent = 'Playing processed video...';
  currentProcessedFrame = currentProcessedFrame < processedStartFrame ? processedStartFrame : currentProcessedFrame;
  if(playInterval) clearInterval(playInterval);
  playInterval = setInterval(() => {
    displayProcessedFrame(currentProcessedFrame);
    if(currentProcessedFrame >= processedEndFrame){
      pauseProcessedPlayback();
      messageDiv.textContent = 'Processed video ended.';
    } else {
      currentProcessedFrame++;
    }
  }, 1000/fps);
}

function pauseProcessedPlayback(){
  if(audioContext)
    audioContext.suspend();
  playingProcessed = false;
  playBtn.textContent = '▶ Play Processed Video';
  if(playInterval){
    clearInterval(playInterval);
    playInterval = null;
  }
  messageDiv.textContent = 'Processed video paused.';
}

function displayMetrics(metrics){
  metricsDiv.innerHTML =
    'Speed: ' + metrics.speed.toFixed(2) + ' km/h &nbsp;&nbsp; | &nbsp;&nbsp;' +
    'Swing: ' + metrics.swing.toFixed(2) + '° &nbsp;&nbsp; | &nbsp;&nbsp;' +
    'Turn: ' + metrics.turn.toFixed(2) + '° &nbsp;&nbsp; | &nbsp;&nbsp;' +
    'Bounce Height: ' + metrics.bounce.toFixed(2) + ' m';
}

let snickCanvas = document.getElementById('snickometerCanvas');
let snickCtx = snickCanvas.getContext('2d');
let audioContext;
let analyser;
let dataArray;
let source;

function setupSnickometerWithVideo(videoElement) {
    if (audioContext) {
        audioContext.close();
    }
    audioContext = new (window.AudioContext || window.webkitAudioContext)();

    const rect = snickCanvas.getBoundingClientRect();
    snickCanvas.width = rect.width;
    snickCanvas.height = rect.height;

    source = audioContext.createMediaElementSource(videoElement);
    analyser = audioContext.createAnalyser();
    analyser.fftSize = 1024; // More sensitive
    dataArray = new Uint8Array(analyser.frequencyBinCount);

    source.connect(analyser);
    analyser.connect(audioContext.destination);

    drawSnickometer();
}

let redPulse = 0;

function drawSnickometer() {
    requestAnimationFrame(drawSnickometer);
    analyser.getByteTimeDomainData(dataArray);

    snickCtx.clearRect(0, 0, snickCanvas.width, snickCanvas.height);

    let sliceWidth = snickCanvas.width / dataArray.length;
    let x = 0;
    let spike = false;
    const amplification = 3.0; // Increased amplification multiplier for bigger spikes
    const spikeThreshold = 5; // Threshold for considering spike (adjusted for amplification)

    snickCtx.beginPath();

    for (let i = 0; i < dataArray.length; i++) {
        // Amplify deviation stronger for bigger spikes
        let v = (dataArray[i] - 128) / 32.0 * amplification; // amplified and scaled
        let y = snickCanvas.height / 2 + v * (snickCanvas.height / 2);

        if (Math.abs(dataArray[i] - 128) > spikeThreshold) {
            spike = true;
        }

        if (i === 0) {
            snickCtx.moveTo(x, y);
        } else {
            snickCtx.lineTo(x, y);
        }
        x += sliceWidth;
    }

    // Change line width if a spike is detected
    snickCtx.lineWidth = spike ? 2.5 : 1.5; // Thicker line for spikes
    snickCtx.strokeStyle = "rgba(255, 255, 255, 0.9)"; // Keep the line color white
    snickCtx.stroke();
}

// Automatically initialize Snickometer when video plays
videoPlayer.onplay = () => {
    if (!audioContext || audioContext.state === 'closed') {
        setupSnickometerWithVideo(videoPlayer);
    }
};

</script>
</body>
</html>
""")

@app.route('/get_frame/<int:frame_num>')
def get_frame(frame_num):
    path = os.path.join(app.config['FRAME_FOLDER'], f"{frame_num}.png")
    if os.path.exists(path):
        return send_from_directory(app.config['FRAME_FOLDER'], f"{frame_num}.png")
    return '', 404

@app.route('/processed_frame/<int:frame_num>')
def processed_frame(frame_num):
    path = os.path.join(app.config['PROCESSED_FOLDER'], f"{frame_num}.jpg")
    if os.path.exists(path):
        return send_from_directory(app.config['PROCESSED_FOLDER'], f"{frame_num}.jpg")
    else:
        return get_frame(frame_num)

@app.route('/set_roi', methods=['POST'])
def set_roi():
    global roi_coords
    data = request.json
    try:
        x = int(data['x'])
        y = int(data['y'])
        w = int(data['width'])
        h = int(data['height'])
        roi_coords = (x, y, w, h)
        return jsonify({'success': True, 'message': f'ROI set to {roi_coords}'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 400

@app.route('/upload', methods=['POST'])
def upload():
    global video_path, roi_coords, frame_count, frame_map, trajectory, accumulated_trajectory
    roi_coords = None
    frame_map = {}
    trajectory = []
    accumulated_trajectory = []

    try:
        # 1. Check for video file in request
        if 'video' not in request.files:
            return jsonify({'success': False, 'message': 'No video file uploaded'}), 400

        file = request.files['video']
        if file.filename == '':
            return jsonify({'success': False, 'message': 'Filename is empty'}), 400

        # 2. Clean the filename
        filename = secure_filename(file.filename)
        if not filename.lower().endswith('.mp4'):
            return jsonify({'success': False, 'message': 'Only MP4 files are supported'}), 400

        # 3. Create uploads folder if missing
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

        # 4. Save the uploaded file
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(video_path)

        # 5. Check if file was saved
        if not os.path.exists(video_path):
            return jsonify({'success': False, 'message': f"File not saved at {video_path}"}), 500

        print(f"✅ Video saved to: {video_path}")

        # 6. Check if ffmpeg is available
        try:
            subprocess.run(['ffmpeg', '-version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except FileNotFoundError:
            return jsonify({'success': False, 'message': 'ffmpeg not found. Please install FFmpeg and add to system PATH'}), 500

        # 7. Extract frames and audio
        extract_frames(video_path)
        extract_audio(video_path, os.path.join(app.config['AUDIO_FOLDER'], 'extracted_audio.mp3'))

        return jsonify({'success': True, 'frame_count': frame_count})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Upload processing failed: {str(e)}'}), 500
    
@app.route('/fetch_video', methods=['POST'])
def fetch_video():
    global video_path, roi_coords, frame_count, frame_map, trajectory, accumulated_trajectory
    roi_coords = None
    frame_map = {}
    trajectory = []
    accumulated_trajectory = []

    data = request.json
    video_url = data.get('url')
    if not video_url:
        return jsonify({'success': False, 'message': 'Video URL not provided'}), 400

    try:
        video_path_local, filename = download_video_from_url(video_url)
        video_path = video_path_local

        extract_frames(video_path)
        extract_audio(video_path, os.path.join(app.config['AUDIO_FOLDER'], 'extracted_audio.mp3'))

        video_url_path = '/uploads/' + filename
        return jsonify({'success': True, 'frame_count': frame_count, 'video_url': video_url_path})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error fetching video: {str(e)}'}), 500

def draw_smooth_line(points, img, color, thickness):
    if len(points) < 5:
        return img
    x = points[:, 0]
    y = points[:, 1]
    window = min(11, len(x) if len(x) % 2 == 1 else len(x) - 1)
    if window < 3:
        return img
    x_s = savgol_filter(x, window, 2)
    y_s = savgol_filter(y, window, 2)
    for i in range(1, len(x_s)):
        cv2.line(img, (int(x_s[i - 1]), int(y_s[i - 1])), (int(x_s[i]), int(y_s[i])), color, thickness)
    return img

def run_analysis_internal(start_frame, end_frame):
    global roi_coords, frame_map, trajectory, accumulated_trajectory

    for f in os.listdir(app.config['PROCESSED_FOLDER']):
        os.remove(os.path.join(app.config['PROCESSED_FOLDER'], f))

    x1, y1, w_roi, h_roi = roi_coords
    x2, y2 = x1 + w_roi, y1 + h_roi

    frame_files = sorted(
        [f for f in os.listdir(app.config['FRAME_FOLDER']) if f.endswith(".png")],
        key=lambda x: int(re.sub(r'\D', '', x))
    )

    frame_map = {}
    trajectory = []
    accumulated_trajectory = []

    last_position = None

    for f in frame_files:
        frame_number = int(re.sub(r'\D', '', f))
        img = cv2.imread(os.path.join(app.config['FRAME_FOLDER'], f))
        base = img.copy()

        if start_frame <= frame_number <= end_frame:
            roi = img[y1:y2, x1:x2]
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mask1 = cv2.inRange(hsv, (0, 100, 50), (10, 255, 255))
            mask2 = cv2.inRange(hsv, (160, 100, 50), (179, 255, 255))
            red_mask = cv2.bitwise_or(mask1, mask2)
            red_mask = cv2.erode(red_mask, None, iterations=1)
            red_mask = cv2.dilate(red_mask, None, iterations=2)

            contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            found_ball = False
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if min(w, h)/max(w, h) >= 0.5 and w <= 10 and h <= 10:
                    cx, cy = x + w//2 + x1, y + h//2 + y1
                    frame_map[frame_number] = (cx, cy)
                    trajectory.append((cx, cy))
                    last_position = (cx, cy)
                    found_ball = True
                    break
            if not found_ball and last_position:
                frame_map[frame_number] = last_position
                trajectory.append(last_position)
        else:
            if last_position:
                frame_map[frame_number] = last_position
                trajectory.append(last_position)

        if frame_number in frame_map:
            cx, cy = frame_map[frame_number]
            accumulated_trajectory.append((cx, cy))
            cv2.rectangle(base, (cx - 5, cy - 5), (cx + 5, cy + 5), (0, 255, 0), 2)

        if len(accumulated_trajectory) >= 6:
            points = np.array(accumulated_trajectory, dtype=np.float32)
            impact_idx = np.argmax(points[:,1])
            before = points[:impact_idx+1]
            after = points[impact_idx:]

            overlay = base.copy()
            for t in [12,10,8]:
                overlay = draw_smooth_line(after, overlay, (0,0,255), t)
            for t in [16,14,12]:
                temp = draw_smooth_line(before, overlay.copy(), (0,0,180), t)
                overlay = cv2.addWeighted(temp, 0.3, overlay, 0.7, 0)

            base = cv2.addWeighted(overlay, 0.6, base, 0.4, 0)

            if start_frame <= frame_number <= end_frame:
                total_distance_m = 20.12
                total_time_s = max((end_frame - start_frame)/60, 1e-5)
                speed = (total_distance_m / total_time_s)*3.6
                cv2.putText(base, f"Speed: {speed:.2f} km/h", (50,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

                if len(before) >=3:
                    x_start, y_start = before[0]
                    x_end, y_end = before[-1]
                    slope = (x_end - x_start) / (y_end-y_start) if (y_end-y_start) != 0 else 0
                    max_dev_px = max(abs(x - (x_start + slope*(y - y_start))) for (x,y) in before)
                    deviation_m = max_dev_px / pixels_per_meter
                    vertical_m = (y_end - y_start) / pixels_per_meter
                    if vertical_m > 0:
                        swing_deg = np.degrees(np.arctan(deviation_m / vertical_m))
                        swing_deg = min(max(swing_deg, 0), 1.5)
                        cv2.putText(base, f"Swing: {swing_deg:.2f}°", (50,90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180,220,255), 2)

                if len(after) >= 2:
                    ya, xa = after[:,1], after[:,0]
                    if len(set(ya)) > 1:
                        ma, _ = np.polyfit(ya, xa, 1)
                        mb, _ = np.polyfit(before[:,1], before[:,0], 1)
                        turn_deg = abs(np.degrees(np.arctan((ma - mb) / (1 + ma * mb))))
                        display_turn = turn_deg if 2.5 < turn_deg < 5.0 else 0.0
                        cv2.putText(base, f"Turn: {display_turn:.2f}°", (50,120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,200,200), 2)

                    peak_y = np.min(after[:,1])
                    bounce_px = after[0][1] - peak_y
                    bounce_height_m = bounce_px / pixels_per_meter if bounce_px > 0 else 0
                    cv2.putText(base, f"Bounce: {bounce_height_m:.2f} m", (50,150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,255,200), 2)

        cv2.imwrite(f"{app.config['PROCESSED_FOLDER']}/{frame_number}.jpg", base)

def compute_metrics(start_frame, end_frame):
    if not accumulated_trajectory or len(accumulated_trajectory) <6:
        return {'speed': 0.0, 'swing': 0.0, 'turn': 0.0, 'bounce': 0.0}

    points = np.array(accumulated_trajectory, dtype=np.float32)
    impact_idx = np.argmax(points[:,1])
    before = points[:impact_idx+1]
    after = points[impact_idx:]

    total_distance_m = 20.12
    total_time_s = max((end_frame - start_frame)/60, 1e-5)
    speed = (total_distance_m / total_time_s)*3.6

    swing_deg = 0.0
    turn_deg = 0.0
    bounce_height_m = 0.0

    if len(before) >=3:
        x_start, y_start = before[0]
        x_end, y_end = before[-1]
        slope = (x_end - x_start) / (y_end-y_start) if (y_end-y_start) != 0 else 0
        max_dev_px = max(abs(x - (x_start + slope*(y - y_start))) for (x,y) in before)
        deviation_m = max_dev_px / pixels_per_meter
        vertical_m = (y_end - y_start) / pixels_per_meter
        if vertical_m > 0:
            swing_deg = np.degrees(np.arctan(deviation_m / vertical_m))
            swing_deg = min(max(swing_deg, 0), 1.5)

    if len(after) >= 2:
        ya, xa = after[:,1], after[:,0]
        if len(set(ya)) > 1 and len(before) >= 2:
            ma, _ = np.polyfit(ya, xa, 1)
            mb, _ = np.polyfit(before[:,1], before[:,0], 1)
            turn_deg_raw = abs(np.degrees(np.arctan((ma - mb) / (1 + ma * mb))))
            turn_deg = turn_deg_raw if 2.5 < turn_deg_raw < 5.0 else 0.0

        peak_y = np.min(after[:,1])
        bounce_px = after[0][1] - peak_y
        bounce_height_m = bounce_px / pixels_per_meter if bounce_px > 0 else 0.0

    return {'speed': float(speed), 'swing': float(swing_deg), 'turn': float(turn_deg), 'bounce': float(bounce_height_m)}

@app.route('/run_analysis', methods=['POST'])
def run_analysis():
    global roi_coords
    data = request.json
    try:
        start_frame = int(data['start_frame'])
        end_frame = int(data['end_frame'])
        if roi_coords is None:
            return jsonify({'success': False, 'message': 'ROI not set'}), 400
        if start_frame > end_frame or start_frame < 0 or end_frame >= frame_count:
            return jsonify({'success': False, 'message': 'Invalid frame range'}), 400

        run_analysis_internal(start_frame, end_frame)
        metrics = compute_metrics(start_frame, end_frame)

        processed_files = [f for f in os.listdir(app.config['PROCESSED_FOLDER']) if f.endswith('.jpg')]
        processed_frame_count = len(processed_files)

        return jsonify({'success': True, 'metrics': metrics, 'processed_frame_count': processed_frame_count})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 400

# if __name__ == '__main__':
#     app.run(port=8072, debug=True)
