from flask import Flask, render_template_string, request, jsonify, send_from_directory, send_file
import os
import cv2
import numpy as np
import re
from werkzeug.utils import secure_filename
from scipy.signal import savgol_filter
import subprocess
import requests
import urllib.request
from urllib.parse import unquote
import ssl
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import io
import base64

index_template = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Ball Tracker</title>
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
<header>Ball Tracker</header>

<div style="display: flex; justify-content: center; gap: 20px; margin: 10px 0;">
    <button id="navBallTrackerBtn" style="padding: 10px 20px; border-radius: 6px; border: none; font-weight: 600; font-size: 1rem; background: #015151; color: #80fff7; cursor: pointer;">
        Ball Tracker
    </button>
    <button id="navTeamAnalysisBtn" style="padding: 10px 20px; border-radius: 6px; border: none; font-weight: 600; font-size: 1rem; background: #015151; color: #80fff7; cursor: pointer;">
        Team Analysis
    </button>
</div>

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

<section id="videoSection" style="display: none;">
  <h3>Video Playback</h3>

  <!-- ✅ Injected video URL via Flask -->
  <video id="videoPlayer" controls style="max-width: 100%; max-height: 400px;">
    <source src="{{VIDEO_URL}}" type="video/mp4" />
    Your browser does not support the video tag.
  </video>

  <div style="display: flex; justify-content: space-between; width: 100%; margin-top: 15px; flex-wrap: wrap;">
    <!-- Left: Frame Info + Buttons -->
    <div style="display: flex; flex-direction: column; align-items: flex-start; gap: 10px; margin-left: 120px;">
      <div id="frameInfo" style="font-size: 1.1rem;">
        Current Frame: <span id="currentFrame">0</span>
      </div>
      <button id="setStartFrameBtn" disabled>Set Start Frame</button>
      <button id="setEndFrameBtn" disabled>Set End Frame</button>
    </div>

    <!-- Right: Snickometer Canvas -->
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

<section id="teamAnalysisSection" style="display: none; background: #2a2a2a; margin: 10px; padding: 15px 20px; border-radius: 10px; box-shadow: 0 0 15px #00555588; width: 90%; max-width: 1400px;">
    <h3 style="text-align: center; color: #00d1b2;">Team Analysis - Areawise Wagon Wheel</h3>
    
    <div style="display: flex; justify-content: space-around; margin: 20px 0; gap: 40px;">
        <!-- Our Team Radar -->
        <div style="flex: 1; text-align: center;">
            <h4 style="color: #00d1b2;">Our Team</h4>
            <img id="ourTeamRadar" src="" alt="Our Team Radar" style="max-width: 100%; height: auto; background: transparent;" />
            
            <!-- Filters for Our Team -->
            <div style="margin-top: 15px; text-align: left; padding: 10px; background: #1c1c1c; border-radius: 8px;">
                <div style="margin-bottom: 10px;">
                    <strong style="color: #00d1b2;">Filter by Runs:</strong>
                    <div style="margin-left: 10px; margin-top: 5px;">
                        <label style="margin-right: 15px; cursor: pointer;">
                            <input type="checkbox" class="ourTeamRunFilter" value="1" checked /> 1s
                        </label>
                        <label style="margin-right: 15px; cursor: pointer;">
                            <input type="checkbox" class="ourTeamRunFilter" value="2" checked /> 2s
                        </label>
                        <label style="margin-right: 15px; cursor: pointer;">
                            <input type="checkbox" class="ourTeamRunFilter" value="3" checked /> 3s
                        </label>
                        <label style="margin-right: 15px; cursor: pointer;">
                            <input type="checkbox" class="ourTeamRunFilter" value="4" checked /> 4s
                        </label>
                        <label style="margin-right: 15px; cursor: pointer;">
                            <input type="checkbox" class="ourTeamRunFilter" value="6" checked /> 6s
                        </label>
                    </div>
                </div>
                
                <div style="margin-bottom: 10px;">
                    <strong style="color: #00d1b2;">Filter by Bowler Type:</strong>
                    <div style="margin-left: 10px; margin-top: 5px;">
                        <label style="margin-right: 15px; cursor: pointer;">
                            <input type="checkbox" class="ourTeamBowlerFilter" value="Pace" checked /> Pace
                        </label>
                        <label style="margin-right: 15px; cursor: pointer;">
                            <input type="checkbox" class="ourTeamBowlerFilter" value="Spin" checked /> Spin
                        </label>
                        <label style="margin-right: 15px; cursor: pointer;">
                            <input type="checkbox" class="ourTeamBowlerFilter" value="Other" checked /> Other
                        </label>
                    </div>
                </div>
                
                <button id="applyOurTeamFilters" style="margin-top: 10px; padding: 8px 16px; border-radius: 6px; border: none; font-weight: 600; background: #015151; color: #80fff7; cursor: pointer;">Apply Filters</button>
            </div>
        </div>
        
        <!-- Opponent Team Radar -->
        <div style="flex: 1; text-align: center;">
            <h4 style="color: #00d1b2;">Opponent Team</h4>
            <img id="opponentTeamRadar" src="" alt="Opponent Team Radar" style="max-width: 100%; height: auto; background: transparent;" />
            
            <!-- Filters for Opponent Team -->
            <div style="margin-top: 15px; text-align: left; padding: 10px; background: #1c1c1c; border-radius: 8px;">
                <div style="margin-bottom: 10px;">
                    <strong style="color: #00d1b2;">Filter by Runs:</strong>
                    <div style="margin-left: 10px; margin-top: 5px;">
                        <label style="margin-right: 15px; cursor: pointer;">
                            <input type="checkbox" class="opponentTeamRunFilter" value="1" checked /> 1s
                        </label>
                        <label style="margin-right: 15px; cursor: pointer;">
                            <input type="checkbox" class="opponentTeamRunFilter" value="2" checked /> 2s
                        </label>
                        <label style="margin-right: 15px; cursor: pointer;">
                            <input type="checkbox" class="opponentTeamRunFilter" value="3" checked /> 3s
                        </label>
                        <label style="margin-right: 15px; cursor: pointer;">
                            <input type="checkbox" class="opponentTeamRunFilter" value="4" checked /> 4s
                        </label>
                        <label style="margin-right: 15px; cursor: pointer;">
                            <input type="checkbox" class="opponentTeamRunFilter" value="6" checked /> 6s
                        </label>
                    </div>
                </div>
                
                <div style="margin-bottom: 10px;">
                    <strong style="color: #00d1b2;">Filter by Bowler Type:</strong>
                    <div style="margin-left: 10px; margin-top: 5px;">
                        <label style="margin-right: 15px; cursor: pointer;">
                            <input type="checkbox" class="opponentTeamBowlerFilter" value="Pace" checked /> Pace
                        </label>
                        <label style="margin-right: 15px; cursor: pointer;">
                            <input type="checkbox" class="opponentTeamBowlerFilter" value="Spin" checked /> Spin
                        </label>
                        <label style="margin-right: 15px; cursor: pointer;">
                            <input type="checkbox" class="opponentTeamBowlerFilter" value="Other" checked /> Other
                        </label>
                    </div>
                </div>
                
                <button id="applyOpponentTeamFilters" style="margin-top: 10px; padding: 8px 16px; border-radius: 6px; border: none; font-weight: 600; background: #015151; color: #80fff7; cursor: pointer;">Apply Filters</button>
            </div>
        </div>
    </div>
</section>

<div id="message"></div>

<script>
const videoInput = document.getElementById('videoFile');
const uploadBtn = document.getElementById('uploadBtn');
const videoUrlInput = document.getElementById('videoUrl');
const urlUploadBtn = document.getElementById('urlUploadBtn');
const videoPlayer = document.getElementById('videoPlayer');
videoPlayer.onloadeddata = function () {
    fetch("/extract_assets", {
        method: 'POST'
    }).then(res => res.json())
      .then(data => {
        if (data.success) {
            messageDiv.textContent = "✅ Frames and audio extracted!";
            setStartFrameBtn.disabled = false;
            setEndFrameBtn.disabled = false;
            videoSection.style.display = 'flex';
        } else {
            messageDiv.textContent = '❌ Frame/audio extraction failed: ' + data.message;
        }
    });
};
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
  })
  .then(res => res.json())
  .then(data => {
    if (data.success) {
      // ✅ Use real frame count and fps from backend
      framesCount = data.frame_count;
      window.fps = data.fps || 60;  // fallback if missing
      window.frameCount = framesCount;

      messageDiv.textContent = `✅ Uploaded. FPS: ${window.fps}, Frames: ${framesCount}`;
      videoUploaded = true;

      const fileURL = URL.createObjectURL(videoInput.files[0]);
      videoPlayer.src = fileURL;
      videoPlayer.load();
      videoSection.style.display = 'flex';

      // ✅ Enable controls
      setStartFrameBtn.disabled = false;
      setEndFrameBtn.disabled = false;
      uploadBtn.disabled = true;
      videoInput.disabled = true;

      // ✅ Reset UI states
      roiSection.style.display = 'none';
      processedSection.style.display = 'none';
    } else {
      messageDiv.textContent = '❌ Error: ' + data.message;
      uploadBtn.disabled = false;
    }
  })
  .catch((error) => {
    console.warn('Upload failed silently:', error);
    messageDiv.textContent = '❌ Upload failed. Please try again.';
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
        messageDiv.textContent = "Video fetched and frames extracted: " + framesCount + " frames.";
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
      messageDiv.textContent = 'ROI set successfully. Running analysis...';
      runAnalysis();
    } else {
      messageDiv.textContent = 'Failed to set ROI: ' + data.message;
    }
  });
});

// Analysis and display functions with slider
function runAnalysis() {
  const start = typeof window.startFrame === 'number' ? window.startFrame : 0;
  const end = typeof window.endFrame === 'number' ? window.endFrame : 0;

  if (start === 0 && end === 0) {
    messageDiv.textContent = '❌ Start and End frames are both 0. Move the video and set them again.';
    return;
  }

  if (start > end) {
    messageDiv.textContent = '❌ Start frame must be before End frame.';
    return;
  }

  console.log("▶ Sending to analysis: start =", start, ", end =", end);

  fetch('/run_analysis', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      start_frame: start,
      end_frame: end
    })
  })
  .then(res => res.json())
  .then(data => {
    if (data.success) {
      messageDiv.textContent = '✅ Analysis complete. Playing processed video.';
      roiSection.style.display = 'none';
      processedSection.style.display = 'flex';
      rewindBtn.disabled = false;
      pauseBtn.disabled = false;
      playBtn.disabled = false;
      forwardBtn.disabled = false;
      processedStartFrame = 0;
      processedEndFrame = data.processed_frame_count ? data.processed_frame_count - 1 : end;
      displayMetrics(data.metrics);
      currentProcessedFrame = processedStartFrame;
      displayProcessedFrame(currentProcessedFrame);
      frameSlider.max = processedEndFrame;
      frameSlider.value = currentProcessedFrame;
    } else {
      messageDiv.textContent = '❌ Analysis failed: ' + data.message;
    }
  })
  .catch(() => {
    messageDiv.textContent = '❌ Analysis request failed.';
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
<script>
                                  
# window.onload = function() {
#     const urlParams = new URLSearchParams(window.location.search);
#     const bucketUrl = urlParams.get('url');

#     if (bucketUrl) {
#         messageDiv.textContent = "Fetching video from bucket... Please wait.";
        
#         fetch(`/play_video?url=${encodeURIComponent(bucketUrl)}`, { method: 'GET' })
#             .then(res => res.json())
#             .then(data => {
#                 if (data.success) {
#                     framesCount = data.frame_count;
#                     videoPlayer.src = data.video_url;
#                     videoPlayer.load();
#                     videoSection.style.display = 'flex';
#                     messageDiv.textContent = "Video loaded successfully.";
#                 } else {
#                     messageDiv.textContent = 'Error: ' + data.message;
#                 }
#             })
#             .catch(() => {
#                 messageDiv.textContent = 'Failed to fetch video from bucket. Please try again.';
#             });
#     }
# };

</script>

<script>
window.addEventListener('DOMContentLoaded', function () {
    const videoPlayer = document.getElementById("videoPlayer");
    const currentFrameSpan = document.getElementById("currentFrame");
    const messageDiv = document.getElementById("message");

    // ✅ Injected from Flask (replace {{FPS}} and {{VIDEO_URL}} at server side)
    const fps = parseFloat("{{FPS}}") || 60;
    const preloadUrl = "{{VIDEO_URL}}";

    if (preloadUrl) {
        videoPlayer.src = preloadUrl;
        videoPlayer.load();
        document.getElementById("videoSection").style.display = "flex";
        document.getElementById("setStartFrameBtn").disabled = false;
        document.getElementById("setEndFrameBtn").disabled = false;
        window.videoUploaded = true;
        messageDiv.textContent = "✅ Video loaded. Set start and end frames.";
    }

    // ✅ Track current frame during video play
    videoPlayer.addEventListener('timeupdate', () => {
        const frame = Math.floor(videoPlayer.currentTime * fps);
        currentFrameSpan.textContent = frame;
    });

    // ✅ Helper function to get current video frame
    function getCurrentVideoFrame() {
        if (!videoPlayer || isNaN(videoPlayer.currentTime)) return 0;
        return Math.floor(videoPlayer.currentTime * fps);
    }

    // ✅ Set Start Frame
    document.getElementById('setStartFrameBtn').addEventListener('click', () => {
        const frame = getCurrentVideoFrame();
        if (frame >= 0) {
            window.startFrame = frame;
            console.log("✅ Set Start Frame:", frame);
            messageDiv.textContent = `✅ Start frame set to ${frame}`;
        } else {
            messageDiv.textContent = "❌ Move video to a valid time before setting start frame.";
        }
    });

    // ✅ Set End Frame
    document.getElementById('setEndFrameBtn').addEventListener('click', () => {
        const frame = getCurrentVideoFrame();
        if (frame >= 0) {
            window.endFrame = frame;
            console.log("✅ Set End Frame:", frame);
            messageDiv.textContent = `✅ End frame set to ${frame}`;
        } else {
            messageDiv.textContent = "❌ Move video to a valid time before setting end frame.";
        }
    });
});
</script>

<script>
// Navigation between sections
const navBallTrackerBtn = document.getElementById('navBallTrackerBtn');
const navTeamAnalysisBtn = document.getElementById('navTeamAnalysisBtn');
const uploadSection = document.getElementById('uploadSection');
const urlSection = document.getElementById('urlSection');

navBallTrackerBtn.addEventListener('click', () => {
    // Show ball tracker sections
    uploadSection.style.display = 'block';
    urlSection.style.display = 'block';
    if (videoSection) videoSection.style.display = videoSection.dataset.visible === 'true' ? 'flex' : 'none';
    if (roiSection) roiSection.style.display = roiSection.dataset.visible === 'true' ? 'block' : 'none';
    if (processedSection) processedSection.style.display = processedSection.dataset.visible === 'true' ? 'flex' : 'none';
    
    // Hide team analysis
    teamAnalysisSection.style.display = 'none';
    
    // Update button styles
    navBallTrackerBtn.style.background = '#00e5ca';
    navBallTrackerBtn.style.color = '#003330';
    navTeamAnalysisBtn.style.background = '#015151';
    navTeamAnalysisBtn.style.color = '#80fff7';
});

navTeamAnalysisBtn.addEventListener('click', () => {
    // Hide ball tracker sections
    uploadSection.style.display = 'none';
    urlSection.style.display = 'none';
    if (videoSection) {
        videoSection.dataset.visible = videoSection.style.display !== 'none' ? 'true' : 'false';
        videoSection.style.display = 'none';
    }
    if (roiSection) {
        roiSection.dataset.visible = roiSection.style.display !== 'none' ? 'true' : 'false';
        roiSection.style.display = 'none';
    }
    if (processedSection) {
        processedSection.dataset.visible = processedSection.style.display !== 'none' ? 'true' : 'false';
        processedSection.style.display = 'none';
    }
    
    // Show team analysis and load charts
    teamAnalysisSection.style.display = 'block';
    loadTeamRadarCharts('our');
    loadTeamRadarCharts('opponent');
    
    // Update button styles
    navTeamAnalysisBtn.style.background = '#00e5ca';
    navTeamAnalysisBtn.style.color = '#003330';
    navBallTrackerBtn.style.background = '#015151';
    navBallTrackerBtn.style.color = '#80fff7';
});

// Team Analysis functionality
const teamAnalysisSection = document.getElementById('teamAnalysisSection');
const ourTeamRadar = document.getElementById('ourTeamRadar');
const opponentTeamRadar = document.getElementById('opponentTeamRadar');
const applyOurTeamFiltersBtn = document.getElementById('applyOurTeamFilters');
const applyOpponentTeamFiltersBtn = document.getElementById('applyOpponentTeamFilters');

// Function to get selected filters
function getSelectedFilters(filterClass) {
    const checkboxes = document.querySelectorAll(`.${filterClass}:checked`);
    return Array.from(checkboxes).map(cb => cb.value);
}

// Function to load radar charts
function loadTeamRadarCharts(teamType) {
    const runFilters = getSelectedFilters(`${teamType}TeamRunFilter`);
    const bowlerFilters = getSelectedFilters(`${teamType}TeamBowlerFilter`);
    
    fetch('/api/team_radar', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            team_type: teamType,
            run_filters: runFilters,
            bowler_filters: bowlerFilters
        })
    })
    .then(res => res.json())
    .then(data => {
        if (data.success) {
            if (teamType === 'our') {
                ourTeamRadar.src = data.radar_image;
            } else {
                opponentTeamRadar.src = data.radar_image;
            }
        } else {
            console.error('Failed to load radar:', data.message);
        }
    })
    .catch(err => {
        console.error('Error loading radar:', err);
    });
}

// Apply filters for our team
applyOurTeamFiltersBtn.addEventListener('click', () => {
    loadTeamRadarCharts('our');
});

// Apply filters for opponent team
applyOpponentTeamFiltersBtn.addEventListener('click', () => {
    loadTeamRadarCharts('opponent');
});
</script>

<script>
const fps = parseFloat("{{FPS}}");
const frameCount = parseInt("{{FRAME_COUNT}}");

videoPlayer.addEventListener('timeupdate', () => {
    const frame = Math.floor(videoPlayer.currentTime * fps);
    document.getElementById('currentFrame').textContent = frame;
});
</script>



</body>
</html>
"""


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

    # Ensure frame folder exists
    if not os.path.exists(app.config['FRAME_FOLDER']):
        os.makedirs(app.config['FRAME_FOLDER'])

    # Clear old frames
    for f in os.listdir(app.config['FRAME_FOLDER']):
        os.remove(os.path.join(app.config['FRAME_FOLDER'], f))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Could not open video: {video_path}")

    cnt = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        # ✅ No resizing, no cropping — preserve original frame
        output_path = os.path.join(app.config['FRAME_FOLDER'], f"{cnt}.png")
        success = cv2.imwrite(output_path, frame)
        if not success:
            raise Exception(f"Failed to write frame {cnt}")

        cnt += 1

    cap.release()
    frame_count = cnt
    return frame_count

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

def map_bowling_type_radar(skill):
    """Map bowling skill to simplified bowling type for radar filtering."""
    skill_str = str(skill).lower()
    if "spin" in skill_str or "off" in skill_str or "leg" in skill_str:
        return "Spin"
    elif "pace" in skill_str or "fast" in skill_str or "medium" in skill_str:
        return "Pace"
    else:
        return "Other"

def generate_session_radar_chart(
    ball_by_ball_df,
    day,
    inning,
    session,
    team_name="Team",
    bowler_type=None,
    run_filter=None
):
    """
    Radar-style session wagon wheel chart with optional run filtering.
    BIG SIZE VERSION (Option A: 600x600).
    """

    df = ball_by_ball_df.copy()

    # ------------------------------
    # NORMALISING run_filter INPUT
    # ------------------------------
    if run_filter is None or str(run_filter).lower() == "all":
        run_set = None
    else:
        if isinstance(run_filter, (list, tuple, set)):
            run_set = set(int(x) for x in run_filter)
        else:
            s = str(run_filter)
            if "," in s:
                run_set = set(int(x.strip()) for x in s.split(",") if x.strip())
            else:
                run_set = set([int(s.strip())])

    # ------------------------------
    # Extract Day / Session columns
    # ------------------------------
    if "Day" not in df.columns or "SessionNo" not in df.columns:
        wide_col = "scrM_IsWideBall" if "scrM_IsWideBall" in df.columns else None
        noball_col = "scrM_IsNoBall" if "scrM_IsNoBall" in df.columns else None
        is_wide = df[wide_col].fillna(0).astype(int) if wide_col else 0
        is_noball = df[noball_col].fillna(0).astype(int) if noball_col else 0
        df["__is_legal"] = 1 - (np.array(is_wide) | np.array(is_noball))

        sort_cols = [c for c in ["scrM_InningNo", "scrM_OverNo", "scrM_DelNo"] if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

        df["__legal_cum"] = df["__is_legal"].cumsum()
        legal_idx_0 = np.maximum(df["__legal_cum"] - 1, 0)
        session_index = (legal_idx_0 // (30 * 6)).astype(int)

        df["Day"] = (session_index // 3) + 1
        df["SessionNo"] = (session_index % 3) + 1

    day_col = "Day"
    session_col = "SessionNo"

    df = df[
        (df[day_col] == day) &
        (df["scrM_InningNo"] == inning) &
        (df[session_col] == session)
    ]

    # ------------------------------
    # RUN FILTER
    # ------------------------------
    if run_set:
        df = df[df["scrM_BatsmanRuns"].isin(run_set)]

    # ------------------------------
    # Bowler Type filter
    # ------------------------------
    if bowler_type:
        if "scrM_BowlerSkill" in df.columns:
            df["BowlingType"] = df["scrM_BowlerSkill"].apply(map_bowling_type_radar)
            df = df[df["BowlingType"] == bowler_type]

    # ------------------------------
    # No Data Chart
    # ------------------------------
    if df.empty:
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))  # bigger
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.spines['polar'].set_visible(False)
        ax.text(0.5, 0.5, "No Data", ha="center", va="center",
                transform=ax.transAxes, color="red", fontsize=24, fontweight="bold")
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=260, transparent=True)
        plt.close(fig)
        buf.seek(0)
        return f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"

    # ------------------------------
    # Sector breakdown
    # ------------------------------
    sectors = ["Mid Wicket", "Square Leg", "Fine Leg", "Third Man",
               "Point", "Covers", "Long Off", "Long On"]
    breakdown_data = [{"1s":0,"2s":0,"3s":0,"4s":0,"6s":0} for _ in sectors]
    sector_map = {name: i for i, name in enumerate(sectors)}

    for _, row in df.iterrows():
        sec = str(row.get("scrM_WagonArea_zName", ""))
        runs = int(row.get("scrM_BatsmanRuns", 0))
        if sec in sector_map and runs > 0:
            idx = sector_map[sec]
            if runs == 1: breakdown_data[idx]["1s"] += 1
            elif runs == 2: breakdown_data[idx]["2s"] += 1
            elif runs == 3: breakdown_data[idx]["3s"] += 1
            elif runs == 4: breakdown_data[idx]["4s"] += 1
            elif runs == 6: breakdown_data[idx]["6s"] += 1

    # ------------------------------
    # Start Plot  (BIGGER SIZE)
    # ------------------------------
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))  # bigger

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines['polar'].set_visible(False)

    scale = 0.9
    ax.set_aspect('equal')

    # ------------------------------
    # Drawing (BIGGER RIM + ELEMENTS)
    # ------------------------------
    rim_radius = 1.10 * scale
    rim_circle = plt.Circle((0, 0), rim_radius, transform=ax.transData._b,
                            color='black', linewidth=26, fill=False,   # thicker rim
                            zorder=5, clip_on=False)
    ax.add_artist(rim_circle)

    ax.add_artist(plt.Circle((0, 0), 1.0 * scale, transform=ax.transData._b,
                             color='#19a94b', zorder=0))
    ax.add_artist(plt.Circle((0, 0), 0.6 * scale, transform=ax.transData._b,
                             color='#4CAF50', zorder=1))
    ax.add_artist(plt.Rectangle((-0.08 * scale / 2, -0.33 * scale / 2),
                                0.08 * scale, 0.33 * scale,
                                transform=ax.transData._b, color='burlywood', zorder=2))

    for angle in np.linspace(0, 2*np.pi, 9):
        ax.plot([angle, angle], [0, 1.0 * scale],
                color='white', linewidth=3, zorder=3)

    # ------------------------------
    # Sector totals + highlight
    # ------------------------------
    sector_runs = [(bd["1s"] + bd["2s"]*2 + bd["3s"]*3 +
                    bd["4s"]*4 + bd["6s"]*6) for bd in breakdown_data]

    total_runs = sum(sector_runs)

    if total_runs > 0:
        max_idx = sector_runs.index(max(sector_runs))
        sector_angles_deg = [112.5, 67.5, 22.5, 337.5,
                             292.5, 247.5, 202.5, 157.5]
        ax.bar(np.deg2rad(sector_angles_deg[max_idx]), 1.0 * scale,
               width=np.radians(45), color='red', alpha=0.25, zorder=1)

    # ------------------------------
    # Fielding labels (BIGGER)
    # ------------------------------
    position_labels = [
        ("Mid Wicket", 112.5, -110, -0.02),
        ("Square Leg", 67.5, -70, -0.02),
        ("Fine Leg", 22.5, -25, 0.00),
        ("Third Man", 337.5, 20, -0.02),
        ("Point", 292.5, 70, -0.01),
        ("Covers", 247.5, 110, -0.01),
        ("Long Off", 202.5, 155, -0.02),
        ("Long On", 157.5, 200, -0.02)
    ]

    for text, angle_deg, rotation_deg, dist_offset in position_labels:
        rad = np.deg2rad(angle_deg)
        ax.text(rad, rim_radius + dist_offset, text,
                color='white', fontsize=16, fontweight='bold',   # bigger labels
                ha='center', va='center', rotation=rotation_deg,
                rotation_mode='anchor', zorder=6)

    # ------------------------------
    # Runs + Percentage text (BIGGER)
    # ------------------------------
    box_positions = [
        (103.5, 0, 0.70), (67.5, 0, 0.70),
        (22.5, 0, 0.80), (337.5, 0, 0.80),
        (295.5, 0, 0.75), (250.5, 0, 0.70),
        (204.5, 1, 0.59), (155.5, 1, 0.59)
    ]

    for i, (angle_deg, rot, dist) in enumerate(box_positions):
        rad = np.deg2rad(angle_deg)
        r = dist * scale
        runs = sector_runs[i]
        pct = (runs / total_runs * 100) if total_runs > 0 else 0

        ax.text(rad, r,
                f"{runs}\n({pct:.1f}%)",
                color='white', fontsize=19, fontweight='bold',   # bigger text
                ha='center', va='center',
                rotation=0,
                linespacing=1.15)

    # ------------------------------
    # EXPORT (BIG)
    # ------------------------------
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=260, transparent=True)
    plt.close(fig)
    buf.seek(0)

    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"

def generate_team_wagon_radar(team_name, df, mode="batting", stance=None, size_inches=8, dpi=260):
    """
    Generate the wagon/radar image for a team aggregated across rows in df.
    - team_name: string (team to compute for)
    - df: ball-by-ball pandas DataFrame (should contain scrM_WagonArea_zName, scrM_BatsmanRuns,
          scrM_IsBoundry, scrM_IsSixer, scrM_tmMIdBattingName, scrM_tmMIdBowlingName)
    - mode: "batting" or "bowling"
      * batting => count runs scored by selected team (scrM_tmMIdBattingName == team_name)
      * bowling => count runs conceded by selected team (scrM_tmMIdBowlingName == team_name)
    - stance: None or "LHB" to mirror labels (keeps compatibility)
    Returns: "data:image/png;base64,..." string (PNG)
    """

    # expected areas (the radar geometry uses this order for RHB orientation)
    labels_expected = ["Mid Wicket","Square Leg","Fine Leg","Third Man","Point","Covers","Long Off","Long On"]

    # If df empty or None, return blank image with text
    if df is None or df.empty:
        # create blank placeholder image
        fig, ax = plt.subplots(figsize=(size_inches, size_inches))
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=18)
        ax.axis("off")
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=dpi, transparent=True)
        plt.close(fig)
        buf.seek(0)
        return "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")

    # select rows depending on mode
    # df already filtered by caller (team OR opponents)
    sel = df.copy()

    # normalize area column name
    area_col = "scrM_WagonArea_zName"
    runs_col = "scrM_BatsmanRuns"

    # If column not found, return placeholder
    if area_col not in sel.columns or runs_col not in sel.columns:
        # produce placeholder
        fig, ax = plt.subplots(figsize=(size_inches, size_inches))
        ax.text(0.5, 0.5, "Wagon area / runs columns missing", ha="center", va="center", fontsize=12)
        ax.axis("off")
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=dpi, transparent=True)
        plt.close(fig)
        buf.seek(0)
        return "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")

    # Clean strings
    sel[area_col] = sel[area_col].astype(str).str.strip()

    # Map canonical DB area names to expected labels.
    # The DB contains: Covers, Fine Leg, Long Off, Long On, Mid Wicket, Point, Square Leg, Third Man
    # We need to reorder into labels_expected.
    # Build a small aggregation keyed by canonical names (DB strings)
    agg = {}
    for lab in labels_expected:
        agg[lab] = {"runs": 0, "1s":0, "2s":0, "3s":0, "4s":0, "6s":0, "balls":0}

    # Build mapping from DB values to our expected labels (handle minor variants)
    db_to_expected = {
        "Covers": "Covers",
        "Fine Leg": "Fine Leg",
        "Long Off": "Long Off",
        "Long On": "Long On",
        "Mid Wicket": "Mid Wicket",
        "Point": "Point",
        "Square Leg": "Square Leg",
        "Third Man": "Third Man",
        # tolerant matches:
        "Cover": "Covers",
        "Fine-Leg": "Fine Leg",
        "Long-On": "Long On",
        "Long-Off": "Long Off",
        "MidWicket": "Mid Wicket",
        "ThirdMan": "Third Man"
    }

    # iterate rows and add to respective bin
    for _, row in sel.iterrows():
        a = str(row.get(area_col, "")).strip()
        if not a:
            continue
        mapped = db_to_expected.get(a, None)
        # try case-insensitive match fallback
        if mapped is None:
            for k,v in db_to_expected.items():
                if k.lower() == a.lower():
                    mapped = v
                    break
        if mapped is None:
            # area not recognized — skip
            continue

        runs_val = 0
        try:
            runs_val = int(row.get(runs_col, 0) or 0)
        except Exception:
            try:
                runs_val = int(float(row.get(runs_col, 0) or 0))
            except Exception:
                runs_val = 0

        agg[mapped]["runs"] += runs_val
        agg[mapped]["balls"] += 1

        # count 1s/2s/3s by runs equality
        if runs_val == 1:
            agg[mapped]["1s"] += 1
        elif runs_val == 2:
            agg[mapped]["2s"] += 1
        elif runs_val == 3:
            agg[mapped]["3s"] += 1

        # boundaries detection if flags present
        if "scrM_IsBoundry" in sel.columns:
            try:
                if int(row.get("scrM_IsBoundry") or 0) == 1:
                    agg[mapped]["4s"] += 1
            except Exception:
                pass
        else:
            # fallback: if runs_val == 4 and not flagged
            if runs_val == 4:
                agg[mapped]["4s"] += 1

        if "scrM_IsSixer" in sel.columns:
            try:
                if int(row.get("scrM_IsSixer") or 0) == 1:
                    agg[mapped]["6s"] += 1
            except Exception:
                pass
        else:
            if runs_val == 6:
                agg[mapped]["6s"] += 1

    # prepare lists in labels_expected order
    sector_runs = [agg[l]["runs"] for l in labels_expected]
    breakdown_data = [{"1s":agg[l]["1s"], "2s":agg[l]["2s"], "3s":agg[l]["3s"], "4s":agg[l]["4s"], "6s":agg[l]["6s"]} for l in labels_expected]
    total_sector_runs = sum(sector_runs)

    # --------------- Now plotting (wagon style) ----------------
    num_vars = 8
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(size_inches, size_inches), subplot_kw=dict(polar=True))
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])
    # hide spine
    try:
        ax.spines['polar'].set_visible(False)
    except Exception:
        pass

    # Scaling
    scale = 0.9
    ax.set_aspect('equal')

    # Black rim (radius relative)
    rim_radius = 1.10 * scale
    rim_circle = plt.Circle(
        (0, 0),
        rim_radius,
        transform=ax.transData._b,
        color='black',
        linewidth=26,
        fill=False,
        zorder=5,
        clip_on=False
    )
    ax.add_artist(rim_circle)

    # Ground circles (green)
    outer_circle = plt.Circle((0, 0), 1.0 * scale, transform=ax.transData._b, color='#19a94b', zorder=0)
    inner_circle = plt.Circle((0, 0), 0.6 * scale, transform=ax.transData._b, color='#4CAF50', zorder=1)
    ax.add_artist(outer_circle)
    ax.add_artist(inner_circle)

    # Pitch rectangle
    pitch_width = 0.08 * scale
    pitch_height = 0.33 * scale
    pitch_x = -pitch_width / 2
    pitch_y = -pitch_height / 2
    pitch = plt.Rectangle((pitch_x, pitch_y), pitch_width, pitch_height, transform=ax.transData._b, color='burlywood', zorder=2)
    ax.add_artist(pitch)

    # Sector lines (every 45°)
    for angle in np.linspace(0, 2 * np.pi, 9):
        ax.plot([angle, angle], [0, 1.0 * scale], color='white', linewidth=3, zorder=3)

    # RHB sector angles used in your previous design
    sector_angles_deg = [112.5, 67.5, 22.5, 337.5, 292.5, 247.5, 202.5, 157.5]

    # Highlight max sector
    if total_sector_runs > 0:
        max_idx = int(np.argmax(sector_runs))
        max_angle = np.deg2rad(sector_angles_deg[max_idx])
        ax.bar(max_angle, 1.0 * scale, width=np.radians(45), color='red', alpha=0.25, zorder=1)

    # Position labels (RHB)
    position_labels = [
        ("Mid Wicket", 112.5, -110, -0.02),
        ("Square Leg", 67.5, -70, -0.02),
        ("Fine Leg", 22.5, -25, 0.00),
        ("Third Man", 337.5, 20, -0.02),
        ("Point", 292.5, 70, -0.01),
        ("Covers", 247.5, 110, -0.01),
        ("Long Off", 202.5, 155, -0.02),
        ("Long On", 157.5, 200, -0.02)
    ]

    for text, angle_deg, rotation_deg, dist_offset in position_labels:
        rad = np.deg2rad(angle_deg)
        ax.text(
            rad,
            rim_radius + dist_offset,
            text,
            color='white',
            fontsize=16,
            fontweight='bold',
            ha='center',
            va='center',
            rotation=rotation_deg,
            rotation_mode='anchor',
            zorder=6
        )

    # Boxes with runs and % values (positions tuned to match style)
    box_positions = [
        (103.5, 0, 0.70),
        (67.5, 0, 0.70),
        (22.5, 0, 0.80),
        (337.5, 0, 0.80),
        (295.5, 0, 0.75),
        (250.5, 0, 0.70),
        (204.5, 1, 0.59),
        (155.5, 1, 0.59)
    ]

    for i, (angle_deg, rotation_deg, dist_offset) in enumerate(box_positions):
        rad = np.deg2rad(angle_deg)
        r = dist_offset * scale
        runs = sector_runs[i]
        percentage = (runs / total_sector_runs * 100) if total_sector_runs > 0 else 0
        label_text = f"{runs}\n({percentage:.1f}%)"
        ax.text(
            rad, r,
            label_text,
            color='white',
            fontsize=19,
            fontweight='bold',
            ha='center',
            va='center',
            rotation=0,
            linespacing=1.15
        )

    # Breakdown small text under boxes
    detail_positions = [
        (116.5, 0, 0.68),
        (78.5, 0, 0.68),
        (29.5, 0, 0.70),
        (330.5, 0, 0.70),
        (283.5, 0, 0.68),
        (239.5, 0, 0.72),
        (200.5, 1, 0.72),
        (163.5, 1, 0.70)
    ]

    for i, (angle_deg, rotation_deg, dist_offset) in enumerate(detail_positions):
        rad = np.deg2rad(angle_deg)
        r = dist_offset * scale
        bd = breakdown_data[i]
        breakdown_text = f"1s:{bd['1s']}  2s:{bd['2s']}\n4s:{bd['4s']}  6s:{bd['6s']}"
        ax.text(
            rad, r,
            breakdown_text,
            color='white',
            fontsize=10,
            ha='center',
            va='center',
            rotation=rotation_deg,
            rotation_mode='anchor',
            zorder=11
        )

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight', dpi=dpi, transparent=True)
    plt.close(fig)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

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
  return render_template_string(index_template.replace("{{VIDEO_URL}}", ""))

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
    
from urllib.parse import urlparse, quote, urlunparse
import urllib.request, ssl
from werkzeug.utils import secure_filename

@app.route('/play_video')
def play_video():
    bucket_url = request.args.get('url')
    if not bucket_url:
        return "Missing video URL", 400

    try:
        # ✅ Re-encode URL safely
        parsed = urlparse(bucket_url)
        encoded_path = quote(parsed.path)
        clean_url = urlunparse((parsed.scheme, parsed.netloc, encoded_path, '', '', ''))

        # ✅ Save filename securely
        filename = secure_filename(os.path.basename(parsed.path))
        local_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # ✅ Optionally skip SSL certs (development only)
        ssl._create_default_https_context = ssl._create_unverified_context

        # ✅ Download video (quick)
        urllib.request.urlretrieve(clean_url, local_path)

        # ✅ Set state (but skip extract_frames!)
        global video_path, roi_coords, frame_map, trajectory, accumulated_trajectory
        video_path = local_path
        roi_coords = None
        frame_map = {}
        trajectory = []
        accumulated_trajectory = []

        # ✅ Read metadata only
        cap = cv2.VideoCapture(local_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # ✅ Inject metadata into template
        rendered_html = render_template_string(
            index_template.replace("{{VIDEO_URL}}", f"/uploads/{filename}")
                          .replace("{{FPS}}", str(fps))
                          .replace("{{FRAME_COUNT}}", str(total_frames))
        )
        return rendered_html

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error loading video: {str(e)}", 500
    
@app.route('/extract_assets', methods=['POST'])
def extract_assets():
    try:
        # Clear folders first
        for folder in [app.config['FRAME_FOLDER'], app.config['AUDIO_FOLDER'], app.config['PROCESSED_FOLDER']]:
            for f in os.listdir(folder):
                os.remove(os.path.join(folder, f))

        # Extract frames + audio
        extract_frames(video_path)
        extract_audio(video_path, os.path.join(app.config['AUDIO_FOLDER'], 'extracted_audio.mp3'))

        return jsonify({'success': True})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)}), 500


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

        # 5. Confirm it saved
        if not os.path.exists(video_path):
            return jsonify({'success': False, 'message': f"File not saved at {video_path}"}), 500

        print(f"📥 Uploaded video saved to: {video_path}")

        # 6. Check if ffmpeg is installed
        try:
            subprocess.run(['ffmpeg', '-version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except FileNotFoundError:
            return jsonify({'success': False, 'message': 'ffmpeg not found. Please install FFmpeg and add to system PATH'}), 500

        # ✅ 7. Clear all old data
        for folder in [app.config['FRAME_FOLDER'], app.config['PROCESSED_FOLDER'], app.config['AUDIO_FOLDER']]:
            for f in os.listdir(folder):
                os.remove(os.path.join(folder, f))

        # ✅ 8. Extract frames and audio
        extract_frames(video_path)
        extract_audio(video_path, os.path.join(app.config['AUDIO_FOLDER'], 'extracted_audio.mp3'))

        # ✅ 9. Use OpenCV to get FPS and frame count
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        frame_count = total_frames  # update global

        print(f"📸 Extracted {total_frames} frames at {fps:.2f} FPS")

        # ✅ 10. Return data to frontend
        return jsonify({
            'success': True,
            'frame_count': total_frames,
            'fps': fps
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Upload processing failed: {str(e)}'}), 500

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
    print(f"🧪 Debug: start_frame={start_frame}, end_frame={end_frame}")
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
    speed = ((total_distance_m / total_time_s) * 3.6) - 10

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
    global roi_coords, frame_count
    data = request.json
    print("📥 Received for analysis:", data)

    try:
        start_frame = int(data['start_frame'])
        end_frame = int(data['end_frame'])

        if roi_coords is None:
            return jsonify({'success': False, 'message': 'ROI not set'}), 400

        # 🛠️ Ensure frames are present
        if not os.listdir(app.config['FRAME_FOLDER']):
            print("⚠️ No frames found, extracting...")
            extract_frames(video_path)

        # 🔄 Recalculate frame_count
        frame_files = sorted([f for f in os.listdir(app.config['FRAME_FOLDER']) if f.endswith('.png')])
        frame_count = len(frame_files)

        if start_frame > end_frame or start_frame < 0 or end_frame >= frame_count:
            return jsonify({'success': False, 'message': 'Invalid frame range'}), 400

        run_analysis_internal(start_frame, end_frame)
        metrics = compute_metrics(start_frame, end_frame)

        processed_files = [f for f in os.listdir(app.config['PROCESSED_FOLDER']) if f.endswith('.jpg')]
        processed_frame_count = len(processed_files)

        return jsonify({'success': True, 'metrics': metrics, 'processed_frame_count': processed_frame_count})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)}), 400


@app.route('/api/team_radar', methods=['POST'])
def team_radar():
    """
    API endpoint to generate team wagon radar charts with filters.
    Expects JSON: {team_type: 'our'|'opponent', run_filters: [1,2,3,4,6], bowler_filters: ['Pace','Spin','Other']}
    """
    try:
        data = request.json
        team_type = data.get('team_type', 'our')
        run_filters = data.get('run_filters', ['1', '2', '3', '4', '6'])
        bowler_filters = data.get('bowler_filters', ['Pace', 'Spin', 'Other'])
        
        # Convert run_filters to integers
        run_filters = [int(r) for r in run_filters]
        
        # Create sample data for demonstration
        # In a real application, this would fetch from database
        sample_data = create_sample_ball_by_ball_data(team_type, run_filters, bowler_filters)
        
        # Generate radar chart
        if sample_data is None or sample_data.empty:
            # Return empty chart
            radar_image = generate_team_wagon_radar(
                team_name=f"{team_type.capitalize()} Team",
                df=pd.DataFrame(),
                mode="batting",
                size_inches=8,
                dpi=260
            )
        else:
            radar_image = generate_team_wagon_radar(
                team_name=f"{team_type.capitalize()} Team",
                df=sample_data,
                mode="batting",
                size_inches=8,
                dpi=260
            )
        
        return jsonify({
            'success': True,
            'radar_image': radar_image
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': str(e)
        }), 400


def create_sample_ball_by_ball_data(team_type, run_filters, bowler_filters):
    """
    Create sample ball-by-ball data for demonstration.
    In a real application, this would query from a database with filters applied.
    """
    # Sample data structure
    areas = ["Mid Wicket", "Square Leg", "Fine Leg", "Third Man", "Point", "Covers", "Long Off", "Long On"]
    
    # Generate random sample data with filters applied
    data_rows = []
    np.random.seed(42 if team_type == 'our' else 84)  # Different seed for opponent
    
    for _ in range(100):  # Generate 100 sample balls
        area = np.random.choice(areas)
        runs = int(np.random.choice(run_filters)) if run_filters else 0
        
        # Only include if runs match filter
        if runs in run_filters:
            row = {
                'scrM_WagonArea_zName': area,
                'scrM_BatsmanRuns': runs,
                'scrM_IsBoundry': 1 if runs == 4 else 0,
                'scrM_IsSixer': 1 if runs == 6 else 0,
                'scrM_tmMIdBattingName': f"{team_type.capitalize()} Team",
                'scrM_tmMIdBowlingName': "Opponent Team" if team_type == 'our' else "Our Team",
                'scrM_BowlerSkill': np.random.choice(['Pace', 'Spin', 'Other'])
            }
            
            # Apply bowler filter
            if row['scrM_BowlerSkill'] in bowler_filters:
                data_rows.append(row)
    
    if not data_rows:
        return pd.DataFrame()
    
    return pd.DataFrame(data_rows)


# if __name__ == '__main__':
#     app.run(port=8072, debug=True)
