#!/usr/bin/env python3
"""
WebRTC receiver that processes video frames with hot dog detection
Receives webcam stream, runs YOLO detection, and sends processed video back
"""

import asyncio
import websockets
import json
import cv2
import numpy as np
import logging
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate, VideoStreamTrack
from aiortc.contrib.media import MediaBlackhole
from av import VideoFrame
import coremltools as ct
from pathlib import Path
from PIL import Image
import time
import subprocess
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
import os
import shutil

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HotDogDetectionTrack(VideoStreamTrack):
    """
    Video track that processes frames with hot dog detection and optional RTMP streaming
    """
    def __init__(self, source_track, model_path=None, rtmp_url=None, local_abr=False, inference_interval=1):
        super().__init__()
        self.source_track = source_track
        self.model = None
        self.frame_count = 0
        self.detection_count = 0
        self.rtmp_url = rtmp_url
        self.local_abr = local_abr
        self.inference_interval = inference_interval
        self.ffmpeg_processes = []  # For multiple ABR streams
        self.last_inference_time = 0
        self.avg_inference_time = 0
        self.inference_times = []
        self.cached_detections = []  # Store last inference result
        
        # Load CoreML model
        if model_path and Path(model_path).exists():
            try:
                self.model = ct.models.MLModel(str(model_path))
                logger.info(f"Loaded model: {model_path}")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
        else:
            logger.warning("No model provided or model not found")
        
        # Setup streaming
        if self.local_abr:
            self.setup_local_abr()
        elif self.rtmp_url:
            self.setup_rtmp_stream()
    
    def setup_rtmp_stream(self):
        """Setup FFmpeg process for RTMP streaming"""
        try:
            # FFmpeg command for RTMP streaming with hardware encoding on macOS
            ffmpeg_cmd = [
                'ffmpeg',
                '-y',  # Overwrite output
                '-f', 'rawvideo',  # Input format
                '-vcodec', 'rawvideo',
                '-pix_fmt', 'bgr24',  # OpenCV uses BGR
                '-s', '1280x720',  # Input size (adjust based on your stream)
                '-r', '30',  # Frame rate
                '-i', '-',  # Input from stdin
                '-c:v', 'h264_videotoolbox',  # Hardware encoder on macOS
                '-b:v', '2M',  # Bitrate
                '-preset', 'fast',
                '-g', '60',  # GOP size
                '-f', 'flv',  # Output format for RTMP
                self.rtmp_url
            ]
            
            self.ffmpeg_process = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            logger.info(f"Started RTMP stream to: {self.rtmp_url}")
            
        except Exception as e:
            logger.error(f"Failed to start RTMP stream: {e}")
            self.ffmpeg_process = None
    
    def stream_to_rtmp(self, frame):
        """Send frame to RTMP stream"""
        if self.ffmpeg_process and self.ffmpeg_process.poll() is None:
            try:
                # Resize frame for streaming (adjust as needed)
                frame_resized = cv2.resize(frame, (1280, 720))
                
                # Write frame to FFmpeg stdin
                self.ffmpeg_process.stdin.write(frame_resized.tobytes())
                self.ffmpeg_process.stdin.flush()
                
            except Exception as e:
                logger.error(f"Error streaming frame to RTMP: {e}")
                # Try to restart FFmpeg process
                self.cleanup_rtmp()
                self.setup_rtmp_stream()
    
    def cleanup_rtmp(self):
        """Cleanup RTMP streaming process"""
        if self.ffmpeg_process:
            try:
                self.ffmpeg_process.stdin.close()
                self.ffmpeg_process.terminate()
                self.ffmpeg_process.wait(timeout=5)
            except:
                self.ffmpeg_process.kill()
            self.ffmpeg_process = None
            logger.info("RTMP stream stopped")
    
    def setup_local_abr(self):
        """Setup multiple FFmpeg processes for local ABR streaming"""
        try:
            # Use absolute path to avoid nesting issues
            hls_dir = Path("hls_output").resolve()
            hls_dir.mkdir(exist_ok=True)
            
            # ABR stream configurations
            streams = [
                {"name": "1080p", "size": "1920x1080", "bitrate": "5M", "fps": "30"},
                {"name": "720p", "size": "1280x720", "bitrate": "3M", "fps": "30"},
                {"name": "480p", "size": "854x480", "bitrate": "1.5M", "fps": "30"},
                {"name": "360p", "size": "640x360", "bitrate": "800k", "fps": "30"}
            ]
            
            for stream in streams:
                # FFmpeg command for each quality
                ffmpeg_cmd = [
                    'ffmpeg',
                    '-y',  # Overwrite output
                    '-f', 'rawvideo',
                    '-vcodec', 'rawvideo',
                    '-pix_fmt', 'bgr24',
                    '-s', '1280x720',  # Input size (will be scaled)
                    '-r', '30',
                    '-i', '-',  # Input from stdin
                    '-c:v', 'h264_videotoolbox',  # M4 hardware encoder
                    '-b:v', stream["bitrate"],
                    '-s', stream["size"],
                    '-r', stream["fps"],
                    '-preset', 'ultrafast',  # Fastest encoding preset
                    '-tune', 'zerolatency',  # Zero latency tuning
                    '-g', '15',  # Smaller GOP size for faster seeking
                    '-f', 'hls',
                    '-hls_time', '0.5',  # 0.5-second segments for ultra-low latency
                    '-hls_list_size', '3',  # Keep only 3 segments (1.5 seconds total)
                    '-hls_flags', 'delete_segments+independent_segments',  # Auto-cleanup + independent segments
                    '-hls_segment_filename', str(hls_dir / f"{stream['name']}_%03d.ts"),
                    str(hls_dir / f"{stream['name']}.m3u8")
                ]
                
                process = subprocess.Popen(
                    ffmpeg_cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                self.ffmpeg_processes.append({
                    'name': stream['name'],
                    'process': process,
                    'config': stream
                })
                
                logger.info(f"Started ABR stream: {stream['name']} @ {stream['bitrate']}")
            
            # Create master playlist
            self.create_master_playlist(hls_dir, streams)
            logger.info(f"Local ABR streaming setup complete. Serving from: {hls_dir}")
            
        except Exception as e:
            logger.error(f"Failed to setup local ABR: {e}")
            self.ffmpeg_processes = []
    
    def create_master_playlist(self, hls_dir, streams):
        """Create HLS master playlist for ABR"""
        master_content = "#EXTM3U\n#EXT-X-VERSION:3\n\n"
        
        # Stream info mapping
        stream_info = {
            "1080p": {"bandwidth": 5000000, "resolution": "1920x1080"},
            "720p": {"bandwidth": 3000000, "resolution": "1280x720"},
            "480p": {"bandwidth": 1500000, "resolution": "854x480"},
            "360p": {"bandwidth": 800000, "resolution": "640x360"}
        }
        
        for stream in streams:
            name = stream["name"]
            info = stream_info[name]
            master_content += f'#EXT-X-STREAM-INF:BANDWIDTH={info["bandwidth"]},RESOLUTION={info["resolution"]}\n'
            master_content += f'{name}.m3u8\n'
        
        master_file = hls_dir / "master.m3u8"
        with open(master_file, 'w') as f:
            f.write(master_content)
        
        logger.info(f"Created master playlist: {master_file}")
    
    def stream_to_abr(self, frame):
        """Send frame to all ABR streams"""
        if not self.ffmpeg_processes:
            return
            
        # Resize frame for consistent input
        frame_720p = cv2.resize(frame, (1280, 720))
        frame_bytes = frame_720p.tobytes()
        
        for stream in self.ffmpeg_processes:
            if stream['process'].poll() is None:  # Process still running
                try:
                    stream['process'].stdin.write(frame_bytes)
                    stream['process'].stdin.flush()
                except Exception as e:
                    logger.warning(f"Error streaming to {stream['name']}: {e}")
    
    def cleanup_abr(self):
        """Cleanup ABR streaming processes"""
        for stream in self.ffmpeg_processes:
            try:
                stream['process'].stdin.close()
                stream['process'].terminate()
                stream['process'].wait(timeout=5)
            except:
                stream['process'].kill()
        self.ffmpeg_processes = []
        logger.info("ABR streams stopped")
    
    def preprocess_frame(self, frame):
        """Convert video frame to format expected by YOLO model"""
        # Convert from BGR to RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to 416x416 (YOLO input size)
        img_resized = cv2.resize(img_rgb, (416, 416))
        
        # Convert to PIL Image
        pil_img = Image.fromarray(img_resized)
        
        return pil_img, frame.shape[:2]  # Return PIL image and original shape
    
    def postprocess_detections(self, prediction, original_shape):
        """Process YOLO output to get hot dog detections with NMS"""
        if not prediction or 'coordinates' not in prediction or 'confidence' not in prediction:
            return []
        
        detections = []
        coordinates = prediction['coordinates']
        confidence = prediction['confidence']
        
        if len(coordinates) == 0:
            return detections
        
        # Hot dog class index
        hotdog_class_idx = 52
        confidence_threshold = 0.1
        
        orig_h, orig_w = original_shape
        
        # Collect all hot dog detections first
        hotdog_boxes = []
        hotdog_confidences = []
        
        for i in range(coordinates.shape[0]):
            # Find predicted class
            predicted_class = np.argmax(confidence[i])
            class_confidence = confidence[i][predicted_class]
            
            # Check if it's a hot dog with sufficient confidence
            if predicted_class == hotdog_class_idx and class_confidence > confidence_threshold:
                # Get bounding box coordinates (normalized)
                x, y, w, h = coordinates[i]
                
                # Convert to pixel coordinates
                x1 = int((x - w/2) * orig_w)
                y1 = int((y - h/2) * orig_h)
                x2 = int((x + w/2) * orig_w)
                y2 = int((y + h/2) * orig_h)
                
                # Ensure bounds
                x1 = max(0, min(x1, orig_w-1))
                y1 = max(0, min(y1, orig_h-1))
                x2 = max(0, min(x2, orig_w-1))
                y2 = max(0, min(y2, orig_h-1))
                
                # Calculate area for size comparison
                area = (x2 - x1) * (y2 - y1)
                
                hotdog_boxes.append([x1, y1, x2, y2])
                hotdog_confidences.append(float(class_confidence))
        
        if not hotdog_boxes:
            return detections
        
        # Apply NMS using OpenCV
        boxes = np.array(hotdog_boxes, dtype=np.float32)
        confidences = np.array(hotdog_confidences, dtype=np.float32)
        
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), confidences.tolist(), confidence_threshold, 0.4)
        
        if len(indices) > 0:
            # Find the biggest box among NMS survivors
            max_area = 0
            biggest_detection = None
            
            for i in indices.flatten():
                x1, y1, x2, y2 = boxes[i].astype(int)
                area = (x2 - x1) * (y2 - y1)
                
                if area > max_area:
                    max_area = area
                    biggest_detection = {
                        'bbox': (x1, y1, x2, y2),
                        'confidence': float(confidences[i])
                    }
            
            if biggest_detection:
                detections.append(biggest_detection)
        
        return detections
    
    def draw_detections(self, frame, detections):
        """Blur hot dogs without drawing borders or labels"""
        processed_frame = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            
            # Expand blur region by 50%
            width = x2 - x1
            height = y2 - y1
            expand_w = int(width * 0.25)  # 25% on each side = 50% total
            expand_h = int(height * 0.25)
            
            # Calculate expanded coordinates with bounds checking
            frame_height, frame_width = processed_frame.shape[:2]
            expanded_x1 = max(0, x1 - expand_w)
            expanded_y1 = max(0, y1 - expand_h)
            expanded_x2 = min(frame_width, x2 + expand_w)
            expanded_y2 = min(frame_height, y2 + expand_h)
            
            # Blur the expanded region
            roi = processed_frame[expanded_y1:expanded_y2, expanded_x1:expanded_x2]
            if roi.size > 0:
                blurred_roi = cv2.GaussianBlur(roi, (51, 51), 0)
                processed_frame[expanded_y1:expanded_y2, expanded_x1:expanded_x2] = blurred_roi
        
        return processed_frame
    
    def draw_timestamp_barcode(self, frame):
        """Draw a QR code encoding the current UTC timestamp"""
        import time
        import qrcode
        import numpy as np
        
        try:
            # Get current UTC timestamp in milliseconds
            timestamp_ms = int(time.time() * 1000)
            
            # Create QR code
            qr = qrcode.QRCode(
                version=1,  # Small QR code
                error_correction=qrcode.constants.ERROR_CORRECT_L,
                box_size=3,  # Size of each box in pixels
                border=1,   # Border size
            )
            qr.add_data(str(timestamp_ms))
            qr.make(fit=True)
            
            # Create QR code image
            qr_img = qr.make_image(fill_color="black", back_color="white")
            qr_array = np.array(qr_img.convert('RGB'))
            
            # Get dimensions
            frame_height, frame_width = frame.shape[:2]
            qr_height, qr_width = qr_array.shape[:2]
            
            # Position at bottom right corner with some padding
            start_x = frame_width - qr_width - 10
            start_y = frame_height - qr_height - 10
            
            # Ensure QR code fits in frame
            if start_x < 0 or start_y < 0:
                return frame  # Skip if frame too small
            
            # Overlay QR code on frame
            frame[start_y:start_y + qr_height, start_x:start_x + qr_width] = qr_array
            
            return frame
            
        except Exception as e:
            # If QR code generation fails, just return original frame
            logger.warning(f"Failed to generate QR code: {e}")
            return frame
    
    def draw_overlay_info(self, frame, detections):
        """Draw performance metrics and timestamp overlay"""
        from datetime import datetime
        
        # Get current time
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Calculate FPS estimate
        fps_estimate = 1000 / self.avg_inference_time if self.avg_inference_time > 0 else 0
    
        # Overlay background for better readability
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (500, 120), (0, 0, 0), -1)  # Black background
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)  # Semi-transparent
        
        # Text properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        text_color = (0, 255, 0)  # Green text
        line_height = 20
        
        y_offset = 30
        
        # Draw metrics
        texts = [
            f"Time: {current_time}",
            f"Frame: {self.frame_count}",
            f"Inference: {self.last_inference_time:.1f}ms (Avg: {self.avg_inference_time:.1f}ms)",
            f"Est. FPS: {fps_estimate:.1f}",
            f"Hot Dogs Detected: {len(detections)} (Total: {self.detection_count})"
        ]
        
        for i, text in enumerate(texts):
            y = y_offset + (i * line_height)
            cv2.putText(frame, text, (15, y), font, font_scale, text_color, font_thickness)
        
        return frame
    
    async def recv(self):
        """Receive frame from source, process it, and return processed frame"""
        # Get frame from source track
        frame = await self.source_track.recv()
        
        # Convert to numpy array
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1
        
        # Process with hot dog detection if model is available
        if self.model is not None:
            try:
                # Check if we should run inference on this frame
                should_run_inference = (self.frame_count % self.inference_interval) == 0
                
                if should_run_inference:
                    start_time = time.time()
                    
                    # Preprocess frame
                    pil_img, original_shape = self.preprocess_frame(img)
                    
                    # Run inference
                    inputs = {"image": pil_img}
                    
                    # Add confidence threshold if model supports it
                    try:
                        spec = self.model.get_spec()
                        for input_desc in spec.description.input:
                            if input_desc.name == "confidenceThreshold":
                                inputs["confidenceThreshold"] = 0.1
                            elif input_desc.name == "iouThreshold":
                                inputs["iouThreshold"] = 0.4
                    except:
                        pass
                    
                    prediction = self.model.predict(inputs)
                    
                    # Process detections and cache them
                    self.cached_detections = self.postprocess_detections(prediction, original_shape)
                    
                    # Track inference time
                    inference_time = (time.time() - start_time) * 1000/4
                    self.last_inference_time = inference_time
                    
                    # Update rolling average of inference times
                    self.inference_times.append(inference_time)
                    if len(self.inference_times) > 30:  # Keep last 30 measurements
                        self.inference_times.pop(0)
                    self.avg_inference_time = sum(self.inference_times) / len(self.inference_times)
                    
                    # Log performance
                    if len(self.cached_detections) > 0:
                        self.detection_count += len(self.cached_detections)
                        logger.info(f"Frame {self.frame_count}: Found {len(self.cached_detections)} hot dogs (inference)")
                    
                    if self.frame_count % 30 == 0:  # Log every 30 frames
                        logger.info(f"Frame {self.frame_count}: {inference_time:.2f}ms inference, "
                                  f"Avg: {self.avg_inference_time:.2f}ms, Interval: every {self.inference_interval} frames")
                else:
                    # Use cached detections from last inference
                    if self.frame_count % 30 == 0 and self.cached_detections:
                        logger.info(f"Frame {self.frame_count}: Using cached detection result")
                
                # Apply detections (either new or cached)
                detections = self.cached_detections
                if detections:
                    img = self.draw_detections(img, detections)
                
            except Exception as e:
                logger.error(f"Error in hot dog detection: {e}")
                # Continue with original frame if detection fails
                detections = []
        else:
            # No model loaded, just pass through with a message
            cv2.putText(img, "NO MODEL LOADED - PASS THROUGH", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            detections = []
        
        # Add timestamp barcode for latency measurement
        img = self.draw_timestamp_barcode(img)
        
        # Add overlay with metrics and timestamp
        img = self.draw_overlay_info(img, detections if 'detections' in locals() else [])
        
        # Stream to output
        if self.local_abr:
            self.stream_to_abr(img)
        elif self.rtmp_url:
            self.stream_to_rtmp(img)
        
        # Convert back to video frame
        new_frame = VideoFrame.from_ndarray(img, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        
        return new_frame
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        if self.local_abr:
            self.cleanup_abr()
        else:
            self.cleanup_rtmp()

class WebRTCReceiver:
    def __init__(self, model_path=None, rtmp_url=None, local_abr=False, http_port=8000, inference_interval=1):
        self.websocket = None
        self.peer_connection = None
        self.model_path = model_path
        self.rtmp_url = rtmp_url
        self.local_abr = local_abr
        self.http_port = http_port
        self.http_server = None
        self.inference_interval = inference_interval
        
        # Start HTTP server for local ABR if enabled
        if self.local_abr:
            self.start_http_server()
        
    def start_http_server(self):
        """Start HTTP server to serve HLS content and HTML files"""
        
        class CORSHTTPRequestHandler(SimpleHTTPRequestHandler):
            def end_headers(self):
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
                self.send_header('Access-Control-Allow-Headers', '*')
                self.send_header('Cache-Control', 'no-cache')
                super().end_headers()
            
            def do_OPTIONS(self):
                self.send_response(200)
                self.end_headers()
        
        def run_server():
            # Use absolute path to avoid nesting issues
            hls_dir = Path("hls_output").resolve()
            hls_dir.mkdir(exist_ok=True)
            
            # Copy HTML file to HLS directory if it exists  
            html_file = Path("webcam_client.html").resolve()
            if html_file.exists():
                shutil.copy2(html_file, hls_dir / "index.html")
                logger.info("Copied webcam_client.html as index.html")
            
            # Copy favicon if it exists
            favicon_file = Path("nothotdog.png").resolve()
            if favicon_file.exists():
                shutil.copy2(favicon_file, hls_dir / "nothotdog.png")
                logger.info("Copied nothotdog.png as favicon")
            else:
                # Create simple index.html for directory listing
                with open(hls_dir / "index.html", "w") as f:
                    f.write("""
                    <html><head><title>M4 ABR Streams</title></head>
                    <body>
                    <h1>M4 Local ABR Streaming</h1>
                    <p>Available streams:</p>
                    <ul>
                        <li><a href="master.m3u8">Master Playlist (ABR)</a></li>
                        <li><a href="1080p.m3u8">1080p Stream</a></li>
                        <li><a href="720p.m3u8">720p Stream</a></li>
                        <li><a href="480p.m3u8">480p Stream</a></li>
                        <li><a href="360p.m3u8">360p Stream</a></li>
                    </ul>
                    <p>Stream will appear once WebRTC connection is established.</p>
                    <p>HTML client not found. Make sure webcam_client.html exists in the project directory.</p>
                    </body></html>
                    """)
            
            # Serve from project root directory
            project_root = Path(".").resolve()
            os.chdir(project_root)
            
            server = HTTPServer(('localhost', self.http_port), CORSHTTPRequestHandler)
            logger.info(f"HTTP Server started at http://localhost:{self.http_port}/")
            logger.info(f"WebRTC Demo: http://localhost:{self.http_port}/")
            logger.info(f"Master playlist: http://localhost:{self.http_port}/hls_output/master.m3u8")
            server.serve_forever()
        
        # Run server in background thread
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
    async def connect_signaling(self, url="ws://localhost:8765"):
        """Connect to signaling server"""
        logger.info(f"Connecting to signaling server: {url}")
        
        self.websocket = await websockets.connect(url)
        
        # Register as Python client
        await self.websocket.send(json.dumps({
            "type": "register",
            "client_type": "python"
        }))
        
        # Wait for registration confirmation
        response = await self.websocket.recv()
        data = json.loads(response)
        
        if data.get("type") == "registered":
            logger.info(f"Registered as client: {data.get('client_id')}")
        else:
            raise Exception("Failed to register with signaling server")
    
    async def handle_signaling_message(self, message):
        """Handle signaling messages from WebSocket"""
        try:
            data = json.loads(message)
            message_type = data.get("type")
            
            if message_type == "offer":
                logger.info("Received offer from browser")
                await self.handle_offer(data)
                
            elif message_type == "ice-candidate":
                logger.info("Received ICE candidate from browser")
                try:
                    # Use aiortc's candidate parsing
                    from aiortc.sdp import candidate_from_sdp
                    candidate = candidate_from_sdp(data["candidate"])
                    candidate.sdpMLineIndex = data["sdpMLineIndex"]
                    candidate.sdpMid = data["sdpMid"]
                    await self.peer_connection.addIceCandidate(candidate)
                except Exception as ice_error:
                    logger.warning(f"Failed to add ICE candidate: {ice_error}")
                    # Continue without this candidate
                
        except Exception as e:
            logger.error(f"Error handling signaling message: {e}")
    
    async def handle_offer(self, offer_data):
        """Handle WebRTC offer from browser"""
        # Create peer connection
        self.peer_connection = RTCPeerConnection()
        
        # Set up event handlers
        @self.peer_connection.on("track")
        def on_track(track):
            logger.info(f"Received track: {track.kind}")
            
            if track.kind == "video":
                # Create hot dog detection track that processes the incoming video
                processed_track = HotDogDetectionTrack(track, self.model_path, self.rtmp_url, self.local_abr, self.inference_interval)
                
                # Add the processed track back to peer connection
                self.peer_connection.addTrack(processed_track)
                
                logger.info("Added hot dog detection track to peer connection")
        
        @self.peer_connection.on("icecandidate")
        async def on_icecandidate(candidate):
            if candidate:
                logger.info("Sending ICE candidate to browser")
                await self.websocket.send(json.dumps({
                    "type": "ice-candidate",
                    "candidate": candidate.candidate,
                    "sdpMLineIndex": candidate.sdpMLineIndex,
                    "sdpMid": candidate.sdpMid
                }))
        
        # Set remote description
        await self.peer_connection.setRemoteDescription(
            RTCSessionDescription(sdp=offer_data["sdp"], type="offer")
        )
        
        # Create answer
        answer = await self.peer_connection.createAnswer()
        await self.peer_connection.setLocalDescription(answer)
        
        # Send answer back
        await self.websocket.send(json.dumps({
            "type": "answer",
            "sdp": answer.sdp
        }))
        
        logger.info("Sent answer to browser")
    
    async def run(self):
        """Main loop to handle WebRTC connections"""
        try:
            # Connect to signaling server
            await self.connect_signaling()
            
            logger.info("WebRTC receiver ready. Waiting for browser connections...")
            
            # Handle signaling messages
            async for message in self.websocket:
                await self.handle_signaling_message(message)
                
        except websockets.exceptions.ConnectionClosed:
            logger.info("Connection to signaling server closed")
        except Exception as e:
            logger.error(f"Error in WebRTC receiver: {e}")
        finally:
            if self.peer_connection:
                await self.peer_connection.close()

async def main():
    """Start the WebRTC receiver"""
    import argparse
    
    parser = argparse.ArgumentParser(description="WebRTC Hot Dog Detection Receiver")
    parser.add_argument("--model", help="Path to CoreML model file")
    parser.add_argument("--rtmp-url", help="RTMP URL for streaming (e.g., rtmp://live.mux.com/live/YOUR_STREAM_KEY)")
    parser.add_argument("--local-abr", action="store_true", help="Enable local ABR streaming instead of RTMP")
    parser.add_argument("--http-port", type=int, default=8000, help="HTTP port for serving HLS (default: 8000)")
    parser.add_argument("--inference-interval", type=int, default=1, help="Run inference every N frames (default: 1, every frame)")
    parser.add_argument("--signaling-url", default="ws://localhost:8765", 
                       help="WebSocket URL for signaling server")
    
    args = parser.parse_args()
    
    # Find model if not specified (use absolute path)
    if not args.model:
        model_dir = Path("models").resolve()
        if model_dir.exists():
            models = list(model_dir.glob("*.mlmodel"))
            if models:
                args.model = str(models[0])  # Use first available model
                logger.info(f"Using model: {args.model}")
    else:
        # Convert relative path to absolute path
        args.model = str(Path(args.model).resolve())
    
    receiver = WebRTCReceiver(args.model, args.rtmp_url, args.local_abr, args.http_port, args.inference_interval)
    await receiver.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Receiver stopped by user")