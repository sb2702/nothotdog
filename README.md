**Here's a README for your Not Hot Dog demo:**

---

```markdown
# Not Hot Dog Demo - Real-time Object Detection with M4

Live WebRTC demo that detects hot dogs in webcam feed and censors them in real-time before streaming to RTMP.

## Architecture

```
Laptop Webcam (Browser) → WebRTC → M4 Mac Mini → YOLO Detection → Censor Hot Dogs → RTMP Output
```

## Prerequisites

### System Requirements
- M4 Mac Mini (or M1/M2/M3)
- macOS 14+ (Sonoma or later)
- Python 3.11+
- FFmpeg with VideoToolbox support
- RTMP server (nginx-rtmp, or stream to YouTube/Twitch)

### Install Dependencies

```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install FFmpeg
brew install ffmpeg

# Install Python dependencies
pip install --upgrade pip
pip install aiortc opencv-python ultralytics torch torchvision numpy aiohttp
```

## Project Structure

```
not-hotdog-demo/
├── README.md
├── signaling_server.py      # WebRTC signaling server
├── webrtc_receiver.py        # Main demo - receives WebRTC, detects hotdogs, outputs RTMP
├── webrtc_sender.html        # Browser client (send webcam)
└── requirements.txt          # Python dependencies
```

## Setup Steps

### 1. Create requirements.txt

```txt
aiortc==1.6.0
opencv-python==4.8.1.78
ultralytics==8.1.0
torch==2.1.0
torchvision==0.16.0
numpy==1.26.2
aiohttp==3.9.1
av==11.0.0
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Download YOLO Model

```bash
# The model will auto-download on first run, but you can pre-download:
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

### 4. Setup RTMP Server (Choose One)

#### Option A: Local nginx-rtmp (for testing)

```bash
# Install nginx with RTMP module
brew tap denji/nginx
brew install nginx-full --with-rtmp-module

# Start nginx
nginx
```

**RTMP URL:** `rtmp://localhost/live/test`

#### Option B: YouTube Live

1. Go to YouTube Studio → Create → Go Live
2. Copy Stream Key
3. **RTMP URL:** `rtmp://a.rtmp.youtube.com/live2/YOUR_STREAM_KEY`

#### Option C: Twitch

1. Go to Twitch Dashboard → Settings → Stream
2. Copy Stream Key
3. **RTMP URL:** `rtmp://live.twitch.tv/app/YOUR_STREAM_KEY`

## Running the Demo

### Terminal 1: Start Signaling Server

```bash
python signaling_server.py
```

**Output:** `Running on http://localhost:8080`

### Terminal 2: Start WebRTC Receiver + AI Processor

```bash
python webrtc_receiver.py --rtmp-url rtmp://localhost/live/test
```

**Options:**
- `--rtmp-url`: RTMP destination (required)
- `--host`: Signaling server host (default: localhost)
- `--port`: Signaling server port (default: 8080)
- `--confidence`: Detection confidence threshold (default: 0.5)

**Example with YouTube:**
```bash
python webrtc_receiver.py --rtmp-url rtmp://a.rtmp.youtube.com/live2/YOUR_KEY
```

### Terminal 3: Open Browser Client

```bash
open webrtc_sender.html
```

**Or manually:** Open `webrtc_sender.html` in Chrome/Safari

1. Click "Allow" for camera permissions
2. Click "Start Streaming"
3. Connection established!

### Terminal 4: Watch RTMP Output

```bash
# Watch with ffplay
ffplay rtmp://localhost/live/test

# Or with VLC
vlc rtmp://localhost/live/test
```

## Testing the Demo

1. **Start all terminals** as described above
2. **Point camera at hot dog** 🌭
3. **Watch it get censored** with blur/box in real-time
4. **Check stream** in ffplay/VLC or on YouTube/Twitch

## Troubleshooting

### Camera Not Working

```bash
# Check camera permissions
# System Settings → Privacy & Security → Camera → Allow Terminal/Chrome
```

### Connection Fails

```bash
# Check signaling server is running
curl http://localhost:8080

# Check firewall isn't blocking port 8080
```

### RTMP Connection Fails

```bash
# Test RTMP server is running
ffmpeg -re -f lavfi -i testsrc=size=1280x720:rate=30 \
  -c:v h264_videotoolbox -f flv rtmp://localhost/live/test
```

### Hot Dogs Not Detected

```bash
# Lower confidence threshold
python webrtc_receiver.py --rtmp-url rtmp://localhost/live/test --confidence 0.3

# Check YOLO model loaded correctly
# Should see: "Downloading yolov8n.pt..." on first run
```

### Frame Rate Issues

```bash
# Check Activity Monitor for CPU/GPU usage
# M4 should handle 1080p30 easily

# Try lower resolution in browser (edit webrtc_sender.html)
video: { width: 1280, height: 720 }  # Instead of 1920x1080
```

## Performance Monitoring

### Check FPS in Terminal 2

```
Processed 30 frames
Processed 60 frames
...
```

### Monitor GPU Usage

```bash
# In another terminal
sudo powermetrics --samplers gpu_power -i 1000
```

## Configuration

### Adjust Detection Parameters

Edit `webrtc_receiver.py`:

```python
# Change confidence threshold
CONFIDENCE_THRESHOLD = 0.5  # Lower = more detections

# Change blur strength
blur_kernel = (51, 51)  # Larger = more blur

# Change detection model
model = YOLO('yolov8s.pt')  # s=small, m=medium, l=large (slower but more accurate)
```

### Adjust Video Quality

Edit `webrtc_receiver.py` FFmpeg command:

```python
'-b:v', '5M',  # Bitrate (higher = better quality, more bandwidth)
'-preset', 'fast',  # Preset (fast/medium/slow)
```

## Architecture Details

### WebRTC Flow

1. Browser captures webcam via `getUserMedia()`
2. Browser creates WebRTC offer (SDP)
3. Signaling server exchanges offer/answer
4. Direct peer-to-peer connection established
5. Video frames flow: Browser → Python (aiortc)

### Processing Pipeline

```
aiortc receives frame
  ↓
Convert to numpy array
  ↓
YOLO detection (YOLOv8n on M4 GPU/Neural Engine)
  ↓
Filter for hot dog class (52)
  ↓
Draw blur/censor box
  ↓
Pipe to FFmpeg
  ↓
Encode with VideoToolbox (h264_videotoolbox)
  ↓
Stream to RTMP
```

### Why M4 is Good for This

- **Unified memory:** Decoded frames → YOLO → encoding, all in same memory
- **Neural Engine:** Accelerates YOLO inference
- **VideoToolbox:** Hardware video encode/decode
- **Zero-copy:** No CPU ↔ GPU transfers
- **Low latency:** <100ms end-to-end possible

## Next Steps

### For Demuxed Talk

1. **Benchmark:** Measure FPS, latency, GPU usage
2. **Compare:** Run same demo on NVIDIA GPU machine
3. **Show results:** M4 unified memory advantage for real-time AI + video

### Extensions

- Detect multiple objects (pizza, donuts, etc.)
- Add fun overlays instead of blur (emoji, text)
- Multi-stream: Process multiple webcams simultaneously
- Add AI upscaling to pipeline
- Background removal instead of object detection

## Files to Create

You'll need to create these three files:

1. `signaling_server.py` - HTTP server for WebRTC signaling
2. `webrtc_receiver.py` - Main demo logic
3. `webrtc_sender.html` - Browser webcam client

(Ask Claude Code to generate these files based on the architecture described above)

## Resources

- [YOLOv8 Docs](https://docs.ultralytics.com/)
- [aiortc Docs](https://aiortc.readthedocs.io/)
- [FFmpeg VideoToolbox](https://trac.ffmpeg.org/wiki/HWAccelIntro)
- [WebRTC API](https://developer.mozilla.org/en-US/docs/Web/API/WebRTC_API)

## License

MIT - Have fun detecting hot dogs! 🌭
```

---

**This README gives Claude Code everything it needs to help you build the demo. Just paste it into your Code session and ask it to generate the three Python/HTML files!**
