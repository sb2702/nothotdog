# !🌭 - Real-time Object Detection on Apple M4

A WebRTC demo showcasing Apple M4's unified memory architecture for real-time AI video processing. Detects hot dogs using CoreML YOLO models and streams with local Adaptive Bitrate (ABR) transcoding - all on a single M4 chip.

## Why M4?

This demo highlights the architectural advantages of Apple Silicon's unified memory:

- **Zero-copy processing:** WebRTC decode → CoreML inference → multi-bitrate encode, all in unified memory
- **Neural Engine acceleration:** Hardware-accelerated YOLO inference (~17ms for YOLOv3)
- **VideoToolbox encoding:** 4 simultaneous H.264 streams (1080p, 720p, 480p, 360p) with hardware acceleration
- **Local ABR streaming:** Traditional cloud architectures require CPU→GPU→CPU transfers and separate transcoding servers. M4 does everything locally with minimal latency.

## Architecture

```
Browser Webcam → WebRTC → M4 CoreML YOLO → Blur Hot Dogs → 4x ABR Encode → HLS Stream
                                                              ↓
                                                         Local HLS Server
```

## Prerequisites

- Apple M4 Mac (M1/M2/M3 also supported)
- macOS 14+ (Sonoma or later)
- Python 3.11+
- FFmpeg with VideoToolbox support

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# The CoreML models are included in the models/ directory
```

## Running the Demo

### 1. Start the Signaling Server

```bash
python signaling_server.py
```

This starts the WebSocket signaling server on port 8765.

### 2. Start the WebRTC Receiver with Local ABR

```bash
python webrtc_receiver.py --local-abr --inference-interval 2
```

This starts the main processing pipeline:
- Receives WebRTC video from browser
- Runs CoreML YOLO inference every 2 frames (adjustable)
- Applies Non-Maximum Suppression to find largest hot dog
- Blurs detected hot dogs
- Encodes to 4 ABR streams simultaneously
- Serves HLS content on http://localhost:8000

**Available options:**
- `--local-abr`: Enable local ABR transcoding with HLS output (recommended)
- `--inference-interval N`: Run inference every N frames to improve performance (default: 1)
- `--http-port PORT`: HTTP server port for HLS content (default: 8000)
- `--rtmp-url URL`: Stream to RTMP instead of local ABR (optional)
- `--rtmp-key KEY`: RTMP stream key (if using RTMP)

### 3. Open the Browser Client

```bash
open webcam_client.html
```

Or manually open `webcam_client.html` in Chrome or Safari.

1. Click "Connect" to establish WebRTC connection
2. Allow camera permissions when prompted
3. The demo will show 3 video streams:
   - **Local webcam** (your raw camera feed)
   - **Processed WebRTC** (with hot dog detection/blur)
   - **ABR Stream** (the final HLS output with adaptive bitrate)

## Performance

Measured on M4 Mac Mini:

- **YOLOv3 CoreML Inference:** ~17.6ms (56.8 FPS)
- **YOLOv3-Tiny CoreML Inference:** ~3.8ms (264.7 FPS)
- **4-stream ABR encoding:** Real-time at 30 FPS (1080p, 720p, 480p, 360p)
- **End-to-end latency:** 3-5 seconds (HLS with 1-second segments)

## Latency Measurement

The demo includes a timestamp barcode system for precise latency measurement:

- Binary barcode encoding UTC timestamp is drawn at the bottom of each processed frame
- JavaScript decoder in the browser reads the barcode from the ABR stream
- Calculates and displays the end-to-end latency in real-time

The latency display shows: `Latency: [timestamp] → [current] = [difference]ms`

## Project Structure

```
nothotdog/
├── README.md
├── requirements.txt
├── signaling_server.py       # WebSocket signaling server
├── webrtc_receiver.py         # Main processing pipeline
├── webcam_client.html         # Browser client
├── test_coreml.py            # Model testing script
├── models/
│   ├── YOLOv3FP16.mlmodel    # CoreML YOLO model
│   └── YOLOv3TinyFP16.mlmodel # Faster CoreML model
└── images/
    └── test.jpg              # Test image
```

## How It Works

### 1. WebRTC Connection

- Browser captures webcam via `getUserMedia()`
- WebSocket signaling server exchanges SDP offers/answers
- Direct peer-to-peer WebRTC connection established
- Video frames stream from browser to Python receiver

### 2. CoreML Object Detection

- Frames converted to PIL images for CoreML
- YOLO model detects objects (COCO dataset, 80 classes)
- Hot dog is class index 52
- Non-Maximum Suppression removes duplicate detections
- Largest bounding box selected for blurring

### 3. Local ABR Transcoding

Four simultaneous H.264 encodes using VideoToolbox:
- **1080p** @ 5 Mbps
- **720p** @ 3 Mbps
- **480p** @ 1.5 Mbps
- **360p** @ 800 Kbps

FFmpeg generates HLS segments (1-second duration) and manifest files. The HTTP server serves the content for adaptive playback in the browser.

### 4. Timestamp Barcode

- 64-bit binary barcode encodes UTC timestamp (milliseconds)
- Drawn at bottom center of each frame (4px bar width, 20px height)
- Blue markers on edges for barcode detection
- JavaScript decoder reads barcode from canvas and calculates latency

## Testing with Static Images

```bash
# Test CoreML models on a static image
python test_coreml.py

# Select model:
# 1. YOLOv3FP16.mlmodel
# 2. YOLOv3TinyFP16.mlmodel
```

## Configuration

### Inference Interval

Run inference less frequently to reduce CPU/GPU load:

```bash
# Run inference every 5 frames, cache results between
python webrtc_receiver.py --local-abr --inference-interval 5
```

### ABR Settings

Edit the `streams` list in `webrtc_receiver.py`:

```python
streams = [
    {"name": "1080p", "size": "1920x1080", "bitrate": "5M", "fps": "30"},
    {"name": "720p", "size": "1280x720", "bitrate": "3M", "fps": "30"},
    {"name": "480p", "size": "854x480", "bitrate": "1.5M", "fps": "30"},
    {"name": "360p", "size": "640x360", "bitrate": "800k", "fps": "30"}
]
```

### Detection Confidence

Lower confidence threshold in `webrtc_receiver.py`:

```python
confidence_threshold = 0.1  # Default is 0.1 for more detections
```

## Troubleshooting

### Camera Permissions

```bash
# System Settings → Privacy & Security → Camera
# Allow access for your browser (Chrome/Safari)
```

### Connection Issues

```bash
# Verify signaling server is running
# Check for "WebSocket server running on port 8765" message

# Check no firewall blocking port 8765
```

### Models Not Loading

```bash
# Ensure models exist in models/ directory
ls -l models/

# Should show:
# YOLOv3FP16.mlmodel
# YOLOv3TinyFP16.mlmodel
```

### FFmpeg VideoToolbox Errors

```bash
# Verify FFmpeg has VideoToolbox support
ffmpeg -codecs | grep videotoolbox

# Should show h264_videotoolbox encoder
```

### HLS Stream Not Loading

```bash
# Check HLS files are being created
ls -l hls_output/

# Should show:
# master.m3u8
# 1080p.m3u8, 720p.m3u8, 480p.m3u8, 360p.m3u8
# .ts segment files

# Verify HTTP server is running on port 8000
curl http://localhost:8000/hls_output/master.m3u8
```

## Optional: Streaming to Mux/RTMP

Instead of local ABR, you can stream to any RTMP service:

```bash
python webrtc_receiver.py \
  --rtmp-url rtmp://global-live.mux.com:5222/app \
  --rtmp-key YOUR_STREAM_KEY
```

Works with:
- Mux
- YouTube Live
- Twitch
- Any RTMP server

## Performance Optimization

### Use Inference Interval

```bash
# Run inference every 3 frames instead of every frame
python webrtc_receiver.py --local-abr --inference-interval 3
```

This reduces GPU load while maintaining smooth detection (results are cached between inference runs).

### Use Tiny Model

Edit `webrtc_receiver.py` to use the faster YOLOv3-Tiny model:

```python
model_path = models_dir / "YOLOv3TinyFP16.mlmodel"
```

Inference time drops from ~17ms to ~3.8ms.

## M4 vs Cloud Architecture Comparison

### Traditional Cloud Pipeline
```
Camera → WebRTC → Server CPU → Transfer to GPU → AI Inference →
Transfer to CPU → Upload to CDN → CDN Transcodes → HLS → User
```

**Latency:** 10-30 seconds (multiple network hops, transcoding delay)
**Cost:** GPU compute + CDN transcoding + bandwidth
**Bottlenecks:** CPU↔GPU transfers, CDN processing queue

### M4 Local Pipeline
```
Camera → WebRTC → M4 Unified Memory → AI + Transcoding → HLS → User
```

**Latency:** 3-5 seconds (local processing, no network hops)
**Cost:** Single M4 device
**Advantages:** Zero-copy unified memory, hardware acceleration for everything

## Use Cases

This architecture is ideal for:
- **Live event production** with real-time AI effects
- **Security/surveillance** with on-device object detection
- **Interactive installations** requiring low-latency AI processing
- **Edge computing** scenarios where cloud isn't feasible
- **Cost-sensitive applications** avoiding cloud GPU/transcoding fees

## Credits

- COCO dataset for YOLO training (hot dog class 52)
- Apple CoreML for Neural Engine acceleration
- aiortc for WebRTC in Python
- FFmpeg for VideoToolbox encoding

## License

MIT
