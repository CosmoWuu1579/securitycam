from flask import Flask, Response, render_template_string, jsonify
import cv2
from ultralytics import YOLO

app = Flask(__name__)

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

model = YOLO("yolo11m.pt")

# Tracking variables
LEFT_END = 750
RIGHT_END = 250
initialPosition = {}
mostRecentPosition = {}
numEntered = 0
frame_count = 0

def generate_frames():
    global initialPosition, mostRecentPosition, numEntered, frame_count
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        frame_count += 1
        
        # Run YOLO inference with tracking
        results = model.track(
            frame, 
            tracker="bytetrack.yaml", 
            persist=True, 
            classes=[0],
            verbose=False,
            iou=0.65,
            conf=0.35,
        )
        
        # Get annotated frame
        annotated_frame = results[0].plot()
        
        # Track detected IDs
        detectedIDs = set()
        if results[0].boxes is not None:
            for box in results[0].boxes:
                if box.id is not None:
                    track_id = int(box.id.item())
                    detectedIDs.add(track_id)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    if track_id not in initialPosition:
                        initialPosition[track_id] = x1 
                    mostRecentPosition[track_id] = x1
        
        # Check for disappeared IDs
        disappeared_ids = initialPosition.keys() - detectedIDs
        
        if frame_count % 30 == 0:
            print(f"Disappeared IDs: {disappeared_ids}")
            for track_id in disappeared_ids:
                print(mostRecentPosition.get(track_id, "Not found"))
                print(initialPosition.get(track_id, "Not found"))
                # Person entered (moved from right to left end)
                if (# initialPosition[track_id] > RIGHT_END and 
                    # mostRecentPosition[track_id] <= RIGHT_END and 
                    mostRecentPosition[track_id] < initialPosition[track_id]):
                    numEntered += 1
                    print(f"ID {track_id} entered. Total entered: {numEntered}")
                
                # Person exited (moved from left to right end)
                elif (# initialPosition[track_id] < LEFT_END and 
                      # mostRecentPosition[track_id] >= LEFT_END and 
                      mostRecentPosition[track_id] > initialPosition[track_id]):
                    numEntered -= 1
                    print(f"ID {track_id} exited. Total entered: {numEntered}")
                
                # Clean up tracking data
                del initialPosition[track_id]
                del mostRecentPosition[track_id]
        
        # Draw boundary lines
        cv2.line(annotated_frame, (RIGHT_END, 0), (RIGHT_END, annotated_frame.shape[0]), 
                 (0, 255, 255), 3)
        cv2.line(annotated_frame, (LEFT_END, 0), (LEFT_END, annotated_frame.shape[0]), 
                 (0, 255, 255), 3)
        
        # Add count overlay with background for better visibility
        text = f"People Inside: {numEntered}"
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 1.8
        thickness = 4
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Draw background rectangle
        cv2.rectangle(annotated_frame, (5, 5), (text_width + 25, text_height + 35), 
                     (0, 0, 0), -1)
        cv2.rectangle(annotated_frame, (5, 5), (text_width + 25, text_height + 35), 
                     (0, 255, 0), 3)
        
        # Draw text
        cv2.putText(annotated_frame, text, (15, text_height + 15),
                    font, font_scale, (0, 255, 0), thickness)
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
        
        # Stream to browser
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
@app.route('/count')
def get_count():
    return jsonify({'count': numEntered})

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template_string("""
    <!DOCTYPE html>
    <html>
      <head>
        <title>Person Tracking System</title>
        <style>
          * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
          }
          
          body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
          }
          
          .header {
            text-align: center;
            color: white;
            margin-bottom: 30px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
          }
          
          .header h1 {
            font-size: 3em;
            margin-bottom: 10px;
            font-weight: 700;
          }
          
          .header p {
            font-size: 1.2em;
            opacity: 0.9;
          }
          
          .stats-container {
            display: flex;
            gap: 20px;
            margin-bottom: 30px;
            flex-wrap: wrap;
            justify-content: center;
          }
          
          .stat-card {
            background: rgba(255, 255, 255, 0.95);
            padding: 25px 40px;
            border-radius: 15px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.2);
            text-align: center;
            min-width: 200px;
            transition: transform 0.3s ease;
          }
          
          .stat-card:hover {
            transform: translateY(-5px);
          }
          
          .stat-label {
            color: #666;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
          }
          
          .stat-value {
            color: #667eea;
            font-size: 3em;
            font-weight: bold;
            font-family: 'Courier New', monospace;
          }
          
          .video-container {
            background: white;
            padding: 20px;
            border-radius: 20px;
            box-shadow: 0 10px 50px rgba(0,0,0,0.3);
            max-width: 1320px;
          }
          
          .video-wrapper {
            position: relative;
            border-radius: 10px;
            overflow: hidden;
            border: 3px solid #667eea;
          }
          
          img {
            display: block;
            width: 100%;
            height: auto;
          }
          
          .status-bar {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            border-radius: 10px;
            margin-top: 20px;
            text-align: center;
            font-size: 1.1em;
          }
          
          .pulse {
            animation: pulse 2s infinite;
          }
          
          @keyframes pulse {
            0%, 100% {
              transform: scale(1);
            }
            50% {
              transform: scale(1.05);
            }
          }
          
          .live-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            background: #ff4444;
            border-radius: 50%;
            margin-right: 8px;
            animation: blink 1.5s infinite;
          }
          
          @keyframes blink {
            0%, 100% {
              opacity: 1;
            }
            50% {
              opacity: 0.3;
            }
          }
          
          .info-section {
            background: rgba(255, 255, 255, 0.95);
            padding: 20px;
            border-radius: 15px;
            margin-top: 20px;
            max-width: 1320px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.2);
          }
          
          .info-section h3 {
            color: #667eea;
            margin-bottom: 10px;
          }
          
          .info-section ul {
            list-style: none;
            padding-left: 0;
          }
          
          .info-section li {
            padding: 8px 0;
            color: #555;
          }
          
          .info-section li:before {
            content: "→ ";
            color: #667eea;
            font-weight: bold;
            margin-right: 5px;
          }
        </style>
        <script>
          // Update count every 2 seconds
          setInterval(function() {
            fetch('/count')
              .then(response => response.json())
              .then(data => {
                document.getElementById('live-count').textContent = data.count;
              });
          }, 2000);
        </script>
      </head>
      <body>
        <div class="header">
          <h1>🎯 Person Tracking System</h1>
          <p>Real-time AI-powered people counting and tracking</p>
        </div>
        
        <div class="stats-container">
          <div class="stat-card pulse">
            <div class="stat-label">Current Count</div>
            <div class="stat-value" id="live-count">0</div>
          </div>
          <div class="stat-card">
            <div class="stat-label">Status</div>
            <div style="margin-top: 15px;">
              <span class="live-indicator"></span>
              <span style="color: #ff4444; font-weight: bold; font-size: 1.5em;">LIVE</span>
            </div>
          </div>
        </div>
        
        <div class="video-container">
          <div class="video-wrapper">
            <img src="{{ url_for('video_feed') }}" alt="Live Feed">
          </div>
          <div class="status-bar">
            <strong>Tracking Active</strong> | Yellow lines mark entry/exit zones
          </div>
        </div>
        
        <div class="info-section">
          <h3>📊 How It Works</h3>
          <ul>
            <li>Yellow boundary lines define the tracking zones</li>
            <li>People crossing from right to left are counted as "entered"</li>
            <li>People crossing from left to right are counted as "exited"</li>
            <li>Each person is assigned a unique tracking ID</li>
            <li>Real-time count updates automatically</li>
          </ul>
        </div>
      </body>
    </html>
    """)

if __name__ == '__main__':
    app.run(debug=True, threaded=True)