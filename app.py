from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
import cv2
import base64
import json
import time
from image_classification import PotholeDetector
import threading
import io
from PIL import Image
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Global detector instance - make it truly global
detector = None
detector_lock = threading.Lock()
detector_initialized = False

def initialize_detector():
    """Initialize the global detector instance and auto-start detection"""
    global detector, detector_initialized
    
    with detector_lock:
        # Only initialize if not already done
        if not detector_initialized:
            # Try different camera indices to find available camera
            camera_indices = [0, 1, 2]  # Try default camera first
            
            for idx in camera_indices:
                try:
                    print(f"Trying camera index {idx}...")
                    test_cap = cv2.VideoCapture(idx)
                    if test_cap.isOpened():
                        test_cap.release()
                        print(f"Camera {idx} detected successfully")
                        detector = PotholeDetector(camera_index=idx)
                        break
                except Exception as e:
                    print(f"Camera {idx} failed: {e}")
                    continue
            
            if detector is None:
                print("Warning: No camera could be initialized, using default")
                detector = PotholeDetector(camera_index=0)
            
            # Auto-start detection
            print("Auto-starting detection...")
            detector.start()
            detector_initialized = True
            
    return detector

# Only initialize once, check for werkzeug reloader
if os.environ.get('WERKZEUG_RUN_MAIN') == 'true' or not app.debug:
    print("Initializing detector on startup...")
    initialize_detector()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/start', methods=['POST'])
def start_detection():
    """Start pothole detection (if not already running)"""
    try:
        with detector_lock:
            if detector and not detector.running:
                detector.start()
                return jsonify({'status': 'success', 'message': 'Detection started'})
            elif detector and detector.running:
                return jsonify({'status': 'warning', 'message': 'Detection already running'})
            else:
                return jsonify({'status': 'error', 'message': 'Detector not initialized'}), 500
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/stop', methods=['POST'])
def stop_detection():
    """Stop pothole detection"""
    try:
        with detector_lock:
            if detector and detector.running:
                detector.stop()
                return jsonify({'status': 'success', 'message': 'Detection stopped'})
            else:
                return jsonify({'status': 'warning', 'message': 'Detection not running'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/restart', methods=['POST'])
def restart_detection():
    """Restart detection (stop and start)"""
    try:
        with detector_lock:
            if detector:
                if detector.running:
                    detector.stop()
                    time.sleep(1)
                detector.start()
                return jsonify({'status': 'success', 'message': 'Detection restarted'})
            else:
                return jsonify({'status': 'error', 'message': 'Detector not initialized'}), 500
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/stats')
def get_stats():
    """Get current detection statistics"""
    try:
        if detector:
            stats = detector.get_statistics()
            # Debug print
            print(f"Stats being sent: frames={stats.get('frames_processed', 0)}, total={stats.get('total_detections', 0)}")
            return jsonify(stats)
        else:
            return jsonify({'error': 'Detector not initialized'}), 500
    except Exception as e:
        print(f"Error in get_stats: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/reset', methods=['POST'])
def reset_stats():
    """Reset statistics"""
    try:
        if detector:
            detector.reset_stats()
            return jsonify({'status': 'success', 'message': 'Stats reset'})
        else:
            return jsonify({'status': 'error', 'message': 'Detector not initialized'}), 500
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/detections')
def get_detections():
    """Get recent detections"""
    try:
        if detector:
            detections = detector.get_detections()
            return jsonify({'detections': detections})
        else:
            return jsonify({'error': 'Detector not initialized'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def generate_frames():
    """Generate frames for video streaming"""
    while True:
        if detector and detector.current_frame is not None:
            # Get the current frame with annotations
            frame = detector.current_frame.copy()
            
            # Draw detections on frame
            for pred in detector.latest_predictions:
                if isinstance(pred, dict):
                    x = pred.get('x', 0)
                    y = pred.get('y', 0)
                    w = pred.get('width', 0)
                    h = pred.get('height', 0)
                    conf = pred.get('confidence', 0)
                    
                    if x and y:
                        x1 = int(x - w/2)
                        y1 = int(y - h/2)
                        x2 = int(x + w/2)
                        y2 = int(y + h/2)
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        label = f"Pothole {conf:.2f}"
                        cv2.putText(frame, label, (x1, y1-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Draw status
            if detector.latest_predictions:
                status = f"WARNING: {len(detector.latest_predictions)} POTHOLE(S)"
                color = (0, 0, 255)
            else:
                status = "CLEAR ROAD"
                color = (0, 255, 0)
                
            cv2.putText(frame, status, (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            # Yield frame in byte format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            # Send a placeholder image when no frame is available
            time.sleep(0.1)

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/settings', methods=['POST'])
def update_settings():
    """Update detection settings"""
    try:
        data = request.json
        if 'interval' in data:
            detector.process_interval = float(data['interval'])
        if 'camera' in data:
            # Restart with new camera
            was_running = detector.running
            if was_running:
                detector.stop()
            detector.camera_index = int(data['camera'])
            if was_running:
                time.sleep(1)
                detector.start()
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

print("Initializing detector on startup...")
initialize_detector()

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, threaded=True, host='0.0.0.0', port=5000)