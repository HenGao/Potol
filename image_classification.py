import cv2
import time
import threading
from inference_sdk import InferenceHTTPClient
import numpy as np
from datetime import datetime
import queue

class PotholeDetector:
    def __init__(self, api_key="4MbQLuyWEuh3RWMV6pyv", camera_index=0):
        """Initialize the pothole detector"""
        self.client = InferenceHTTPClient(
            api_url="https://detect.roboflow.com",
            api_key=api_key
        )
        
        self.camera_index = camera_index
        self.model_id = "pothole-detection-yolov8/1"
        
        # Control flags
        self.running = False
        self.thread = None
        
        # Stats
        self.frame_count = 0
        self.total_detections = 0
        self.current_frame = None
        self.latest_predictions = []
        
        # Video capture
        self.cap = None
        
        # Frame processing interval - 1 frame per second for Pi
        self.process_interval = 1.0  # seconds
        
        # Queue for thread-safe communication
        self.detection_queue = queue.Queue()
        
    def initialize_camera(self):
        """Initialize the camera with lower resolution for Pi"""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print(f"Camera {self.camera_index} not found")
            raise Exception("No camera available")
            
        # Set lower resolution for better performance on Pi
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Lower resolution
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Set buffer size to 1 to always get the latest frame
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Set FPS to 1 to reduce capture rate
        self.cap.set(cv2.CAP_PROP_FPS, 1)
        
        # Get actual camera properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 1
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Camera initialized: {self.width}x{self.height} @ {self.fps}fps")
        
    def start(self):
        """Start the detection in a separate thread"""
        if self.running:
            print("Detector is already running")
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._detection_loop)
        self.thread.daemon = True
        self.thread.start()
        print("Pothole detection started (1 FPS mode for Pi)")
        
    def stop(self):
        """Stop the detection"""
        if not self.running:
            print("Detector is not running")
            return
            
        self.running = False
        
        # Wait for thread to finish
        if self.thread:
            self.thread.join(timeout=2)
            
        # Clean up
        if self.cap:
            self.cap.release()
            self.cap = None
            
        print("Pothole detection stopped")
        print(f"Summary: {self.frame_count} frames, {self.total_detections} detections")
        
    def _detection_loop(self):
        """Optimized detection loop for Raspberry Pi - truly 1 FPS"""
        try:
            self.initialize_camera()
        except Exception as e:
            print(f"Camera initialization failed: {e}")
            self.running = False
            return
        
        while self.running:
            start_time = time.time()
            
            # Clear buffer and get fresh frame
            # This ensures we get the latest frame, not an old buffered one
            ret = False
            for _ in range(5):  # Try to clear buffer
                ret, frame = self.cap.read()
                if ret:
                    break
                    
            if not ret:
                print("Failed to read frame")
                time.sleep(1)  # Wait before retry
                continue
                
            # Store current frame for display
            self.current_frame = frame.copy()
            
            # Process the frame
            self.frame_count += 1
            self._process_frame(frame)
            
            # Calculate how long to sleep to maintain 1 FPS
            elapsed = time.time() - start_time
            sleep_time = max(0, self.process_interval - elapsed)
            
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                print(f"Warning: Processing took {elapsed:.2f}s, longer than interval")
                
    def reset_stats(self):
        """Reset statistics"""
        self.frame_count = 0
        self.total_detections = 0
        self.latest_predictions = []
        while not self.detection_queue.empty():
            try:
                self.detection_queue.get_nowait()
            except:
                break
        print("Stats reset")
        
    def _process_frame(self, frame):
        """Process a single frame for pothole detection"""
        try:
            start = time.time()
            
            # Optional: Resize frame before sending to API to reduce bandwidth
            # resized = cv2.resize(frame, (416, 416))  # YOLO typical size
            
            # Run inference
            result = self.client.infer(frame, model_id=self.model_id)
            
            # Extract predictions
            predictions = []
            if isinstance(result, dict):
                predictions = result.get('predictions', [])
            elif isinstance(result, list):
                predictions = result
                
            self.latest_predictions = predictions
            
            # Update statistics
            if predictions:
                detection_count = len(predictions)
                self.total_detections += detection_count
                
                # Add to queue for GUI/app consumption
                self.detection_queue.put({
                    'timestamp': datetime.now().isoformat(),
                    'frame_number': self.frame_count,
                    'detections': detection_count,
                    'predictions': predictions
                })
                
                print(f"Frame {self.frame_count}: {detection_count} pothole(s) detected (took {time.time()-start:.2f}s)")
            else:
                print(f"Frame {self.frame_count}: Clear road (took {time.time()-start:.2f}s)")
                
        except Exception as e:
            print(f"Detection error: {str(e)[:100]}")
        
    def get_current_frame(self):
        """Get the current frame (for GUI display)"""
        return self.current_frame
        
    def get_statistics(self):
        """Get current statistics"""
        return {
            'frames_processed': self.frame_count,
            'total_detections': self.total_detections,
            'is_running': self.running,
            'latest_predictions': self.latest_predictions,
            'camera_index': self.camera_index,
            'camera_resolution': f"{self.width}x{self.height}" if hasattr(self, 'width') else "Unknown"
        }
        
    def get_detections(self):
        """Get pending detections from queue"""
        detections = []
        while not self.detection_queue.empty():
            try:
                detections.append(self.detection_queue.get_nowait())
            except:
                break
        return detections