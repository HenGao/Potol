import cv2
import time
import threading
from inference_sdk import InferenceHTTPClient
import numpy as np
from datetime import datetime
import queue

class PotholeDetector:
    def __init__(self, api_key="4MbQLuyWEuh3RWMV6pyv", camera_index=1):
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
        
        # Frame processing interval
        self.process_interval = 1.0  # seconds
        
        # Queue for thread-safe communication
        self.detection_queue = queue.Queue()
        
    def initialize_camera(self):
        """Initialize the camera"""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print(f"Camera {self.camera_index} not found, trying default camera")
            self.cap = cv2.VideoCapture(0)
            
        if not self.cap.isOpened():
            raise Exception("No camera available")
            
        # Get camera properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
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
        self.thread.daemon = True  # Thread will stop when main program stops
        self.thread.start()
        print("Pothole detection started")
        
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
            
        cv2.destroyAllWindows()
        print("Pothole detection stopped")
        print(f"Summary: {self.frame_count} frames, {self.total_detections} detections")
        
    def _detection_loop(self):
        """Main detection loop (runs in separate thread)"""
        try:
            self.initialize_camera()
        except Exception as e:
            print(f"Camera initialization failed: {e}")
            self.running = False
            return
            
        last_process_time = 0
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read frame")
                break
                
            self.current_frame = frame.copy()
            current_time = time.time()
            
            # Process one frame per interval
            if current_time - last_process_time >= self.process_interval:
                last_process_time = current_time
                self.frame_count += 1
                
                # Process frame
                self._process_frame(frame)
                
            # Display frame (optional - remove for headless operation)
            if self.current_frame is not None:
                self._display_frame(self.current_frame)
                
            # Check for 'q' key (backup stop method)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                print("Stop key pressed")
                self.running = False
                break
                
    def _process_frame(self, frame):
        """Process a single frame for pothole detection"""
        try:
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
                self.total_detections += len(predictions)
                
                # Add to queue for GUI/app consumption
                self.detection_queue.put({
                    'timestamp': datetime.now().isoformat(),
                    'frame_number': self.frame_count,
                    'detections': len(predictions),
                    'predictions': predictions
                })
                
                print(f"Frame {self.frame_count}: {len(predictions)} pothole(s) detected")
                
        except Exception as e:
            print(f"Detection error: {str(e)[:100]}")
            
    def _display_frame(self, frame):
        """Display frame with annotations"""
        display_frame = frame.copy()
        
        # Draw detections
        for pred in self.latest_predictions:
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
                    
                    # Draw box
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    
                    # Draw label
                    label = f"Pothole {conf:.2f}"
                    cv2.putText(display_frame, label, (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Draw status
        if self.latest_predictions:
            status = f"⚠ {len(self.latest_predictions)} POTHOLE(S)"
            color = (0, 0, 255)
        else:
            status = "✓ CLEAR ROAD"
            color = (0, 255, 0)
            
        cv2.putText(display_frame, status, (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Draw stats
        cv2.putText(display_frame, f"Frame: {self.frame_count} | Total: {self.total_detections}", 
                   (10, self.height - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (255, 255, 255), 1)
        
        cv2.imshow('Pothole Detection', display_frame)
        
    def get_current_frame(self):
        """Get the current frame (for GUI display)"""
        return self.current_frame
        
    def get_statistics(self):
        """Get current statistics"""
        return {
            'frames_processed': self.frame_count,
            'total_detections': self.total_detections,
            'is_running': self.running,
            'latest_predictions': self.latest_predictions
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
    
# # Simple console example
# if __name__ == "__main__":
#     # Create detector
#     detector = PotholeDetector(camera_index=1)
    
#     try:
#         # Start detection
#         detector.start()
        
#         # Run for some time (your app would handle this differently)
#         print("Press Enter to stop...")
#         input()  # Wait for Enter key
        
#     finally:
#         # Stop detection
#         detector.stop()