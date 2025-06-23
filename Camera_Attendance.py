import cv2
import os
import datetime
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, List

class AttendanceSystem:
    
    def __init__(self, image_dir, excel_dir, avatar_path, excel_filename, camera_index: int = 0):
        """
        Args:
            image_dir (str): Directory to save captured images
            excel_dir (str): Directory to save Excel records
            avatar_path (str): Path to the avatar overlay image
            excel_filename (str): Name of the Excel file
            camera_index (int): Camera index (0 for default camera)
        """
        self.image_dir = image_dir
        self.excel_dir = excel_dir
        self.avatar_path = avatar_path
        self.excel_filename = excel_filename
        self.camera_index = camera_index
        
        # System components
        self.cap = None
        self.face_cascade = None
        self.avatar = None
        self.avatar_mask = None
        
        # Configuration
        self.avatar_size = (400, 400)
        self.avatar_pos = (0, 0)
        self.cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        
        # Initialize the system
        self.setup_attendance()
    
    def setup_attendance(self):
        self.create_directory()
        self.init_excel_file()
        self.load_face_detector()
        
    def create_directory(self):
        directories = [self.image_dir, self.excel_dir]
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"Created directory: {directory}")
    
    def init_excel_file(self):
        excel_path = os.path.join(self.excel_dir, self.excel_filename)
        
        if not os.path.exists(excel_path):
            df = pd.DataFrame(columns=['Date', 'Time', 'Image_Path', 'Image_Name'])
            df.to_excel(excel_path, index=False)
            print(f"Created new Excel file: {excel_path}")
        return excel_path
    
    def load_face_detector(self):
        self.face_cascade = cv2.CascadeClassifier(self.cascade_path)
        if self.face_cascade.empty():
            raise RuntimeError("Failed to load face cascade classifier")
    
    def _initialize_camera(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise IOError(f"Error: Cannot open camera with index {self.camera_index}")
        return self.cap
    
    def load_avatar(self):
        if not os.path.exists(self.avatar_path):
            print(f"Warning: Avatar not found at {self.avatar_path}. System will work without avatar overlay.")
            return None, None
            
        avatar = cv2.imread(self.avatar_path, cv2.IMREAD_UNCHANGED)
        if avatar is None:
            print(f"Warning: Failed to load avatar from {self.avatar_path}")
            return None, None
        
        # Resize and process avatar
        avatar = cv2.resize(avatar, self.avatar_size)
        if avatar.shape[2] == 4:  # Has alpha channel
            _, mask = cv2.threshold(avatar[:,:,3], 1, 255, cv2.THRESH_BINARY)
            return avatar, mask
        else:
            return avatar, None
    
    def is_face_align(self, face: Tuple[int, int, int, int]) -> bool:
        """
        Args:
            face: Tuple of (x, y, width, height) for the detected face
            
        Returns:
            bool: True if face is aligned with avatar
        """
        fx, fy, fw, fh = face
        ax, ay, aw, ah = self.avatar_pos + self.avatar_size
        
        # Check if face center is within avatar bounds
        face_center = (fx + fw//2, fy + fh//2)
        return (ax <= face_center[0] <= ax+aw and ay <= face_center[1] <= ay+ah)
    
    def add_to_excel(self, image_filename, image_path):
        excel_path = os.path.join(self.excel_dir, self.excel_filename)
        
        try:
            df = pd.read_excel(excel_path)
            
            now = datetime.datetime.now()
            
            new_record = {
                'Date': now.strftime("%Y-%m-%d"),
                'Time': now.strftime("%H:%M:%S"),
                'Image_Path': image_path,
                'Image_Name': image_filename,
            }
            
            new_df = pd.DataFrame([new_record])
            df = pd.concat([df, new_df], ignore_index=True)
            
            df.to_excel(excel_path, index=False)
            print(f"Added record to Excel: {image_filename}")
            
        except Exception as e:
            print(f"Error adding to Excel: {str(e)}")
    
    def avatar_overlay(self, frame: np.ndarray):
        if self.avatar is None:
            return frame
            
        display_frame = frame.copy()
        
        if self.avatar.shape[2] == 4:  # Has alpha channel
            alpha = self.avatar[:,:,3] / 255.0
            overlay = self.avatar[:,:,:3]
            
            x, y = self.avatar_pos
            for c in range(3):
                display_frame[y:y+self.avatar_size[1], x:x+self.avatar_size[0], c] = \
                    (1-alpha)*display_frame[y:y+self.avatar_size[1], x:x+self.avatar_size[0], c] + \
                    alpha*overlay[:,:,c]
        else:
            # Simple overlay without alpha
            x, y = self.avatar_pos
            display_frame[y:y+self.avatar_size[1], x:x+self.avatar_size[0]] = self.avatar
            
        return display_frame
    
    def release_camera(self):
        """Release the camera resource if it is open."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            print("Camera released")

    def capture_single_image(self, save_image: bool = True, window_name: str = "Tracking") -> Optional[Tuple[np.ndarray, str]]:
        if self.cap is None:
            self._initialize_camera()
        
        # Load avatar if not already loaded
        if self.avatar is None:
            self.avatar, self.avatar_mask = self.load_avatar()
        
        print(f"Opening camera window: {window_name}")
        print("Instructions:")
        print("- Align your face with the avatar outline")
        print("- Press ENTER when aligned to capture image")
        print("- Press ESC to cancel and close window")
        
        captured_frame = None
        captured_filename = None
        
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            while True:
                # Check if window was closed by user
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    print("Capture window closed by user")
                    break
                
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                    
                frame = cv2.flip(frame, 1)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
                
                # Calculate avatar position
                height, width = frame.shape[:2]
                self.avatar_pos = ((width - self.avatar_size[0]) // 2, height - self.avatar_size[1] - 10)
                
                # Process faces (only keep largest if multiple exist)
                main_face = None
                if len(faces) > 0:
                    main_face = max(faces, key=lambda x: x[2]*x[3])
                
                # Prepare display frame with avatar overlay
                display_frame = self.avatar_overlay(frame)
                
                # Face detection and alignment feedback
                alignment_status = ""
                is_aligned = False
                
                if main_face is not None:
                    x, y, w, h = main_face
                    is_aligned = self.is_face_align(main_face)
                    
                    if is_aligned:
                        color = (0, 255, 0)  # Green
                        alignment_status = "ALIGNED"
                    else:
                        color = (0, 0, 255)  # Red
                        alignment_status = "Move face into frame"
                    
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 3)
                else:
                    alignment_status = "No face detected"
                
                # UI Elements
                cv2.putText(display_frame, alignment_status, (10, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(display_frame, "SPACE = Capture | ESC = Cancel", (10, height-20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Show alignment indicator
                # if is_aligned:
                #     cv2.putText(display_frame, "READY TO CAPTURE!", (10, 80), 
                #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                
                cv2.imshow(window_name, display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC - Cancel
                    print("Capture cancelled by user")
                    break
                elif key == 13:  # SPACE - Capture
                    if main_face is not None and is_aligned:
                        # Successful capture
                        captured_frame = frame.copy()
                        
                        if save_image:
                            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                            captured_filename = f"aligned_face_{timestamp}.jpg"
                            image_path = os.path.join(self.image_dir, captured_filename)
                            
                            cv2.imwrite(image_path, captured_frame)
                            self.add_to_excel(captured_filename, image_path)
                            print(f"Successfully captured and saved: {captured_filename}")
                        else:
                            captured_filename = "temp_capture"
                            print("Image captured (not saved)")
                        
                        # Show capture confirmation for 1 second
                        confirm_frame = display_frame.copy()
                        cv2.putText(confirm_frame, "CAPTURED!", (width//2-100, height//2), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)
                        cv2.imshow(window_name, confirm_frame)
                        cv2.waitKey(1000)
                        
                        break
                    else:
                        print("❌ Cannot capture: Face not properly aligned")
                        # Flash red to indicate failed capture
                        fail_frame = display_frame.copy()
                        cv2.putText(fail_frame, "ALIGN FACE FIRST!", (10, height//2), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
                        cv2.imshow(window_name, fail_frame)
                        cv2.waitKey(500)  # Show for 0.5 seconds
        
        except Exception as e:
            print(f"Error during capture: {str(e)}")
        
        # finally:
        #     # Always close the window
        #     cv2.destroyWindow(window_name)
        #     cv2.waitKey(1)  # Ensure window is properly closed
        #     self.release_camera()
        #     print(f"Camera window '{window_name}' closed")
        
        finally:
            try:
                # Only destroy window if it still exists
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 0:
                    cv2.destroyWindow(window_name)
                    print(f"Camera window '{window_name}' closed")
            except:
                pass
            
            # Always release camera resources
            self.release_camera()
            print("Camera resources released")
            
            # Process any pending events
            cv2.waitKey(1)
        
        # Return the captured frame and filename if successful
        if captured_frame is not None:
            return captured_frame, captured_filename
        else:
            return None
    
    def start_live_capture(self, window_name: str = "Face Alignment System"):
        if self.cap is None:
            self._initialize_camera()
            
        if self.avatar is None:
            self.avatar, self.avatar_mask = self.load_avatar()
        
        print("Align your face with the avatar outline. Press ENTER to capture, ESC to exit.")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
                
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
            
            # Calculate avatar position
            height, width = frame.shape[:2]
            self.avatar_pos = ((width - self.avatar_size[0]) // 2, height - self.avatar_size[1] - 10)
            
            # Process faces (only keep largest if multiple exist)
            main_face = None
            if len(faces) > 0:
                main_face = max(faces, key=lambda x: x[2]*x[3])
                faces = [main_face]
            
            # Prepare display frame with avatar overlay
            display_frame = self.avatar_overlay(frame)
            
            # Face detection and alignment feedback
            alignment_status = ""
            if main_face is not None:
                x, y, w, h = main_face
                is_aligned = self.is_face_align(main_face)
                
                if is_aligned:
                    color = (0, 255, 0)  # Green
                    alignment_status = "Aligned - Press SPACE to capture"
                else:
                    color = (0, 0, 255)  # Red
                    alignment_status = "Move into frame"
                
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
            else:
                alignment_status = "No face detected"
            
            # UI Elements
            cv2.putText(display_frame, alignment_status, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(display_frame, "Press ESC to exit", (10, height-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            cv2.imshow(window_name, display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == 32:  # SPACE
                if main_face is not None and self.is_face_align(main_face):
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    image_filename = f"{timestamp}.jpg"
                    image_path = os.path.join(self.image_dir, image_filename)
                    
                    cv2.imwrite(image_path, frame)
                    self.add_to_excel(image_filename, image_path)
                    print(f"Saved aligned face image: {image_filename}")
                else:
                    print("Capture failed: Face not aligned with avatar")
    
    def get_attendance_records(self):
        excel_path = os.path.join(self.excel_dir, self.excel_filename)
        try:
            return pd.read_excel(excel_path)
        except Exception as e:
            print(f"Error reading Excel file: {str(e)}")
            return pd.DataFrame()
    
    def cleanup(self):
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

def create_attendance_system(image_dir, excel_dir, avatar_path):
    return AttendanceSystem(image_dir, excel_dir, avatar_path)


if __name__ == "__main__":
    print("Face Alignment Attendance System - Library Mode")
    print("Creating system with default settings...")

    IMAGE_DIR = r"D:\Code\Python\Jupyter\OpenCV Image Record\Capture_Image"
    EXCEL_DIR = r"D:\Code\Python\Jupyter\OpenCV Image Record\Excel_Records"
    AVATAR_PATH = r"D:\Code\Python\Jupyter\OpenCV Image Record\Avatar Icon\Avatar Outline.png"
    
    # Create system instance
    attendance_system = AttendanceSystem(
        image_dir=IMAGE_DIR,
        excel_dir=EXCEL_DIR,
        avatar_path=AVATAR_PATH,
        excel_filename="Attandance.xlsx"
    )
    
    try:
        # Start the live capture system
        attendance_system.capture_single_image()
    finally:
        # Clean up resources
        attendance_system.cleanup()
