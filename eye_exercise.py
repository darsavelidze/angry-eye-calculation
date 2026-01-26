"""
Eye Gymnastics Program
Программа для гимнастики глаз с 6 упражнениями
"""

import cv2
import torch
import numpy as np
import time
from enum import Enum
from dataclasses import dataclass
from l2cs import Pipeline, render
import pyrealsense2 as rs


class ExerciseType(Enum):
    UP_DOWN = 1
    LEFT_RIGHT = 2
    DIAGONAL_LU_RD = 3
    DIAGONAL_LD_RU = 4
    CIRCLE_CW = 5
    CIRCLE_CCW = 6


@dataclass
class ExercisePoint:
    """Target point for gaze"""
    x: int
    y: int
    name: str


class EyeExerciseProgram:
    def __init__(self, width=640, height=480, hold_time=1.0):
        """
        Initialize eye exercise program
        
        Args:
            width: Screen width
            height: Screen height
            hold_time: Time in seconds to hold gaze on target (default 1.0 sec)
        """
        self.width = width
        self.height = height
        self.hold_time = hold_time
        self.center_x = width // 2
        self.center_y = height // 2
        
        # Initialize gaze pipeline
        self.gaze_pipeline = Pipeline(
            weights='models/L2CSNet_gaze360.pkl',
            arch='ResNet50',
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
        
        # Circle parameters
        self.circle_radius = 30
        self.circle_color = (0, 255, 255)  # Yellow
        self.circle_thickness = 2
        self.fill_color = (255, 0, 0)  # Blue
        
        # Define exercises
        self.exercises = self._define_exercises()
        self.current_exercise_idx = 0
        self.current_point_idx = 0
        self.gaze_start_time = None
        self.current_pitch = 0
        self.current_yaw = 0
        
    def _define_exercises(self):
        """Define all 6 exercises"""
        exercises = []
        
        # Exercise 1: Up and Down (3 times)
        up_down = {
            'name': 'Up and Down',
            'repeats': 3,
            'points': [
                ExercisePoint(self.center_x, 80, 'UP'),
                ExercisePoint(self.center_x, 400, 'DOWN'),
            ]
        }
        exercises.append(up_down)
        
        # Exercise 2: Left and Right (3 times)
        left_right = {
            'name': 'Left and Right',
            'repeats': 3,
            'points': [
                ExercisePoint(60, self.center_y, 'LEFT'),
                ExercisePoint(580, self.center_y, 'RIGHT'),
            ]
        }
        exercises.append(left_right)
        
        # Exercise 3: Diagonal Left-Up to Right-Down (3 times)
        diag_lu_rd = {
            'name': 'Diagonal: Left-Up → Right-Down',
            'repeats': 3,
            'points': [
                ExercisePoint(80, 80, 'LEFT-UP'),
                ExercisePoint(560, 400, 'RIGHT-DOWN'),
            ]
        }
        exercises.append(diag_lu_rd)
        
        # Exercise 4: Diagonal Left-Down to Right-Up (3 times)
        diag_ld_ru = {
            'name': 'Diagonal: Left-Down → Right-Up',
            'repeats': 3,
            'points': [
                ExercisePoint(80, 400, 'LEFT-DOWN'),
                ExercisePoint(560, 80, 'RIGHT-UP'),
            ]
        }
        exercises.append(diag_ld_ru)
        
        # Exercise 5: Circle Clockwise
        circle_cw = {
            'name': 'Circle Clockwise',
            'repeats': 1,
            'points': self._generate_circle_points(clockwise=True)
        }
        exercises.append(circle_cw)
        
        # Exercise 6: Circle Counter-Clockwise
        circle_ccw = {
            'name': 'Circle Counter-Clockwise',
            'repeats': 1,
            'points': self._generate_circle_points(clockwise=False)
        }
        exercises.append(circle_ccw)
        
        return exercises
    
    def _generate_circle_points(self, clockwise=True, num_points=8):
        """Generate points in a circle"""
        points = []
        radius = 120  # Increased radius for better circular motion
        
        if clockwise:
            angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
        else:
            angles = np.linspace(2*np.pi, 0, num_points, endpoint=False)
        
        for i, angle in enumerate(angles):
            x = int(self.center_x + radius * np.cos(angle))
            y = int(self.center_y + radius * np.sin(angle))
            
            # Clamp to safe boundaries (30 pixels from edge)
            x = np.clip(x, 60, 580)
            y = np.clip(y, 60, 420)
            
            direction = ['RIGHT', 'DOWN-RIGHT', 'DOWN', 'DOWN-LEFT', 
                        'LEFT', 'UP-LEFT', 'UP', 'UP-RIGHT'][i % 8]
            points.append(ExercisePoint(x, y, direction))
        
        return points
    
    def gaze_point_to_screen(self, pitch, yaw, frame_width=640, frame_height=480):
        """
        Convert gaze angles (pitch, yaw) to absolute screen coordinates
        pitch: Gaze pitch angle in RADIANS (from L2CS)
        yaw: Gaze yaw angle in RADIANS (from L2CS)
        Returns absolute position on screen (0-width, 0-height)
        """
        # Calculate gaze direction using L2CS formula (already in radians)
        # dx = -sin(pitch) * cos(yaw)
        # dy = -sin(yaw)
        dx = -np.sin(pitch) * np.cos(yaw)
        dy = -np.sin(yaw)
        
        # Scale to screen coordinates
        scale = 200  # Sensitivity
        
        x = self.center_x + dx * scale
        y = self.center_y + dy * scale
        
        # Clamp to screen boundaries
        x = int(np.clip(x, 0, self.width - 1))
        y = int(np.clip(y, 0, self.height - 1))
        
        return x, y
    
    def draw_gaze_arrow_at_center(self, frame, pitch, yaw, length=80):
        """Draw gaze direction arrow at screen center"""
        # Calculate arrow end point using L2CS formula (pitch/yaw already in radians)
        dx = -length * np.sin(pitch) * np.cos(yaw)
        dy = -length * np.sin(yaw)
        
        start_point = (self.center_x, self.center_y)
        end_point = (int(self.center_x + dx), int(self.center_y + dy))
        
        # Draw arrow
        cv2.arrowedLine(frame, start_point, end_point, 
                       (0, 255, 255), 2, cv2.LINE_AA, tipLength=0.2)
        
        # Draw center circle
        cv2.circle(frame, start_point, 5, (0, 255, 255), -1)
        
        return frame
    
    def is_gaze_on_target(self, pitch, yaw, target_x, target_y, angle_threshold=30, verbose=True):
        """
        Check if gaze direction vector aligns with vector from center to target
        
        Args:
            pitch: Gaze pitch angle in RADIANS (from L2CS)
            yaw: Gaze yaw angle in RADIANS (from L2CS)
            target_x: Target point X coordinate
            target_y: Target point Y coordinate
            angle_threshold: Maximum angle difference in degrees (default 30)
            verbose: Print debug information
        
        Returns:
            True if gaze direction aligns with target direction
        """
        # Calculate target vector from center to target point
        target_vec_x = target_x - self.center_x
        target_vec_y = target_y - self.center_y
        
        # Normalize target vector
        target_length = np.sqrt(target_vec_x**2 + target_vec_y**2)
        if target_length == 0:
            return False
        
        target_vec_x /= target_length
        target_vec_y /= target_length
        
        # Calculate target angle from center (in degrees for logging)
        target_angle_deg = np.degrees(np.arctan2(target_vec_y, target_vec_x))
        
        # Calculate gaze direction vector - using L2CS formula from vis.py
        # L2CS pitch and yaw are already in RADIANS
        # dx = -length * sin(pitch) * cos(yaw)  -> horizontal component
        # dy = -length * sin(yaw)               -> vertical component
        gaze_vec_x = -np.sin(pitch) * np.cos(yaw)
        gaze_vec_y = -np.sin(yaw)
        
        # Normalize gaze vector
        gaze_length = np.sqrt(gaze_vec_x**2 + gaze_vec_y**2)
        if gaze_length < 0.001:  # Looking straight ahead
            if verbose:
                print(f"[GAZE DEBUG] Looking straight ahead (small gaze vector)")
            return False
            
        gaze_vec_x /= gaze_length
        gaze_vec_y /= gaze_length
        
        # Calculate gaze angle (in degrees for logging)
        gaze_angle_deg = np.degrees(np.arctan2(gaze_vec_y, gaze_vec_x))
        
        # Calculate dot product (cosine of angle between vectors)
        dot_product = gaze_vec_x * target_vec_x + gaze_vec_y * target_vec_y
        
        # Clamp dot product to [-1, 1] to avoid numerical errors in arccos
        dot_product = np.clip(dot_product, -1.0, 1.0)
        
        # Calculate angle between vectors in degrees
        angle_diff = np.degrees(np.arccos(dot_product))
        
        if verbose:
            pitch_deg = np.degrees(pitch)
            yaw_deg = np.degrees(yaw)
            print(f"[GAZE DEBUG]")
            print(f"  Pitch={pitch_deg:.1f}° (rad:{pitch:.3f}), Yaw={yaw_deg:.1f}° (rad:{yaw:.3f})")
            print(f"  Gaze Vector: ({gaze_vec_x:.2f}, {gaze_vec_y:.2f}), Angle: {gaze_angle_deg:.1f}°")
            print(f"  Target: ({target_x}, {target_y}), Angle: {target_angle_deg:.1f}°")
            print(f"  Angle Diff: {angle_diff:.1f}° (threshold={angle_threshold}°)")
            print(f"  Match: {'YES ✓' if angle_diff < angle_threshold else 'NO'}")
            print()
        
        # Check if angle difference is within threshold
        return angle_diff < angle_threshold
    
    def draw_circle_with_progress(self, frame, center_x, center_y, target_x, target_y, 
                                 progress, is_current=False):
        """Draw a circle with progress fill"""
        circle_color = (0, 255, 255)  # Yellow
        
        if is_current:
            circle_color = (0, 255, 0)  # Green for current target
        
        # Draw circle outline
        cv2.circle(frame, (center_x, center_y), self.circle_radius, 
                  circle_color, self.circle_thickness)
        
        # Draw progress arc (pie-style fill)
        if progress > 0:
            # Draw filled pie sector (progress indicator)
            radius = self.circle_radius + 3
            num_segments = max(1, int((progress / 100.0) * 32))  # 32 segments = full circle
            
            for i in range(num_segments):
                angle1 = (i / 32.0) * 360 - 90  # Start from top
                angle2 = ((i + 1) / 32.0) * 360 - 90
                
                # Convert angles to radians
                a1 = np.radians(angle1)
                a2 = np.radians(angle2)
                
                # Calculate end points of the arc
                x1 = int(center_x + radius * np.cos(a1))
                y1 = int(center_y + radius * np.sin(a1))
                x2 = int(center_x + radius * np.cos(a2))
                y2 = int(center_y + radius * np.sin(a2))
                
                # Draw line from center through arc
                cv2.line(frame, (center_x, center_y), (x1, y1), self.fill_color, 2)
                cv2.line(frame, (center_x, center_y), (x2, y2), self.fill_color, 2)
            
            # Fill the sector with polygon
            if num_segments > 1:
                points = [(center_x, center_y)]
                for i in range(num_segments + 1):
                    angle = (i / 32.0) * 360 - 90
                    a = np.radians(angle)
                    x = int(center_x + radius * np.cos(a))
                    y = int(center_y + radius * np.sin(a))
                    points.append((x, y))
                
                pts = np.array(points, np.int32)
                cv2.fillPoly(frame, [pts], self.fill_color)
        
        return frame
    
    def get_current_target(self):
        """Get current target point"""
        if self.current_exercise_idx < len(self.exercises):
            exercise = self.exercises[self.current_exercise_idx]
            points = exercise['points']
            if self.current_point_idx < len(points):
                return points[self.current_point_idx]
        return None
    
    def move_to_next_point(self):
        """Move to next point or exercise"""
        exercise = self.exercises[self.current_exercise_idx]
        points = exercise['points']
        
        self.current_point_idx += 1
        
        # Check if we need to repeat exercise
        if self.current_point_idx >= len(points):
            self.current_point_idx = 0
            exercise['repeats'] -= 1
            
            if exercise['repeats'] <= 0:
                # Move to next exercise
                self.current_exercise_idx += 1
                if self.current_exercise_idx >= len(self.exercises):
                    return False  # Program finished
        
        self.gaze_start_time = None
        return True  # Continue program
    
    def process_frame(self, frame):
        """Process frame and return annotated frame with gaze tracking"""
        try:
            results = self.gaze_pipeline.step(frame)
            
            if results is not None and len(results.pitch) > 0:
                # Get first face's gaze
                pitch = results.pitch[0]
                yaw = results.yaw[0]
                
                self.current_pitch = pitch
                self.current_yaw = yaw
                
                # Convert gaze angles to screen coordinates (relative from center)
                gaze_x, gaze_y = self.gaze_point_to_screen(pitch, yaw)
                
                # Draw bounding boxes only (without gaze arrows on face)
                for bbox in results.bboxes:
                    x_min = int(bbox[0])
                    if x_min < 0:
                        x_min = 0
                    y_min = int(bbox[1])
                    if y_min < 0:
                        y_min = 0
                    x_max = int(bbox[2])
                    y_max = int(bbox[3])
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
                
                # Draw gaze arrow at screen center instead of on face
                frame = self.draw_gaze_arrow_at_center(frame, pitch, yaw)
                
                return frame, gaze_x, gaze_y
            
        except Exception as e:
            print(f"Error processing frame: {e}")
        
        return frame, None, None
    
    def run(self):
        """Main program loop"""
        # Initialize RealSense D415 camera
        pipeline = rs.pipeline()
        config = rs.config()
        
        # Configure streams from D415 camera
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # Start streaming
        profile = pipeline.start(config)
        
        print("Starting Eye Gymnastics Program with D415 Camera...")
        print("Press 'q' to exit, 's' to skip current exercise")
        
        try:
            while True:
                # Capture frame from RealSense D415
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                
                if not color_frame:
                    continue
                
                frame = np.asanyarray(color_frame.get_data())
                
                # Process frame for gaze
                frame, gaze_x, gaze_y = self.process_frame(frame)
                
                # Draw background info
                exercise = self.exercises[self.current_exercise_idx]
                target = self.get_current_target()
                
                if target is None:
                    # Program finished
                    cv2.putText(frame, "Congratulations! Program completed!",
                               (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
                    cv2.imshow('Eye Gymnastics', frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    continue
                
                # Draw exercise info
                ex_text = f"Exercise {self.current_exercise_idx + 1}/6: {exercise['name']}"
                cv2.putText(frame, ex_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                repeat_text = f"Repeat: {exercise['repeats']}"
                cv2.putText(frame, repeat_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                target_text = f"Target: {target.name}"
                cv2.putText(frame, target_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Check gaze alignment with CURRENT TARGET ONLY
                current_progress = 0
                if gaze_x is not None and gaze_y is not None:
                    # Use pitch and yaw angles to check direction alignment with current target only
                    if self.is_gaze_on_target(self.current_pitch, self.current_yaw, target.x, target.y, verbose=True):
                        if self.gaze_start_time is None:
                            self.gaze_start_time = time.time()
                        
                        elapsed = time.time() - self.gaze_start_time
                        current_progress = min(100, int((elapsed / self.hold_time) * 100))
                        print(f">>> PROGRESS: {current_progress}% ({elapsed:.2f}s / {self.hold_time}s)")
                        
                        # Check if hold time complete
                        if elapsed >= self.hold_time:
                            print(f">>> TARGET COMPLETE! Moving to next...")
                            self.move_to_next_point()
                    else:
                        self.gaze_start_time = None
                
                # Draw all target points
                points = exercise['points']
                for i, point in enumerate(points):
                    is_current = (i == self.current_point_idx)
                    progress = current_progress if is_current else 0
                    
                    self.draw_circle_with_progress(frame, point.x, point.y, 
                                                 gaze_x or self.center_x, 
                                                 gaze_y or self.center_y,
                                                 progress, is_current)
                
                # Draw current gaze point
                if gaze_x is not None and gaze_y is not None:
                    cv2.circle(frame, (gaze_x, gaze_y), 5, (0, 0, 255), -1)
                
                cv2.imshow('Eye Gymnastics', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Skip to next exercise
                    self.current_exercise_idx += 1
                    self.current_point_idx = 0
                    self.gaze_start_time = None
        
        except KeyboardInterrupt:
            print("Stream interrupted by user")
        
        finally:
            cv2.destroyAllWindows()
            # Stop streaming
            pipeline.stop()
            print("Camera stream stopped")


if __name__ == '__main__':
    program = EyeExerciseProgram(width=640, height=480, hold_time=1.0)
    program.run()
