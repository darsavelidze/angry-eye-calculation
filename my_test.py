from l2cs import Pipeline, render
import cv2
import torch
import pyrealsense2 as rs

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Configure streams from D415 camera
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Initialize gaze pipeline
gaze_pipeline = Pipeline(
    weights='models/L2CSNet_gaze360.pkl',
    arch='ResNet50',
    device=torch.device('cuda')  # Using CUDA GPU
)

print("Starting video stream from D415 camera...")
print("Press 'q' to exit")

try:
    while True:
        # Capture frame from RealSense D415
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        
        if not color_frame:
            continue
        
        frame = cv2.numpy.asanyarray(color_frame.get_data())
        
        # Process frame and visualize
        try:
            results = gaze_pipeline.step(frame)
            frame = render(frame, results)
        except ValueError as e:
            # No faces detected in frame
            cv2.putText(frame, 'No face detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Display result
        cv2.imshow('Gaze Estimation - D415', frame)
        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Stream interrupted by user")

finally:
    cv2.destroyAllWindows()
    # Stop streaming
    pipeline.stop()
    print("Camera stream stopped")