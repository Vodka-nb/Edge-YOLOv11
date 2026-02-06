import warnings
from ultralytics import YOLO
import os
from pathlib import Path
import cv2
from region_count_dev import count_objects_in_region, get_center_region  # Import functions from region_count.py

warnings.filterwarnings('ignore')

def create_output_folder(base_path, base_name):
    """
    Create a unique output folder based on the base name. If the folder already exists, increment the name.
    :param base_path: The base directory where the folder should be created.
    :param base_name: The base folder name.
    :return: A unique folder path.
    """
    output_folder = os.path.join(base_path, base_name)
    counter = 1
    while os.path.exists(output_folder):
        output_folder = os.path.join(base_path, f"{base_name}_{counter}")
        counter += 1
    os.makedirs(output_folder)
    return output_folder

if __name__ == '__main__':
    # Initialize YOLO model
    model = YOLO('runs/train/exp-yolov11s-a+b+c/weights/best.pt')  # Model path

    # Input video path
    video_path = 'video/litchi1.mp4'

    # Extract the video file name without extension
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # Open the video to get its dimensions (width and height)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video dimensions: {video_width}x{video_height}")

    # Get the center detection region based on video dimensions
    region = get_center_region(video_width, video_height)
    print(f"Center region coordinates: {region}")

    # Set the parameters for save_video and show_video
    save_video = False   # Set to True to save video
    show_video = False  # Set to True to show video with detections

    # Track with YOLO (without save and show options directly in track function)
    results = model.track(
        source=video_path,
        tracker='ultralytics/cfg/trackers/botsort.yaml',
        imgsz=1024,
        project='runs/test',
        name='test',
        save=False,  # Comment out save in the track function
        save_txt=False,  # Comment out TXT saving
        show=False  # Comment out show in the track function
    )

    # Create the output folder based on project and name
    output_folder = create_output_folder('runs/test', 'test')
    print(f"Saving output to: {output_folder}")

    # Set the output path for the TXT file (dynamic name)
    output_txt_path = os.path.join(output_folder, f'{video_name}.txt')

    # Initialize a list to store tracking results and a set to track counted objects
    all_results = []
    counted_objects = set()  # To track objects that have been counted in the region

    # Video writer setup to save the result as a video (if save_video=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video_path = os.path.join(output_folder, f"{video_name}_result.mp4")
    out_video = cv2.VideoWriter(output_video_path, fourcc, 30, (video_width, video_height)) if save_video else None

    # Process results frame by frame
    for frame_idx, result in enumerate(results):
        boxes = result.boxes  # Get detected boxes for the current frame

        if boxes is not None:
            # Convert the frame result (image) into an OpenCV format (frame)
            frame = result.plot()  # Get the current frame with bounding boxes drawn by YOLO

            # Draw the detection bounding boxes and the region
            cv2.rectangle(frame, (region[0][0], region[0][1]), (region[1][0], region[1][1]), (0, 0, 255), 2)  # Draw region
            cv2.putText(frame, f'Objects in region: {len(counted_objects)}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            for box in boxes:
                # Ensure box.xyxy is a 1D tensor with 4 elements (x1, y1, x2, y2)
                if box.xyxy.ndimension() == 2 and box.xyxy.shape[1] == 4:
                    # Convert tensor to list and unpack the coordinates
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    # Count objects entering the region (center area)
                    detected_boxes = [(x1, y1, x2, y2, int(box.id) if box.id is not None else -1)]
                    counted_objects, count = count_objects_in_region(detected_boxes, region, counted_objects)

                    # Append results in format [frame_idx, obj_id, x, y, w, h, conf]
                    all_results.append([
                        frame_idx,
                        int(box.id) if box.id is not None else -1,  # Object ID (if available)
                        x1, y1, x2-x1, y2-y1,  # Bounding box: x, y, width, height
                        box.conf.item()  # Confidence score
                    ])

                    # Draw bounding boxes
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                    # 检查 id 是否存在，避免 NoneType 错误
                    if box.id is not None:
                        cv2.putText(frame, f'ID: {int(box.id)}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, 'ID: Unknown', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 0), 2)

            # Save the frame to video file (if save_video=True)
            if save_video and out_video is not None:
                out_video.write(frame)

            # Display the frame with the detection region (when show_video=True)
            if show_video:
                cv2.imshow('Detection with Region', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    # Save all results to a single TXT file
    if all_results:
        with open(output_txt_path, 'w') as f:
            for line in all_results:
                f.write(' '.join(map(str, line)) + '\n')

        print(f"Tracking results saved to {output_txt_path}")
    else:
        print(f"No valid detections found in {video_name}.")

    # Release the video writer and close the OpenCV windows
    if save_video and out_video is not None:
        out_video.release()

    cap.release()
    cv2.destroyAllWindows()
