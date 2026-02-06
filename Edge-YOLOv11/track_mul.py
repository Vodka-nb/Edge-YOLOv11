import warnings
from ultralytics import YOLO
import os
from pathlib import Path
import cv2
import csv
from region_count_dev import count_objects_in_region, get_center_region

warnings.filterwarnings('ignore')

# ======== 配置区域 START ========
model_path = 'runs/train/exp-yolov11s-a+b+c/weights/best.pt'
input_video_dir = 'video/'  # 输入视频文件夹
output_base_dir = 'runs/all'  # 输出根文件夹
save_video = True
show_video = False
imgsz = 1024
tracker_cfg = 'ultralytics/cfg/trackers/botsort.yaml'
# ======== 配置区域 END ========

def create_output_folder(base_path, base_name):
    output_folder = os.path.join(base_path, base_name)
    counter = 1
    while os.path.exists(output_folder):
        output_folder = os.path.join(base_path, f"{base_name}_{counter}")
        counter += 1
    os.makedirs(output_folder)
    return output_folder

def process_video(video_path, model, output_folder):
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None, None

    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Processing {video_name}: {video_width}x{video_height}")

    region = get_center_region(video_width, video_height)
    print(f"Center region: {region}")

    results = model.track(
        source=video_path,
        tracker=tracker_cfg,
        imgsz=imgsz,
        save=False,
        save_txt=False,
        show=False
    )

    output_txt_path = os.path.join(output_folder, f'{video_name}.txt')
    all_results = []
    counted_objects = set()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video_path = os.path.join(output_folder, f"{video_name}_result.mp4")
    out_video = cv2.VideoWriter(output_video_path, fourcc, 30, (video_width, video_height)) if save_video else None

    for frame_idx, result in enumerate(results):
        boxes = result.boxes
        if boxes is None:
            continue

        frame = result.plot()
        cv2.rectangle(frame, region[0], region[1], (0, 0, 255), 2)
        cv2.putText(frame, f'Objects in region: {len(counted_objects)}', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        for box in boxes:
            if box.xyxy.ndimension() == 2 and box.xyxy.shape[1] == 4:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                obj_id = int(box.id) if box.id is not None else -1
                detected_boxes = [(x1, y1, x2, y2, obj_id)]
                counted_objects, count = count_objects_in_region(detected_boxes, region, counted_objects)

                all_results.append([
                    frame_idx,
                    obj_id,
                    x1, y1, x2 - x1, y2 - y1,
                    box.conf.item()
                ])

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                id_text = f'ID: {obj_id}' if obj_id != -1 else 'ID: Unknown'
                cv2.putText(frame, id_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if save_video and out_video is not None:
            out_video.write(frame)

        if show_video:
            cv2.imshow('Detection with Region', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if all_results:
        with open(output_txt_path, 'w') as f:
            for line in all_results:
                f.write(' '.join(map(str, line)) + '\n')
        print(f"Saved results to {output_txt_path}")
    else:
        print(f"No valid detections in {video_name}")

    if save_video and out_video is not None:
        out_video.release()

    cap.release()
    if show_video:
        cv2.destroyAllWindows()

    return video_name, len(counted_objects)

if __name__ == '__main__':
    model = YOLO(model_path)

    input_path = Path(input_video_dir)
    if not input_path.exists():
        print(f"Input directory does not exist: {input_video_dir}")
        exit()

    video_files = list(input_path.glob('*.mp4')) + list(input_path.glob('*.avi')) + list(input_path.glob('*.MOV'))

    if not video_files:
        print("No video files found in the input directory.")
        exit()

    output_folder = create_output_folder(output_base_dir, 'result')
    print(f"All results will be saved in: {output_folder}")

    # 保存统计结果的列表
    summary_data = []

    for video in video_files:
        video_name, count = process_video(str(video), model, output_folder)
        if video_name is not None:
            summary_data.append([video_name, count])

    # 写入CSV汇总表格
    csv_output_path = os.path.join(output_folder, 'summary.csv')
    with open(csv_output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Video Name', 'Counted Objects in Region'])
        writer.writerows(summary_data)

    print(f"Summary CSV saved to: {csv_output_path}")
    print("Batch tracking complete.")
