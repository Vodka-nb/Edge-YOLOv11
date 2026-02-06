# Initialize a dictionary to store object crossing status
object_crossed_line = {}

def count_objects_in_region(boxes, region, counted_objects, frame_threshold=20):
    """
    Count objects crossing the central vertical line from top to bottom of the video.
    :param boxes: List of detection boxes [x1, y1, x2, y2, obj_id]
    :param region: The region to check, as a tuple [(x1, y1), (x2, y2)].
    :param counted_objects: Set of objects already counted.
    :param frame_threshold: Minimum number of frames an object must stay in the region to be counted.
    :return: Updated counted_objects set and current count of objects crossing the line.
    """
    region_count = 0  # Initialize region count

    # Define the central vertical line (x_center is the center of the video)
    x_center = region[0][0]  # We use the x-coordinate of the defined region's left side (central line)

    for box in boxes:
        x1, y1, x2, y2, obj_id = box

        # Check if the object has already crossed the line
        if obj_id not in counted_objects:
            # Calculate the center of the object
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            # Check if the object's center has crossed the center line (x_center)
            if x1 < x_center < x2 or x1 < x_center < x2:  # Object passes through the line
                if obj_id not in object_crossed_line:
                    # Register the object crossing the line and count it
                    counted_objects.add(obj_id)
                    object_crossed_line[obj_id] = True
                    region_count += 1

    return counted_objects, region_count

def get_center_region(video_width, video_height, region_ratio=(0.5, 1)):
    """
    Get the coordinates of the vertical center line.
    :param video_width: Width of the video.
    :param video_height: Height of the video.
    :return: The region as a tuple [(x1, y1), (x2, y2)] where the line is drawn.
    """
    # The region for the vertical center line is just a line in the middle
    x_center = video_width // 2  # The x-coordinate of the center line
    y1 = 0
    y2 = video_height

    # Returning the region that defines the line (as the line starts from top to bottom)
    return [(x_center, y1), (x_center, y2)]

# Example usage:

if __name__ == "__main__":
    # Example video dimensions
    video_width = 1280
    video_height = 720

    # Define the center line (just the x-coordinate)
    region = get_center_region(video_width, video_height)

    # Example list of detected boxes [(x1, y1, x2, y2, obj_id)]
    detected_boxes = [
        (100, 200, 150, 250, 1),  # Example detection for object 1
        (400, 300, 450, 350, 2),  # Example detection for object 2
        (150, 200, 200, 250, 1),  # Object 1, already counted
    ]

    # Set to keep track of already counted object IDs
    counted_objects = set()

    # Count objects crossing the center line
    counted_objects, region_count = count_objects_in_region(detected_boxes, region, counted_objects)

    print(f"Objects crossed the line: {region_count}")
    print(f"Center line region: {region}")
    print(f"Counted object IDs: {counted_objects}")
