import os
import cv2
import argparse

def extract_frames(input_dir, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Loop through each class folder
    for class_name in sorted(os.listdir(input_dir)):
        class_path = os.path.join(input_dir, class_name)

        # Skip non-directory items
        if not os.path.isdir(class_path):
            continue

        # Create corresponding class folder in output
        output_class_path = os.path.join(output_dir, class_name)
        os.makedirs(output_class_path, exist_ok=True)

        # Process each video
        for video_name in sorted(os.listdir(class_path)):
            if not video_name.endswith(".avi"):
                continue

            video_path = os.path.join(class_path, video_name)
            video_stem = os.path.splitext(video_name)[0]

            # Create output folder for this video
            output_video_path = os.path.join(output_class_path, video_stem)
            os.makedirs(output_video_path, exist_ok=True)

            # Read video
            cap = cv2.VideoCapture(video_path)
            frame_num = 1

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_filename = f"frame_{frame_num:04d}.jpg"
                frame_filepath = os.path.join(output_video_path, frame_filename)

                cv2.imwrite(frame_filepath, frame)
                frame_num += 1

            cap.release()

    print("Frame extraction complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from video dataset.")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to the directory containing video class folders.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save extracted frames.")

    args = parser.parse_args()

    extract_frames(args.input_dir, args.output_dir)
