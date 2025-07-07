import os
import json


def generate_metadata(dataset_root):
    # Define dataset splits
    splits = ["train", "val", "testA"]

    for split in splits:
        # Path to the split directory
        split_dir = os.path.join(dataset_root, split)

        # Check if the split starts with 'test'
        if split.startswith("test"):
            # Output text file for the split
            output_txt_file = os.path.join(dataset_root, f"{split}_files.txt")

            # Check if the text file already exists and remove it
            if os.path.exists(output_txt_file):
                os.remove(output_txt_file)
                print(f"Existing file list {output_txt_file} has been removed.")

            # List to store file paths
            file_list = []

            # Walk through all subdirectories to find .mp4 files
            for root, _, files in os.walk(split_dir):
                for video_file in files:
                    if video_file.endswith(".mp4"):
                        # Append the relative file path to the list
                        relative_path = os.path.relpath(os.path.join(root, video_file), dataset_root)
                        file_list.append(relative_path)

            # Save the file list to a text file
            with open(output_txt_file, "w") as f:
                for file_path in file_list:
                    f.write(f"{file_path}\n")

            print(f"File list for {split} saved to {output_txt_file}")
        else:
            # Output metadata file for the split
            output_metadata_file = os.path.join(dataset_root, f"{split}_metadata.json")

            # Check if the metadata file already exists and remove it
            if os.path.exists(output_metadata_file):
                os.remove(output_metadata_file)
                print(f"Existing metadata file {output_metadata_file} has been removed.")

            # List to store metadata entries
            metadata = []

            # Walk through all subdirectories to find JSON files
            for root, _, files in os.walk(split_dir):
                for json_file in files:
                    if json_file.endswith(".json"):
                        with open(os.path.join(root, json_file), "r") as f:
                            video_data = json.load(f)

                        # Extract relevant fields
                        metadata_entry = {
                            "file": video_data.get("file"),
                            "original": video_data.get("original"),
                            "split": split,  # Set the split dynamically
                            "fake_segments": video_data.get("fake_segments", []),
                            "fps": 25,  # Default FPS
                            "visual_fake_segments": video_data.get("visual_fake_segments", []),
                            "audio_fake_segments": video_data.get("audio_fake_segments", []),
                            "audio_model": video_data.get("audio_model"),
                            "modify_type": video_data.get("modify_type"),
                            "video_frames": video_data.get("video_frames"),
                            "audio_frames": video_data.get("audio_frames")
                        }

                        # Append to the metadata list
                        metadata.append(metadata_entry)

            # Save metadata to a JSON file
            with open(output_metadata_file, "w") as f:
                json.dump(metadata, f, indent=4)

            print(f"Metadata for {split} saved to {output_metadata_file}")