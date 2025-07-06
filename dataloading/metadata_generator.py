import os
import json


def generate_metadata(path_parent_dir):
    # Output metadata file
    output_metadata_file = os.path.join(path_parent_dir, "train_metadata.json")

    # Check if the metadata file already exists and remove it
    if os.path.exists(output_metadata_file):
        os.remove(output_metadata_file)
        print(f"Existing metadata file {output_metadata_file} has been removed.")

    # List to store metadata entries
    metadata = []

    # Walk through all subdirectories to find JSON files
    for root, _, files in os.walk(path_parent_dir):
        for json_file in files:
            if json_file.endswith(".json"):
                with open(os.path.join(root, json_file), "r") as f:
                    video_data = json.load(f)
                
                # Extract relevant fields
                metadata_entry = {
                    "file": video_data.get("file"),
                    "original": video_data.get("original"),
                    "split": "train",  # Set the split
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

    print(f"Metadata saved to {output_metadata_file}")