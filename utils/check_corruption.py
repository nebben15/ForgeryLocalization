import os
import imageio
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def check_video(video_path: str, check_frames: bool = False) -> None:
    """
    Checks a single video file for corruption using imageio.

    Args:
        video_path (str): Path to the video file.
        check_frames (bool): If True, checks the video frame by frame. Otherwise, only checks if the video can be loaded.

    Prints:
        The status of the video file (valid or corrupted).
    """
    try:
        # Attempt to read the video file
        reader = imageio.get_reader(video_path, 'ffmpeg')
        if check_frames:
            for _ in reader:  # Iterate through frames
                pass
        #print(f"Valid: {video_path}")
    except Exception as e:
        pass
        #print(f"Corrupted: {video_path} ({e})")

def check_videos_for_corruption(dataset_path: str, max_workers: int = 4, check_frames: bool = False) -> list:
    """
    Checks all video files in the dataset path (including subfolders) for corruption using imageio.
    This function is parallelized using ThreadPoolExecutor and displays a tqdm progress bar.

    Args:
        dataset_path (str): Path to the dataset root directory.
        max_workers (int): Maximum number of threads to use for parallel processing.
        check_frames (bool): If True, checks each video frame by frame. Otherwise, only checks if the video can be loaded.

    Returns:
        list: A list of corrupted video file paths.
    """
    video_paths = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(('.mp4', '.avi', '.mkv', '.mov')):  # Add other video extensions if needed
                video_paths.append(os.path.join(root, file))

    corrupted_files = []

    def process_video(video_path):
        try:
            # Attempt to read the video file
            reader = imageio.get_reader(video_path, 'ffmpeg')
            if check_frames:
                for _ in reader:  # Iterate through frames
                    pass
        except Exception as e:
            corrupted_files.append(video_path)

    with tqdm(total=len(video_paths), desc="Checking videos") as progress_bar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for _ in executor.map(process_video, video_paths):
                progress_bar.update(1)

    return corrupted_files