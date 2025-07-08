import argparse
import sys
import os
import toml
import torch
import os
from pathlib import Path
from datetime import datetime

# relative imports to root 
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, repo_root)
from AVDeepfake1Mpp.code.loaders import AVDeepfake1mDataModule, Metadata
from models.batfd.model import Batfd, BatfdPlus
from models.batfd.inference import inference_model
from models.batfd.post_process import post_process
from AVDeepfake1Mpp.code.utils import read_json

def infer(args):
    # Determine device
    if args["gpus"] > 0 and torch.cuda.is_available():
        device = f"cuda:{torch.cuda.current_device()}"
    else:
        device = "cpu"

    print(f"Using device: {device}")

    config = toml.load(args["config"])

    # Define paths relative to the specified output directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    temp_dir = os.path.join(args["output_dir"], config["name"], timestamp)
    checkpoint_name = os.path.splitext(os.path.basename(args["checkpoint"]))[0]
    output_file = os.path.join(temp_dir, f"{checkpoint_name}_{args['subset']}.json")

    model_type = config["model_type"]

    if model_type == "batfd_plus":
        model = BatfdPlus.load_from_checkpoint(args["checkpoint"])
    elif model_type == "batfd":
        model = Batfd.load_from_checkpoint(args["checkpoint"])
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    print("Loaded model!")
    model.eval()

    # Setup DataModule
    dm_dataset_name = config["dataset"]
    is_plusplus = dm_dataset_name == "avdeepfake1m++"

    dm = AVDeepfake1mDataModule(
        root=args["data_root"],
        temporal_size=config["num_frames"],
        max_duration=config["max_duration"],
        require_match_scores=False,
        batch_size=1, # due to the problem from lightning, only 1 is supported
        num_workers=args["num_workers"],
        get_meta_attr=model.get_meta_attr,
        return_file_name=True,
        is_plusplus=is_plusplus,
        test_subset=args["subset"] if args["subset"] in ("test", "testA", "testB") else None
    )
    print("Loaded DataModule!")
    dm.setup()

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    Path(temp_dir).mkdir(parents=True, exist_ok=True)

    if args["subset"] in ("test", "testA", "testB"):
        dataloader = dm.test_dataloader()
        metadata_path = os.path.join(dm.root, f"{args['subset']}_metadata.json")
    elif args["subset"] == "val":
        dataloader = dm.val_dataloader()
        metadata_path = os.path.join(dm.root, "val_metadata.json")
    else:
        raise ValueError("Invalid subset")

    if os.path.exists(metadata_path):
        metadata = [Metadata(**each) for each in read_json(metadata_path)]
    else:
        metadata = [
            Metadata(file=file_name, 
                     original=None,
                     split=args["subset"],
                     fake_segments=[],
                     fps=25,
                     visual_fake_segments=[],
                     audio_fake_segments=[],
                     audio_model="",
                     modify_type="",
                     # handle by the predictor in `inference_model`
                     video_frames=-1,
                     audio_frames=-1)
            for file_name in dataloader.dataset.file_list
        ]

    inference_model(
        model_name=config["name"], 
        model=model, 
        dataloader=dataloader,
        metadata=metadata,
        max_duration=config["max_duration"], 
        model_type=config["model_type"], 
        gpus=args["gpus"], 
        temp_dir=temp_dir
    )

    post_process(
        model_name=config["name"], 
        save_path=output_file,
        metadata=metadata, 
        fps=25, 
        alpha=config["soft_nms"]["alpha"], 
        t1=config["soft_nms"]["t1"], 
        t2=config["soft_nms"]["t2"], 
        dataset_name=dm_dataset_name,
        output_dir=temp_dir
    )

    print(f"Inference complete. Results saved to {output_file}")


# Update main to call infer(args)
def main():
    parser = argparse.ArgumentParser(description="BATFD/BATFD+ Inference")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the TOML configuration file.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the model checkpoint.")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory of the dataset.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the output files.")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of workers for data loading.")
    parser.add_argument("--subset", type=str, choices=["val", "test", "testA", "testB"], 
                        default="test", help="Dataset subset.")
    parser.add_argument("--gpus", type=int, default=1,
                        help="Number of GPUs. Set to 0 for CPU.")

    args = parser.parse_args()
    infer(vars(args))


if __name__ == '__main__':
    main()
