from cog import BasePredictor, Input, Path
import sys
import os
from pathlib import Path as PathlibPath
import zipfile
import shutil

from gyumin_kohya.train_network import NetworkTrainer
from gyumin_kohya.train_network import setup_parser
from argparse import Namespace

DEFAULT_MODEL = "digiplay/Photon_v1"
DEFAULT_MODEL_PATH = "/src/photon-cache"

class Predictor(BasePredictor):
  def unzip_images(self, images_zip, steps_per_img, identifier, class_word):
    with zipfile.ZipFile(str(images_zip), "r") as zip_ref:
      for file_name in zip_ref.namelist():
        if not file_name.startswith("inputs"):
          continue
        zip_ref.extract(file_name, "./images")

    if (len(class_word.strip().split(" ")) > 1 or class_word.strip() == ""):
      raise ValueError("class_word cannot contain blanks")
    if (len(identifier.strip().split(" ")) > 1 or identifier.strip() == ""):
      raise ValueError("identifier cannot contain blanks")

    # rename the extracted folder
    os.rename("./images/inputs",
              f"./images/{steps_per_img}_{identifier} {class_word}")

  def setup(self):
    """Load the model into memory to make running multiple predictions efficient"""
    PathlibPath.mkdir(PathlibPath("./images"), exist_ok=True)
    PathlibPath.mkdir(PathlibPath("./outputs"), exist_ok=True)
    print("Setting up the model...")

  # The arguments and types the model takes as input
  def predict(self,
              pretrained_model: str = Input(
                  description="Huggingface model to use for training. Use default value for faster training.",
                  default=DEFAULT_MODEL,
              ),
              identifier: str = Input(
                  description="A unique identifier of target concept aka trigger words. Must exclude blanks.",
                  default="Jason",
                  min_length=1,
                  max_length=30,
              ),
              class_word: str = Input(
                  description="The class of target concept. Must exclude blanks.",
                  default="man",
                  min_length=1,
                  max_length=30,
              ),
              steps_per_img: int = Input(
                  description="Number of steps per image",
                  default=150,
                  le=500,
                  ge=50,
              ),
              batch_size: int = Input(
                  description="Batch size(number of steps that will be processed at once).",
                  default=8,
                  le=16,
                  ge=1,
              ),
              lora_dim: int = Input(
                  description="The dimension of the LoRA network",
                  default=8,
                  le=128,
                  ge=1,
              ),
              lora_alpha: int = Input(
                  description="The alpha of the LoRA network",
                  default=4,
                  le=128,
                  ge=1,
              ),
              images_zip: Path = Input(
                  description="A zipped folder named 'inputs' containing images to predict on: inputs.zip>inputs>image1.jpg, image2.jpg, ...",
              )
              ) -> Path:
    """Run a single prediction on the model"""
    # Clean up the directories
    shutil.rmtree("./images")
    shutil.rmtree("./outputs")
    PathlibPath.mkdir(PathlibPath("./images"), exist_ok=True)
    PathlibPath.mkdir(PathlibPath("./outputs"), exist_ok=True)

    # Unzip the images
    print("Unzipping images...")
    self.unzip_images(images_zip, steps_per_img, identifier, class_word)

    num_of_images = len([x for x in os.listdir(
        f'./images/{steps_per_img}_{identifier} {class_word}')
        if x.endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp"))])
    batch_count = num_of_images * steps_per_img // batch_size

    # Run the training
    print("Running training...")
    print("Configs:")
    print(f"\t-Pretrained_model: {pretrained_model}")
    print(f"\t-Identifier: {identifier}")
    print(f"\t-Class: {class_word}")
    print(f"\t-Images: {num_of_images}")
    print(f"\t-Steps per image: {steps_per_img}")
    print(
        f"\t-Total steps: {num_of_images} x {steps_per_img} = {steps_per_img * num_of_images}")
    print(f"\t-Batch size: {batch_size}")
    print(
        f"\t-Batch count: {steps_per_img * num_of_images} / {batch_size} = {batch_count}")

    print("=====================================")
    train_args = {
        "num_cpu_threads_per_process": 1,
        "xformers": True,
        "pretrained_model_name_or_path": pretrained_model if pretrained_model != DEFAULT_MODEL else DEFAULT_MODEL_PATH,
        "train_data_dir": "./images",
        "output_dir": "./outputs",
        "output_name": "output",
        "enable_bucket": True,
        "min_bucket_reso": 256,
        "max_bucket_reso": 2048,
        "save_model_as": "safetensors",
        "prior_loss_weight": 1.0,
        "learning_rate": 1e-4,
        "lr_scheduler": "cosine",
        "lr_warmup_steps": 420,
        "optimizer_type": "AdamW8bit",
        "resolution": "512",
        "max_train_steps": 4000,
        "max_train_epochs": 1,
        "train_batch_size": batch_size,
        "mixed_precision": "fp16",
        "cache_latents": True,
        "save_every_n_epochs": 1,
        "network_module": "networks.lora",
        "network_dim": lora_dim,
        "network_alpha": lora_alpha,
        "debug_dataset": False,
        "huggingface_token": "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"
    }

    parser = setup_parser()
    args, _ = parser.parse_known_args(namespace=Namespace(**train_args))

    # Run the training
    trainer = NetworkTrainer()
    trainer.train(args)
    print("=====================================")
    print("Training done!")

    # Zip the output
    print("Zipping output...")
    with zipfile.ZipFile("./output.zip", "w") as zip_ref:
      zip_ref.write("./outputs/output.safetensors")

    return Path("./output.zip")
