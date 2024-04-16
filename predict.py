from cog import BasePredictor, Input, Path, File
import sys
import os
from pathlib import Path as PathlibPath
import subprocess
import zipfile
import shutil


class Predictor(BasePredictor):
  def unzip_images(self, images_zip, steps_per_img, identifier, class_word):
    with zipfile.ZipFile(images_zip, "r") as zip_ref:
      for file_name in zip_ref.namelist():
        if not file_name == "inputs":
          continue
        zip_ref.extract(file_name, "./images")

    if (len(class_word.strip().split(" ")) > 1 or class_word.strip() == ""):
      raise ValueError("class_word cannot contain blanks")
    if (len(identifier.strip().split(" ")) > 1 or identifier.strip() == ""):
      raise ValueError("identifier cannot contain blanks")

    # rename the extracted folder
    os.rename("./images/inputs",
              f"./images/{steps_per_img}_{identifier} {class_word}")

  def run_command(command):
    # Start the subprocess
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)

    # Read and display the output line by line
    for line in iter(process.stdout.readline, ''):
      # Use sys.stdout.write to display the output in Jupyter Notebook
      sys.stdout.write(line)
      sys.stdout.flush()

    process.stdout.close()
    return_code = process.wait()
    if return_code:
      raise subprocess.CalledProcessError(return_code, command)

  def setup(self):
    """Load the model into memory to make running multiple predictions efficient"""
    PathlibPath.mkdir(PathlibPath("./images"), exist_ok=True)
    PathlibPath.mkdir(PathlibPath("./outputs"), exist_ok=True)

  # The arguments and types the model takes as input
  def predict(self,
              pretrained_model: str = Input(
                  description="Huggingface model to use for training",
                  default="digiplay/Photon_v1",
              ),
              identifier: str = Input(
                  description="A unique identifier of target concept aka trigger words",
                  default="Jason",
                  min_length=1,
                  max_length=30,
              ),
              class_word: str = Input(
                  description="The class of target concept",
                  default="man",
                  min_length=1,
                  max_length=30,
              ),
              steps_per_img: int = Input(
                  description="Number of steps per image",
                  default=150,
                  le=50,
                  ge=500,
              ),
              batch_size: int = Input(
                  description="Batch size(number of steps that will be processed at once). Strongly recommend to keep it as-is",
                  default=8,
                  le=1,
                  ge=16,
              ),
              lora_dim: int = Input(
                  description="The dimension of the LoRA network",
                  default=8,
                  le=1,
                  ge=128,
              ),
              lora_alpha: int = Input(
                  description="The alpha of the LoRA network",
                  default=4,
                  le=1,
                  ge=128,
              ),
              learning_rate: float = Input(
                  description="The learning rate of the model",
                  default=1e-4,
                  le=1e-6,
                  ge=1e-2,
              ),
              images_zip: File = Input(
                  description="A zip file containing images to predict on"
              )
              ) -> Path:
    """Run a single prediction on the model"""
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
    print(f"\t-Steps per image: {steps_per_img}")
    print(
        f"\t-Total steps: {num_of_images} x {steps_per_img} = {steps_per_img * num_of_images}")
    print(f"\t-Batch size: {batch_size}")
    print(
        f"\t-Batch count: {steps_per_img * num_of_images} / {batch_size} = {batch_count}")
    command = [
        "accelerate", "launch",
        "--config_file", "./accelerate.yaml",
        "--num_cpu_threads_per_process", "1",
        "sd-scripts/train_network.py",
        "--xformers",
        "--pretrained_model_name_or_path", pretrained_model,
        "--train_data_dir", "./images",
        "--output_dir", "./outputs",
        "--output_name", "output",
        "--min_bucket_reso", "256",
        "--max_bucket_reso", "2048",
        "--save_model_as", "safetensors",
        "--prior_loss_weight", "1.0",
        "--learning_rate", learning_rate,
        "--lr_scheduler", "cosine",
        "--lr_warmup_steps", "420",
        "--optimizer_type", "AdamW8bit",
        "--resolution", "512",
        "--max_train_steps", "6000",
        "--max_train_epochs", "1",
        "--train_batch_size", batch_size,
        "--mixed_precision", "fp16",
        "--cache_latents",
        "--save_every_n_epochs", "1",
        "--network_module", "networks.lora",
        "--network_dim", lora_dim,
        "--network_alpha", lora_alpha,
    ]
    self.run_command(command)

    # Zip the output
    print("Zipping output...")
    with zipfile.ZipFile("./output.zip", "w") as zip_ref:
      zip_ref.write("./outputs/output.safetensors")

    # Empty the directories
    shutil.rmtree("./images")
    shutil.rmtree("./outputs")
    PathlibPath.mkdir(PathlibPath("./images"), exist_ok=True)
    PathlibPath.mkdir(PathlibPath("./outputs"), exist_ok=True)

    return Path("./output.zip")
