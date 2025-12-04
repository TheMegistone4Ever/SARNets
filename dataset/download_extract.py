import os
import subprocess
import time
import zipfile

output_dir = "./bright_dataset"
os.makedirs(output_dir, exist_ok=True)

start_time = time.time()


def run_command(command_list):
    try:
        subprocess.run(command_list, check=True)
        print(f"Command executed successfully: {' '.join(command_list)}")
    except subprocess.CalledProcessError as e:
        print(f"Error executing: {' '.join(command_list)}")
        raise e


urls = [
    "https://huggingface.co/datasets/Kullervo/BRIGHT/resolve/main/pre-event.zip",
    "https://huggingface.co/datasets/Kullervo/BRIGHT/resolve/main/post-event.zip",
    "https://huggingface.co/datasets/Kullervo/BRIGHT/resolve/main/target.zip",
]

print("Starting dataset download...")
for url in urls:
    filename = os.path.basename(url)
    print(f"Downloading {filename}...")
    run_command([
        "aria2c",
        "-x", "16", "-s", "16", "-k", "1M",
        "-d", output_dir,
        "-o", filename,
        url
    ])
    print(f"Finished downloading {filename}")

print("\nStarting dataset extraction...")
for zip_name in ["pre-event.zip", "post-event.zip", "target.zip"]:
    zip_path = os.path.join(output_dir, zip_name)
    if os.path.exists(zip_path):
        print(f"Extracting {zip_name}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(output_dir)
        print(f"Finished extracting {zip_name}")
    else:
        print(f"Warning: {zip_path} not found for extraction.")

elapsed_time = time.time() - start_time
print(f"Dataset downloaded and extracted in {elapsed_time:.2f} seconds")
