from os import makedirs, path, listdir, remove
from shutil import move, rmtree
from subprocess import run, CalledProcessError
from time import time
from zipfile import ZipFile

output_dir = "./bright_dataset"
makedirs(output_dir, exist_ok=True)


def run_command(command_list):
    try:
        run(command_list, check=True)
        print(f"Command executed successfully: {' '.join(command_list)}")
    except CalledProcessError as e:
        print(f"Error executing: {' '.join(command_list)}")
        raise e


urls = [
    "https://huggingface.co/datasets/Kullervo/BRIGHT/resolve/main/pre-event.zip",
    "https://huggingface.co/datasets/Kullervo/BRIGHT/resolve/main/post-event.zip",
    "https://huggingface.co/datasets/Kullervo/BRIGHT/resolve/main/target.zip",
]

if __name__ == "__main__":
    print("Starting dataset download...")
    start_time = time()
    zip_names = list()

    for url in urls:
        filename = path.basename(url)
        print(f"Downloading {filename}...")
        run_command([
            "aria2c",
            "-x", "16", "-s", "16", "-k", "1M",
            "-d", output_dir,
            "-o", filename,
            url
        ])
        zip_names.append(filename)
        print(f"Finished downloading {filename}")

    print("\nStarting dataset extraction...")
    for zip_name in zip_names:
        zip_path = path.join(output_dir, zip_name)
        target_folder_name = zip_name.replace(".zip", "")
        final_folder_path = path.join(output_dir, target_folder_name)
        temp_extract_path = path.join(output_dir, f"temp_{target_folder_name}")

        if path.exists(zip_path):
            print(f"Extracting {zip_name}...")

            with ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(temp_extract_path)

            makedirs(final_folder_path, exist_ok=True)
            extracted_items = listdir(temp_extract_path)

            if len(extracted_items) == 1 and path.isdir(path.join(temp_extract_path, extracted_items[0])):
                print("Detected single top-level folder in zip, moving its contents up.")

                inner_folder = path.join(temp_extract_path, extracted_items[0])
                for item in listdir(inner_folder):
                    move(path.join(inner_folder, item), final_folder_path)

                print(f"Removed inner folder: {inner_folder}")
            else:
                print("Multiple items detected, moving all to final folder.")

                for item in extracted_items:
                    move(path.join(temp_extract_path, item), final_folder_path)

                print("All items moved.")

            print("Cleaning up temporary files...")
            rmtree(temp_extract_path)
            remove(zip_path)
            print("Cleanup done.")
            print(f"Finished extracting {zip_name}")
        else:
            print(f"Warning: {zip_path} not found for extraction.")

    elapsed_time = time() - start_time
    print(f"Dataset downloaded and extracted in {elapsed_time:.2f} seconds")
