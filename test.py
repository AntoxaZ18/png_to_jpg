from PIL import Image
import shutil
import os
import cv2
from time import time
from concurrent.futures import ThreadPoolExecutor, wait, ProcessPoolExecutor
import argparse

TARGET_SIZE = (1440, 810)
QUALITY = 75


def convert_and_save_pillow(filepath: str, output_path: str):
    with Image.open(filepath) as img:
        if img.mode in ('RGBA', 'LA', 'P'):
            img = img.convert('RGB')
        img_resized = img.resize(TARGET_SIZE)

        img_resized.save(output_path, 'JPEG', quality=QUALITY)



def generate_png_dataset(filepath: str, out_folder: str, sample_size: int):
    for i in range(sample_size):
        shutil.copy(filepath, f"{out_folder}/{filepath.split('.')[0]}_{i}.png")



def convert_all_files(png_folder: str, jpg_folder: str, lib: str = 'opencv'):
    all_files = os.listdir(png_folder)

    convert_function = None

    if lib == 'opencv':
        convert_function = convert_and_save_cv2
    else:
        convert_function = convert_and_save_pillow

    for png_file in all_files:
        jpg_name = f"{png_file.split('.')[0]}.jpg"
        convert_function(f"{png_folder}/{png_file}", f"{jpg_folder}/{jpg_name}")


def convert_and_save_cv2(filepath: str, output_path: str):
    img = cv2.imread(filepath, cv2.IMREAD_COLOR)

    img_resized = cv2.resize(img, TARGET_SIZE)

    cv2.imwrite(output_path, img_resized, [int(cv2.IMWRITE_JPEG_QUALITY), QUALITY])

def convert_all_files_pooled(png_folder: str, jpg_folder: str, lib: str = 'opencv'):
    
    if lib == 'opencv':
        print("used opencv for convertion")
        convert_function = convert_and_save_cv2
    else:
        print("used pillow for convertion")
        convert_function = convert_and_save_pillow
    
    all_files = os.listdir(png_folder)
    futures = []
    with ThreadPoolExecutor() as executor:
        for png_file in all_files:
            jpg_name = f"{png_file.split('.')[0]}.jpg"
            futures.append(executor.submit(convert_function, f"{png_folder}/{png_file}", f"{jpg_folder}/{jpg_name}"))
        
        wait(futures)

    return len(futures)


def main(input_folder: str, output_folder: str, mode: str = "opencv", generate_files: bool = False):

    if generate_files:
        generate_png_dataset("image.png", input_folder, 100)

    start = time()
    
    total_converted = convert_all_files_pooled(input_folder, output_folder, lib=mode) 

    end = time()

    print(f"total: {end - start} s", f"per image: {(end - start) / total_converted:.3f} s")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-g", "--gen", help="generate 100 files", default=False)
    parser.add_argument("-i", "--input", help="input folder")
    parser.add_argument("-o", "--output", help="output folder")
    parser.add_argument("-m", "--mode", help="opencv vs pil", default="opencv")

    args = parser.parse_args()

    main(args.input, args.output, mode=args.mode, generate_files=args.gen)



