import os
import random
import csv
import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image

######
# Set up.
######
# Parameters.
BASE_PATH = "finetune"
OUTPUT_PATH = "generated_SD"
CSV_FILENAME = "log_SD.csv"
DEVICE = "cuda"
MODEL_PATH = "/scratch/vvnmax002/checkpoints/SD_full"

# Global variables.
PATHOLOGIES_LIST = [
    'joint effusion', 
    'lateral epicondyle displaced', 
    'soft tissue swelling', 
    'supracondylar fracture'
]
MASK_AREAS = ['joint', 'upper', 'whole']
VIEWS = ['AP', 'LAT']

# Set up the pipeline.
print(f"====== Loading model from {MODEL_PATH} ======")
pipe = AutoPipelineForInpainting.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16
).to(DEVICE)

# Logging CSV: Number, ID, View, Pathology 1, Pathology 2, ..., Normal.
csv_header = ['Number', 'ID', 'View'] + PATHOLOGIES_LIST + ['Normal']
csv_file = open(CSV_FILENAME, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(csv_header)
global_counter = 0

######
# Supporting functions.
######
def log_to_csv(filepath, view, active_pathologies):
    """
    Logs the generated image and its info to the CSV.
    active_pathologies: List of strings of pathologies present in the image.
    """
    global global_counter # Use outer declaration.
    global_counter += 1
    
    # Create a row.
    row = [global_counter, filepath, view]
    
    # 1 => active pathology, 0 => this pathology is inactive.
    is_normal = True
    for p in PATHOLOGIES_LIST:
        if p in active_pathologies:
            row.append(1)
            is_normal = False   # If we have a pathology, it is not 'normal'.
        else:
            row.append(0)
    
    # Finally, the normal column.
    row.append(1 if is_normal else 0)
    csv_writer.writerow(row)

def get_view_text(view_code):
    """
    Simple function used to avoid boilerplate code - primarily for constructing the prompt.
    """
    return "An anteroposterior" if view_code == "AP" else "A lateral"

def genImg(prompt, base_image, mask_image, save_folder, filename):
    """
    Generate and save the image using the pipeline.
    """
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, filename)

    result = pipe(
        prompt=prompt,
        image=base_image,
        mask_image=mask_image,
        height=1024,
        width=1024,
        num_inference_steps=30
    ).images[0]

    result.save(save_path)

######
# Step 1: Generate single pathology images only.
######
print("\n====== Starting single pathology generation ======")

for view in VIEWS: # AP, LAT.
    for mask_area in MASK_AREAS: # joint, upper, whole.
        # e.g: finetune/normal/AP/joint/.
        source_dir = os.path.join(BASE_PATH, "normal", view, mask_area)
        
        # Ignore if not valid - safety check.
        if not os.path.exists(source_dir):
            print(f"Skipping {source_dir}, not found!")
            continue
            
        all_files = [f for f in os.listdir(source_dir) if f.endswith('.jpg')] # Note, jpg => radiographs, png => masks, txt => label.
        total_available = len(all_files)
        
        if total_available == 0: 
            continue

        print(f"Processing {view}/{mask_area} ({total_available} base images available)")

        # Generate 500 images per mask area (for each pathology).
        for pathology in PATHOLOGIES_LIST:
            # e.g: generated/joint effusion/AP/
            dest_folder = os.path.join(OUTPUT_PATH, pathology, view)
            
            for i in range(500):
                # Sample sequentially to ensure that each base image is used at least once, then randomly.
                if i < total_available:
                    base_name = all_files[i]
                else:
                    base_name = random.choice(all_files)

                # Load base image and its respective mask.
                image_path = os.path.join(source_dir, base_name)
                mask_name = base_name.replace(".jpg", "-masklabel.png") # 123456_AP.jpg => 123456_AP-masklabel.png.
                mask_path = os.path.join(source_dir, mask_name) # Mask should always be in the same folder as the base image.

                if not os.path.exists(mask_path):
                    print(f"Mask not found for an image! {mask_path}.")
                    continue

                base_image = load_image(image_path)
                mask_image = load_image(mask_path)
                
                # Construct the prompt.
                view_text = get_view_text(view)
                prompt = f"{view_text} radiograph of an elbow displaying {pathology}"
                
                # E.g: 123456_AP_joint_1.jpg added index here to avoid replacing images.
                save_name = f"{base_name.replace('.jpg', '')}_{mask_area}_{i}.jpg"
                
                genImg(prompt, base_image, mask_image, dest_folder, save_name)
                log_to_csv(os.path.join(dest_folder[13:], save_name), view, [pathology])

######
# Step 2: Generate normal images.
######
print("\n====== Starting Normal Generation ======")

for pathology in PATHOLOGIES_LIST:
    patho_base_path = os.path.join(BASE_PATH, pathology)
    
    if not os.path.exists(patho_base_path): 
        print(f"Skipping {patho_base_path}, not found!")
        continue
        
    for view in VIEWS:
        # E.g.: finetune/joint effusion/AP/
        curr_dir = os.path.join(patho_base_path, view)
        if not os.path.exists(curr_dir):
            print(f"Skipping {curr_dir}, not found!")
            continue

        files = [f for f in os.listdir(curr_dir) if f.endswith('.jpg')]

        # E.g.: generated/normal/AP/
        dest_folder = os.path.join(OUTPUT_PATH, "normal", view)

        for base_name in files:
            image_path = os.path.join(curr_dir, base_name)
            mask_name = base_name.replace(".jpg", "-masklabel.png")
            mask_path = os.path.join(curr_dir, mask_name)

            if not os.path.exists(mask_path): 
                print(f"Mask not found for an image! {mask_path}.")
                continue

            base_image = load_image(image_path)
            mask_image = load_image(mask_path)

            view_text = get_view_text(view)
            prompt = f"{view_text} radiograph of a healthy elbow"

            # E.g.: 123456_AP_normal.jpg.
            save_name = f"{base_name.replace('.jpg', '')}_normal.jpg"
            
            genImg(prompt, base_image, mask_image, dest_folder, save_name)
            log_to_csv(os.path.join(dest_folder[13:], save_name), view, []) # See above, empty list => normal image.

######
# Step 3: Multi-pathology generation. Note, uses whole mask images only!
######
print("\n====== Starting multi-pathology generation ======")
counts_to_generate = [2, 3, 4] # 2 pathologies, 3 pathologies, 4 pathologies.
images_per_group = 1000 # 1000 images each (500 AP:500 LAT).

for num_pathos in counts_to_generate:
    print(f"Generating images with {num_pathos} pathologies...")
    for view in VIEWS:
        # E.g.: finetune/normal/AP/whole
        source_dir = os.path.join(BASE_PATH, "normal", view, "whole")
        
        if not os.path.exists(source_dir):
            print(f"Skipping {source_dir}, not found!")
            continue
            
        all_files = [f for f in os.listdir(source_dir) if f.endswith('.jpg')]
            
        # E.g.: generated/multi-pathology/AP/, note that the CSV should be used to parse these images!
        dest_folder = os.path.join(OUTPUT_PATH, "multi-pathology", view)

        # 500 per view.
        for i in range(500):
            base_name = random.choice(all_files) # Randomly select base image.
            
            image_path = os.path.join(source_dir, base_name)
            mask_name = base_name.replace(".jpg", "-masklabel.png")
            mask_path = os.path.join(source_dir, mask_name)
            
            if not os.path.exists(mask_path): 
                print(f"Mask not found for an image! {mask_path}.")
                continue

            base_image = load_image(image_path)
            mask_image = load_image(mask_path)

            # Randomly select n pathologies from the list.
            selected_pathos = random.sample(PATHOLOGIES_LIST, k=num_pathos)
            
            # Use ...displaying joint effusion, soft tissue swelling, ... for prompt.
            patho_str = ", ".join(selected_pathos)
            view_text = get_view_text(view)
            prompt = f"{view_text} radiograph of an elbow displaying {patho_str}"
            
            # Filename: ID_mixed_count_index.jpg
            save_name = f"{base_name.replace('.jpg', '')}_mixed{num_pathos}_{i}.jpg"
            
            genImg(prompt, base_image, mask_image, dest_folder, save_name)
            log_to_csv(os.path.join(dest_folder[13:], save_name), view, selected_pathos)

# Clean up.
csv_file.close()
print(f"====== Processing Complete. Log saved to {CSV_FILENAME} ======")
