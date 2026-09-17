import os

# Dataset paths
dataset_path = r"D:\minorproject\archive\Lung Segmentation"

image_path = os.path.join(dataset_path, "CXR_png")
mask_path = os.path.join(dataset_path, "masks")


# Get all X-ray images
images = [
    file for file in os.listdir(image_path)
    if file.endswith(".png")
]

# Get all mask images
masks = [
    file for file in os.listdir(mask_path)
    if file.endswith(".png")
]


# Display total count
print("Total X-ray Images:", len(images))
print("Total Masks:", len(masks))


# Find matching image-mask pairs
matching_pairs = []

for image in images:

    # Example:
    # CHNCXR_0001_0.png
    #
    # Corresponding mask:
    # CHNCXR_0001_0_mask.png

    mask_name = image.replace(".png", "_mask.png")

    if mask_name in masks:
        matching_pairs.append((image, mask_name))


print("Matching Image-Mask Pairs:", len(matching_pairs))


# Display first 10 pairs
print("\nFirst 10 Matching Pairs:\n")

for image, mask in matching_pairs[:10]:
    print("Image:", image)
    print("Mask :", mask)
    print("-" * 40)



    