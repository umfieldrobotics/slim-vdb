import cv2
import os
import re

# Input directory containing the images
# input_directory = '/home/yuzhen/Desktop/first/Azure-Kinect-Samples/opencv-kinfu-samples/build/kinfu_tag'

# input_directory = '/home/yuzhen/SegFormer-tensorrt/result/kinfu_tag'
input_directory = "/home/yuzhen/Desktop/semanticVDB/vdbfusion/examples/cpp/jpg_folder"
# Output video path and filename
# output_video_path = '/home/yuzhen/SegFormer-tensorrt/video_result/kinfu_segformer_tag_1*1_change_material.mp4'
output_video_path = "/home/yuzhen/Desktop/semanticVDB/vdbfusion/examples/cpp/jpg_folder/output0_49.mp4"

# Get the list of image files in the input directory
file_list = os.listdir(input_directory)
image_files = [file for file in file_list if file.endswith('.jpg')]




# # Sort the image files in alphanumeric order
# image_files.sort()



# Sort the image files based on the numerical index in the filename
image_files.sort(key=lambda x: int(re.search(r'\d+', x).group()))

# Print the sorted image filenames
for image_file in image_files:
    print(image_file)




num_images = len(image_files)
print(f"Number of images: {num_images}")





# Read the first image to obtain the width and height
first_image_path = os.path.join(input_directory, image_files[0])
first_image = cv2.imread(first_image_path)
height, width, _ = first_image.shape

# Define the video codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use appropriate codec for your desired video format
fps = 10 # Set the frames per second
output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Iterate over the image files and write each frame to the video
for image_file in image_files:
    image_path = os.path.join(input_directory, image_file)
    frame = cv2.imread(image_path)
    output_video.write(frame)

# Release the VideoWriter and close the video file
output_video.release()

print(f"Video saved at: {output_video_path}")


 
