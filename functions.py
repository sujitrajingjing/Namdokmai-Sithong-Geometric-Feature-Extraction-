import numpy as np
import cv2
import os
import csv
from PIL import Image

# Define the colors
global color_red
color_red = (0, 0, 255)

global color_green
color_green = (0, 255, 0)

global color_blue
color_blue = (255, 0, 0)

# Input inversed mask and get the centroid
def getCentroidFromMoments(mask, top_left, bottom_right):
    x1, y1 = top_left
    x2, y2 = bottom_right

    # Get the region of interest from the mask
    roi = mask[y1:y2+1, x1:x2+1]

    # Calculate moments for the region of interest
    moments = cv2.moments(roi)

    # Calculate Centroid
    if moments['m00'] != 0:  # Avoid division by zero
        cx_roi = int(moments['m10'] / moments['m00'])
        cy_roi = int(moments['m01'] / moments['m00'])
        cx = cx_roi + x1
        cy = cy_roi + y1
    else:
        cx, cy = 0, 0  # Handle an empty mask
        
    return cx, cy

# Function to get inverse mask of an image
def getInvMask(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Define range for white (experiment with these ranges)
    lower_white = np.array([0, 0, 200])  # Lower: High value, low saturation
    upper_white = np.array([220, 20, 255]) # Upper: Allow some hue variation
    # Create a mask for the mango
    mask = cv2.inRange(hsv_image, lower_white, upper_white)
    # Invert the mask and combine
    inverse_mask = cv2.bitwise_not(mask)
    # define the kernel for Morphological operation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    morph1 = cv2.morphologyEx(inverse_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    morph = cv2.morphologyEx(morph1, cv2.MORPH_CLOSE, kernel, iterations=1)
    return morph

# save data to CSV file
def save_to_csv(filename, headers, data):
    with open(filename, mode='w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        writer.writerows(data)

# Automatically find and draw the major axis, perpendicular line 1, perpendicular line 2
def draw_lines_from_point(img, start_point, edges, topleft, bottomright):
    # Create a copy of the image to avoid modifying the original image.
    img_copy = img.copy()
    # Define colors
    color_red = (0, 0, 255)
    color_green = (255, 0, 0)
    color_blue = (0, 255, 0)
    # Degree of increment
    n = 0.25
    # Initialize variables to track the longest red line
    longest_red_line = ((0,0),(0,0))
    longest_red_line_length = 0.0
    longest_red_line_direction = 0.0
    # Iterate over all directions in n-degree increments.
    for direction in np.arange(0.0, 360.0, n):
        # Initialize the end point as None.
        end_point = None
        # Slowly increasing the length and stop when it hits the edge pixel
        # Starting from 100 pixel to speed up time
        for i in range(1, 3000):
            # Calculate the potential end point.
            # Before you delete the second variable because you think it is redundant, it is there because sometime int() and round() give different output and it will cause error.
            x = start_point[0] + int(np.cos(direction * np.pi / 180) * i)
            y = start_point[1] + int(np.sin(direction * np.pi / 180) * i)
            x2 = start_point[0] + int(round(np.cos(direction * np.pi / 180) * i))
            y2 = start_point[1] + int(round(np.sin(direction * np.pi / 180) * i))
            # Check if the potential end point is within the image boundaries.
            # if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            if topleft[0] <= x < bottomright[0] and topleft[1] <= y < bottomright[1]:
                # Check if the potential end point is on an edge pixel.
                if edges[y, x] == 255 or edges[y2, x2] == 255:
                    end_point = (x, y)
                    break  # Stop the line when it hits an edge pixel.
            else:
                break  # Stop the line if it goes out of the image bounds.

            
        if end_point is not None:
            line_length = np.sqrt((end_point[0] - start_point[0])**2 + (end_point[1] - start_point[1])**2)

            # Update the longest red line if the current line is longer
            if line_length > longest_red_line_length:
                longest_red_line = (start_point, end_point)
                longest_red_line_length = line_length
                longest_red_line_direction = direction

    # DRAWING PERPENDICULAR LINES
    # Declaring all variables we need later
    # Declare and init direction
    perpendicular_direction = longest_red_line_direction + 90.0
    # Declare and init longest perpendicular lines
    perpendicular_line1_longest = ((0,0),(0,0))
    perpendicular_line2_longest = ((0,0),(0,0))
    # Declare and init longest perpendicular lines length
    perpendicular_line1_longest_length = 0.0
    perpendicular_line2_longest_length = 0.0
    # Declare the endpoints

    
    # Remake the longest line
    for i in range(1, int(longest_red_line_length)):
            end_point1 = None
            end_point2 = None
            # Calculate the start point.
            x = longest_red_line[0][0] + int(np.cos(longest_red_line_direction * np.pi / 180) * i)
            y = longest_red_line[0][1] + int(np.sin(longest_red_line_direction * np.pi / 180) * i)
            # Calculate end point for line 1
            for j in range(1, 1500):
                # Keep increasing the length till it hits the edge. Before you delete the second variable because you think it is redundant, it is there because sometime int() and round() give different output and it will cause error.
                # line_length = np.sqrt((end_point[0] - start_point[0])**2 + (end_point[1] - start_point[1])**2)
                perpendicular_line1_end = [x + int(np.cos(perpendicular_direction * np.pi / 180) * j), y + int(np.sin(perpendicular_direction * np.pi / 180) * j)]
                perpendicular_line1_end2 = [x + int(round(np.cos(perpendicular_direction * np.pi / 180) * j)), y + int(round(np.sin(perpendicular_direction * np.pi / 180) * j))]
                # Check if image is out of bound
                # if 0 <= perpendicular_line1_end[0] < img.shape[1] and 0 <= perpendicular_line1_end[1] < img.shape[0]:
                if topleft[0] <= perpendicular_line1_end[0] < bottomright[0] and topleft[1] <= perpendicular_line1_end[1] < bottomright[1]:
                    # Check if the potential end point is on an edge pixel.
                    if edges[perpendicular_line1_end[1], perpendicular_line1_end[0]] == 255 or edges[perpendicular_line1_end2[1], perpendicular_line1_end2[0]] == 255:
                        end_point1 = (perpendicular_line1_end[0], perpendicular_line1_end[1])
                        break  # Stop the line when it hits an edge pixel.
                else:
                    break  # Stop the line if it goes out of the image bounds.
            
            # Calculate end point for line 2
            for k in range(1, 1500):
                # Keep increasing the length till it hits the edge. Before you delete the second variable because you think it is redundant, it is there because sometime int() and round() give different output and it will cause error.
                perpendicular_line2_end = [x - int(np.cos(perpendicular_direction * np.pi / 180) * k), y - int(np.sin(perpendicular_direction * np.pi / 180) * k)]
                perpendicular_line2_end2 = [x - int(round(np.cos(perpendicular_direction * np.pi / 180) * k)), y - int(round(np.sin(perpendicular_direction * np.pi / 180) * k))]
                # Check if image is out of bound
                if topleft[0] <= perpendicular_line2_end[0] < bottomright[0] and topleft[1] <= perpendicular_line2_end[1] < bottomright[1]:
                    # Check if the potential end point is on an edge pixel.
                    if edges[perpendicular_line2_end[1], perpendicular_line2_end[0]] == 255 or edges[perpendicular_line2_end2[1], perpendicular_line2_end2[0]] == 255:
                        end_point2 = (perpendicular_line2_end[0], perpendicular_line2_end[1])
                        break  # Stop the line when it hits an edge pixel.
                else:
                    break  # Stop the line if it goes out of the image bounds.

            if end_point1 is not None:
            # Calculate the length of the red line
                perpendicular_line1_length = np.sqrt((end_point1[0] - x)**2 + (end_point1[1] - y)**2)
                # Update the erpendicular_line1_longest if the current line is longer
                if perpendicular_line1_length > perpendicular_line1_longest_length:
                    perpendicular_line1_longest = ((x,y),end_point1)
                    perpendicular_line1_longest_length = perpendicular_line1_length
            
            if end_point2 is not None:
            # Calculate the length of the red line
                perpendicular_line2_length = np.sqrt((end_point2[0] - x)**2 + (end_point2[1] - y)**2) 
                # Update the erpendicular_line2_longest if the current line is longer
                if perpendicular_line2_length > perpendicular_line2_longest_length:
                    perpendicular_line2_longest = ((x,y), end_point2)
                    perpendicular_line2_longest_length = perpendicular_line2_length
                
    # Print the length of the lines
    print("Longest line length: "+str(round(longest_red_line_length,2)) + " pixels")
    print("Perpendicular line1 length: "+str(round(perpendicular_line1_longest_length,2)) + " pixels")
    print("Perpendicular line2 length: "+str(round(perpendicular_line2_longest_length,2)) + " pixels")

    # Prepare data for CSV
    data = {'Longest Line': longest_red_line_length,
        'Perpendicular Line 1': perpendicular_line1_longest_length,
        'Perpendicular Line 2': perpendicular_line2_longest_length}

    # Draw all lines
    thickness = 1
    img_copy = cv2.line(img_copy, longest_red_line[0], longest_red_line[1], color_red, thickness)
    img_copy = cv2.line(img_copy, perpendicular_line1_longest[0], perpendicular_line1_longest[1], color_green, thickness)
    img_copy = cv2.line(img_copy, perpendicular_line2_longest[0], perpendicular_line2_longest[1], color_blue, thickness)
    # # Label the line length
    # font = cv2.FONT_HERSHEY_DUPLEX
    # font_scale = 0.6
    # thick = 1
    # img_copy = cv2.putText(img_copy, "Major axis: "+str(round(longest_red_line_length,2))+ " pixels", longest_red_line[1], font, font_scale, color_red, thick)
    # img_copy = cv2.putText(img_copy, "Minor axis1: "+str(round(perpendicular_line1_longest_length,2))+ " pixels", perpendicular_line1_longest[1], font, font_scale, color_green, thick)
    # img_copy = cv2.putText(img_copy, "Minor axis2: "+str(round(perpendicular_line2_longest_length,2))+ " pixels", perpendicular_line2_longest[1], font, font_scale, color_blue, thick)
    return img_copy, data, longest_red_line[1], perpendicular_line1_longest[1], perpendicular_line2_longest[1]

# Function for getting the overlap area of 2 bounding boxes
def get_overlapping_area(bbox1, bbox2):
  # Get the coordinates of the overlapping rectangle
  x_left = max(bbox1[0], bbox2[0])
  y_top = max(bbox1[1], bbox2[1])
  x_right = min(bbox1[2], bbox2[2])
  y_bottom = min(bbox1[3], bbox2[3])

  # Check if there is any overlap
  if x_right < x_left or y_bottom < y_top:
    return None  # No overlap

  return (x_left, y_top, x_right, y_bottom)

# Function for getting the overlap area of 2 bounding boxes
def get_largest_bb(bbox1, bbox2):
  # Get the coordinates of the overlapping rectangle
  x_left = min(bbox1[0], bbox2[0])
  y_top = min(bbox1[1], bbox2[1])
  x_right = max(bbox1[2], bbox2[2])
  y_bottom = max(bbox1[3], bbox2[3])

  return (x_left, y_top, x_right, y_bottom)

# Function for drawing bounding boxes
def draw_bounding_boxes(img, bbox1, bbox2, bbox_overlap):

    image = img.copy()
    # Convert the float to int
    # (X,Y)
    box1_start = (int(bbox1[0]), int(bbox1[1]))
    box1_end = (int(bbox1[2]), int(bbox1[3]))
    box2_start = (int(bbox2[0]), int(bbox2[1]))
    box2_end = (int(bbox2[2]), int(bbox2[3]))
    box_overlap_start = (int(bbox_overlap[0]), int(bbox_overlap[1]))
    box_overlap_end = (int(bbox_overlap[2]), int(bbox_overlap[3]))
    
    # Draw bounding box 1
    cv2.rectangle(image, box1_start, box1_end, color_red, 1)
    # Draw bounding box 2
    cv2.rectangle(image, box2_start, box2_end, color_green, 1)
    # Draw bounding box overlap
    cv2.rectangle(image, box_overlap_start, box_overlap_end, color_blue, 1)
    # Get inversed mask of the image
    inverse_mask = getInvMask(image)
    # Define a threshold to identify white pixels (adjust as needed)
    threshold = 200  
    # Get pixels above the threshold  
    white_pixels = np.where(inverse_mask >= threshold)
    # Count the number of white pixels for area
    area = len(white_pixels[0])
    # Get the edge
    edges = cv2.Canny(inverse_mask, 20, 150)
    # Get pixels above the threshold  
    white_pixels = np.where(edges >= threshold)
    # Count the number of white pixels for perimeter
    perimeter = len(white_pixels[0])
    # Get centroid from region of interest
    cx, cy = getCentroidFromMoments(inverse_mask, box_overlap_start, box_overlap_end)
    # Draw a dot to represent the centroid of the overlap region
    cv2.circle(image, (int(cx), int(cy)), 5, (0, 0, 255), -1)
    # Draw the lines
    image_with_lines, data, major_axis_text_loc, minor_axis1_text_loc, minor_axis2_text_loc = draw_lines_from_point(image, (cx, cy), edges, (min(bbox1[0], bbox2[0]), min(bbox1[1], bbox2[1])), (max(bbox1[2], bbox2[2]), max(bbox1[3], bbox2[3])))
    data["area"] = area
    data["perimeter"] = perimeter

    return image_with_lines, data, major_axis_text_loc, minor_axis1_text_loc, minor_axis2_text_loc

# Image scaling function
def scaleImage(image,scale_percent):
    src_img = image.copy()
    # Resize the image
    width = int(src_img.shape[1] * scale_percent / 100)
    height = int(src_img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(src_img, dim, interpolation=cv2.INTER_AREA)
    return img

# Image resizing function
def resize_image(img, new_width = 600):
    # Get the original dimensions
    height, width = img.shape[:2]

    # Calculate the new height to maintain the aspect ratio
    new_height = int((new_width / width) * height)

    # Resize the image
    resized_mat = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    return resized_mat

# Generate an empty list to put in the .csv
def empty_csv(csv_name):
    data = {
        'name': csv_name,
        'Longest Line': 0,
        'Perpendicular Line 1': 0,
        'Perpendicular Line 2': 0,
        'area': 0,
        'perimeter' : 0,
    }
    return data