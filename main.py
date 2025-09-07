import customtkinter
import tkinter as tk
from tkinter import filedialog, messagebox
from ultralytics import YOLO
import numpy as np
import cv2
import os
import sys
from functions import *
from PIL import Image
from collections import defaultdict
import threading

if os.environ.get("MANGO_APP_ALREADY_RUNNING") == "1":
    sys.exit()

# Mark this instance as running
os.environ["MANGO_APP_ALREADY_RUNNING"] = "1"

# ---- GLOBAL VARIABLE FOR CALIBRATION ----
PIXEL_PER_CM = None

# ---- GLOBAL VARIABLE FOR SCALING ----
IMAGE_SCALE = None

# ---- CALIBRATION WINDOW ----
class CalibrationWindow(customtkinter.CTk):
  def __init__(self):
      super().__init__()
      self.title("Calibration")
      self.geometry("400x200")

      label = customtkinter.CTkLabel(self, text="Enter Pixel per CM value:", font=("Arial", 14))
      label.pack(pady=20)

      self.entry = customtkinter.CTkEntry(self, width=150)
      self.entry.pack(pady=10)

      submit_btn = customtkinter.CTkButton(self, text="Submit", command=self.submit_value)
      submit_btn.pack(pady=10)

  def submit_value(self):
      global PIXEL_PER_CM
      try:
          PIXEL_PER_CM = float(self.entry.get())
          if PIXEL_PER_CM <= 0:
              raise ValueError
      except ValueError:
          messagebox.showerror("Invalid Input", "Please enter a valid positive number.")
          return

      # Close calibration window
      self.destroy()

# ---- IMAGE SCALING WINDOW ----
class ScaleWindow(customtkinter.CTk):
  def __init__(self):
      super().__init__()
      self.title("Image Scaling")
      self.geometry("400x200")

      label = customtkinter.CTkLabel(self, text="Enter % for down scaling the image:", font=("Arial", 14))
      label.pack(pady=20)

      self.entry = customtkinter.CTkEntry(self, width=150)
      self.entry.pack(pady=10)

      submit_btn = customtkinter.CTkButton(self, text="Submit", command=self.submit_value)
      submit_btn.pack(pady=10)

  def submit_value(self):
      global IMAGE_SCALE
      try:
          IMAGE_SCALE = float(self.entry.get())
          if IMAGE_SCALE <= 0:
              raise ValueError
          elif IMAGE_SCALE > 100:
              raise ValueError
      except ValueError:
          messagebox.showerror("Invalid Input", "Please enter a valid positive number not over 100.")
          return

      # Close calibration window
      self.destroy()

# ---- MAIN WINDOW CLASS ----
class App(customtkinter.CTk):
  def __init__(self):
    super().__init__()
    self.title("Mango peduncle detection")
    self.geometry(f"{540}x{700}")
    self.filename = ""
    self.input_folder = ""
    self.output_detection_folder = ""
    self.create_widgets()
    self.result_label = customtkinter.CTkLabel(self, justify="left", text="")
    self.result_label.grid(row=2, column=1, padx=10, pady=10)
    self.result_label2 = customtkinter.CTkLabel(self, text="")
    self.result_label2.grid(row=2, column=2, padx=10, pady=10)
    self.result_image_label = customtkinter.CTkLabel(self, text="")
    self.result_image_label.grid(row=3, column=1, padx=10, pady=10)


  def create_widgets(self):
    # create tabview
    self.tabview = customtkinter.CTkTabview(self, width=300)
    self.tabview.grid(row=1, column=1, padx=(20, 0), pady=(20, 0), sticky="nsew")
    self.tabview.add("Image File")
    self.tabview.add("Image Folder")
    # configure grid of individual tabs
    self.tabview.tab("Image File").grid_columnconfigure(0, weight=1)
    self.tabview.tab("Image Folder").grid_columnconfigure(0, weight=1)

    # Input folder selection with label and browse button
    self.label_input_file = customtkinter.CTkLabel(self.tabview.tab("Image File"), text="Input File:")
    self.label_input_file.grid(row=0, column=0, padx=10, pady=10)
    self.filename_entry = customtkinter.CTkEntry(self.tabview.tab("Image File"), width=200)
    self.filename_entry.grid(row=0, column=1, padx=10, pady=10)
    self.button_browse_input = customtkinter.CTkButton(self.tabview.tab("Image File"), text="Browse", command=self.selectfile)
    self.button_browse_input.grid(row=0, column=2, padx=10, pady=10)

    # Output folder selection with label and browse button
    self.label_output_folder = customtkinter.CTkLabel(self.tabview.tab("Image File"), text="Output Folder:")
    self.label_output_folder.grid(row=1, column=0, padx=10, pady=10)
    self.output_folder_entry = customtkinter.CTkEntry(self.tabview.tab("Image File"), width=200)
    self.output_folder_entry.grid(row=1, column=1, padx=10, pady=10)
    self.button_browse_output = customtkinter.CTkButton(self.tabview.tab("Image File"), text="Browse", command=self.select_output_folder)
    self.button_browse_output.grid(row=1, column=2, padx=10, pady=10)
    
    # Run detection button
    self.button_run = customtkinter.CTkButton(self.tabview.tab("Image File"), text="Run", command=self.run_detection_file)
    self.button_run.grid(row=2, column=1, padx=10, pady=10)
    # Clear input button
    self.button_clear = customtkinter.CTkButton(self.tabview.tab("Image File"), text="Clear", command=self.clear_input_folder)
    self.button_clear.grid(row=2, column=2, padx=10, pady=10)

    # Input folder selection with label and browse button
    self.label_input_folder = customtkinter.CTkLabel(self.tabview.tab("Image Folder"), text="Input Folder:")
    self.label_input_folder.grid(row=0, column=0, padx=10, pady=10)
    self.input_folder_entry = customtkinter.CTkEntry(self.tabview.tab("Image Folder"), width=200)
    self.input_folder_entry.grid(row=0, column=1, padx=10, pady=10)
    self.button_browse_input = customtkinter.CTkButton(self.tabview.tab("Image Folder"), text="Browse", command=self.select_input_folder)
    self.button_browse_input.grid(row=0, column=2, padx=10, pady=10)

    # Output folder selection with label and browse button
    self.label_output_folder = customtkinter.CTkLabel(self.tabview.tab("Image Folder"), text="Output Folder:")
    self.label_output_folder.grid(row=1, column=0, padx=10, pady=10)
    self.output_folder_entry2 = customtkinter.CTkEntry(self.tabview.tab("Image Folder"), width=200)
    self.output_folder_entry2.grid(row=1, column=1, padx=10, pady=10)
    self.button_browse_output = customtkinter.CTkButton(self.tabview.tab("Image Folder"), text="Browse", command=self.select_output_folder)
    self.button_browse_output.grid(row=1, column=2, padx=10, pady=10)

    # Run detection button
    self.button_run = customtkinter.CTkButton(self.tabview.tab("Image Folder"), text="Run", command=self.run_detection_folder)
    self.button_run.grid(row=2, column=1, padx=10, pady=10)
    # Clear input button
    self.button_clear = customtkinter.CTkButton(self.tabview.tab("Image Folder"), text="Clear", command=self.clear_input_folder)
    self.button_clear.grid(row=2, column=2, padx=10, pady=10)

  def open_input_dialog_event(self):
      dialog = customtkinter.CTkInputDialog(text="Type in a number:", title="CTkInputDialog")
      print("CTkInputDialog:", dialog.get_input())

  def selectfile(self):
    self.filename = filedialog.askopenfilename()
    self.filename_entry.delete(0, tk.END)
    self.filename_entry.insert(0, self.filename)

  def select_input_folder(self):
    self.input_folder = filedialog.askdirectory()
    self.input_folder_entry.delete(0, tk.END)
    self.input_folder_entry.insert(0, self.input_folder)

  def select_output_folder(self):
    self.output_detection_folder = filedialog.askdirectory()
    self.output_folder_entry.delete(0, tk.END)
    self.output_folder_entry.insert(0, self.output_detection_folder)
    self.output_folder_entry2.delete(0, tk.END)
    self.output_folder_entry2.insert(0, self.output_detection_folder)
  
  def clear_input_folder(self):
     self.input_folder_entry.delete(0, 'end')
     self.output_folder_entry.delete(0, 'end')
     self.output_folder_entry2.delete(0, 'end')
     self.filename_entry.delete(0, 'end')

  def run_detection_folder(self):
      self.result_label.configure(text="Processing folder... Please wait.")

      def threaded_run():
          self.run_detection(input_type='folder')
          self.after(0, lambda: self.result_label.configure(text="Done processing folder."))

      threading.Thread(target=threaded_run).start()

  def run_detection_file(self):
      self.result_label.configure(text="Processing file... Please wait.")

      def threaded_run():
          self.run_detection(input_type='file')
          # self.after(0, lambda: self.result_label.configure(text="Done processing file."))

      threading.Thread(target=threaded_run).start()

  def run_detection(self, input_type):
      if input_type not in ['folder', 'file']:
          raise ValueError("Invalid input type. Must be 'folder' or 'file'.")

      if input_type == 'folder':
          if not self.input_folder or not self.output_detection_folder:
              messagebox.showwarning("Input Required", "Please select both input and output folders")
              return
          input_path = self.input_folder
      elif input_type == 'file':
          if not self.filename or not self.output_detection_folder:
              messagebox.showwarning("Input Required", "Please select both input file and output folders")
              return
          if not (self.filename.endswith('.jpg') or self.filename.endswith('.png') or self.filename.endswith('.JPG') or self.filename.endswith('.PNG')):
              messagebox.showwarning("Wrong input", "Please select file ending in '.jpg' or '.png'")
              return
          input_path = self.filename

      # Load pre-trained YOLO model
      model = YOLO(get_file_path('model.pt'))
      os.makedirs(self.output_detection_folder, exist_ok=True)

      # Define global variable for image scaling
      csv_data = []

      def process_image(filepath):
          csv_filename = os.path.join(self.output_detection_folder, "detection.csv")
          headers = ['name','Longest Line', 'Perpendicular Line 1', 'Perpendicular Line 2', 'area', 'perimeter','Longest Line (cm)','Perpendicular Line 1 (cm)','Perpendicular Line 2 (cm)']
          top_conf_0 = 0 
          top_conf_1 = 0
          xy_0 = []
          xy_1 = []
          src_img = cv2.imread(filepath)
          img = scaleImage(src_img, scale_percent=IMAGE_SCALE)
          # img = resize_image(src_img, 1000)
          results = model(img, conf=0.25, max_det=10)
          base_filename, _ = os.path.splitext(os.path.basename(filepath))
          print("Current file: {}".format(base_filename))
          csv_name = base_filename
          for r in results:
              boxes = r.boxes.cpu().numpy()
              for box in boxes:
                confidence = box.conf
                class_id = box.cls
                xyxy = box.xyxy
                if class_id == 0:
                  if top_conf_0 < confidence:
                    top_conf_0 = confidence
                    xy_0 = xyxy.tolist()[0]
                if class_id == 1:
                  if top_conf_1 < confidence:
                    top_conf_1 = confidence
                    xy_1 = xyxy.tolist()[0]
              xyxys = np.array([xy_0, xy_1], dtype="object")
              if len(boxes.xyxy) == 1:
                  print("Fruit or peduncle not detected")
                  self.result_label.configure(text="Fruit or peduncle not detected.")
                  csv_data.append(empty_csv(csv_name))
                  save_to_csv(csv_filename, headers, csv_data)
              elif len(xyxys) == 2:
                  bbox1 = xyxys[0]
                  bbox2 = xyxys[1]
                  overlapping_area = get_overlapping_area(bbox1, bbox2)
                  if overlapping_area:
                    # Crop the image to remove unwanted parts
                    # [rows, columns] 
                    largest_bbox = get_largest_bb(bbox1, bbox2)
                    crop = img[int(largest_bbox[1]-10):int(largest_bbox[3]+10), int(largest_bbox[0]-10):int(largest_bbox[2]+10)]   

                    # Print the coordinates of bounding boxes
                    print(f"bbox1 coordinate ({bbox1[0]}, {bbox1[1]}, {bbox1[2]}, {bbox1[3]})")
                    print(f"bbox2 coordinate ({bbox2[0]}, {bbox2[1]}, {bbox2[2]}, {bbox2[3]})")
                    print("Coordinates of overlapping area:", overlapping_area)
                    image_bbox_drawn, data, major_axis_text_loc, minor_axis1_text_loc, minor_axis2_text_loc = draw_bounding_boxes(img, bbox1, bbox2, overlapping_area)

                    # Label the line length
                    font = cv2.FONT_HERSHEY_DUPLEX
                    font_scale = 0.6
                    thick = 1
                    data['Longest Line'] *= 100/IMAGE_SCALE
                    data['Perpendicular Line 1'] *= 100/IMAGE_SCALE
                    data['Perpendicular Line 2'] *= 100/IMAGE_SCALE
                    data['Longest Line (cm)'] = data['Longest Line']/PIXEL_PER_CM
                    data['Perpendicular Line 1 (cm)'] = data['Perpendicular Line 1']/PIXEL_PER_CM
                    data['Perpendicular Line 2 (cm)'] = data['Perpendicular Line 2']/PIXEL_PER_CM
                    data['area']*= 100/IMAGE_SCALE
                    data['perimeter']*= 100/IMAGE_SCALE
                    image_bbox_drawn = cv2.putText(image_bbox_drawn, "Major axis: "+str(round(data["Longest Line"],2))+ " pixels", major_axis_text_loc, font, font_scale, color_red, thick)
                    image_bbox_drawn = cv2.putText(image_bbox_drawn, "Minor axis1: "+str(round(data["Perpendicular Line 1"],2))+ " pixels", minor_axis1_text_loc, font, font_scale, color_green, thick)
                    image_bbox_drawn = cv2.putText(image_bbox_drawn, "Minor axis2: "+str(round(data["Perpendicular Line 2"],2))+ " pixels", minor_axis2_text_loc, font, font_scale, color_blue, thick)
                    image_bbox_drawn = cv2.putText(image_bbox_drawn, "Major axis: "+str(round(data["Longest Line (cm)"],2))+ " CM", (major_axis_text_loc[0], major_axis_text_loc[1]+20), font, font_scale, color_red, thick)
                    image_bbox_drawn = cv2.putText(image_bbox_drawn, "Minor axis1: "+str(round(data["Perpendicular Line 1 (cm)"],2))+ " CM", (minor_axis1_text_loc[0], minor_axis1_text_loc[1]+20), font, font_scale, color_green, thick)
                    image_bbox_drawn = cv2.putText(image_bbox_drawn, "Minor axis2: "+str(round(data["Perpendicular Line 2 (cm)"],2))+ " CM", (minor_axis2_text_loc[0], minor_axis2_text_loc[1]+20), font, font_scale, color_blue, thick)
                  
                    # Saving results to csv
                    data["name"] = csv_name
                    csv_data.append(data)
                    # cv2.imwrite(edge_path, edges)
                    # save output images
                    detection_path = os.path.join(self.output_detection_folder, base_filename + '_detection.png')
                    cv2.imwrite(detection_path, image_bbox_drawn)
                    cv2.imwrite('temp_image.png', image_bbox_drawn)

                    self.result_label.configure(text=" Longest Line: {} CM\n Perpendicular Line 1 (minor axis 1): {} CM \n Perpendicular Line 2(minor axis 2): {} CM \n Area: {} px\n Perimeter: {} px\n Output File: {}"
                        .format(round(csv_data[-1]['Longest Line']/PIXEL_PER_CM, 2), round(csv_data[-1]['Perpendicular Line 1']/PIXEL_PER_CM, 2),
                                round(csv_data[-1]['Perpendicular Line 2']/PIXEL_PER_CM, 2), csv_data[-1]['area'],
                                csv_data[-1]['perimeter'], csv_data[-1]['name'] + '.png'))

                    # Keep a reference to avoid garbage collection
                    self.current_image = customtkinter.CTkImage(light_image=Image.open('temp_image.png'), dark_image=Image.open('temp_image.png'), size=(450, 300))
                    self.result_image_label.configure(image=self.current_image, text="")
                    save_to_csv(csv_filename, headers, csv_data)
                    print(f"Results saved to {csv_filename}")
                  else:
                      print("No overlap between bounding boxes.")
                      self.result_label.configure(text="No overlap between bounding boxes.")
                      csv_data.append(empty_csv(csv_name))
                      save_to_csv(csv_filename, headers, csv_data)
              else:
                  print("More than 1 fruit and 1 peduncle detected")
                  self.result_label.configure(text="More than 1 fruit and 1 peduncle detected")
                  csv_data.append(empty_csv(csv_name))
                  save_to_csv(csv_filename, headers, csv_data)


      if input_type == 'folder':
          for filename in os.listdir(input_path):
              if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.JPG') or filename.endswith('.PNG'):
                  filepath = os.path.join(input_path, filename)
                  process_image(filepath)
      else:
          process_image(input_path)

def get_file_path(filename):
	bundle_dir = getattr(sys, '_MEIPASS', os.path.abspath(os.path.dirname(__file__)))
	path_to_file = os.path.abspath(os.path.join(bundle_dir, filename))
	return path_to_file

if __name__ == "__main__":
    calibration = CalibrationWindow()
    calibration.mainloop()
    scale = ScaleWindow()
    scale.mainloop()
    app = App()
    app.mainloop()
