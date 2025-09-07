import cv2
import numpy as np
from scipy import ndimage
import os

def canny_edge(img):
    return cv2.Canny(img, 30, 150)

def sobel_edge(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(grad_x, grad_y)
    return np.uint8(np.clip(sobel, 0, 255))

def roberts_edge(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel_x = np.array([[1, 0], [0, -1]])
    kernel_y = np.array([[0, 1], [-1, 0]])
    grad_x = ndimage.convolve(gray.astype(float), kernel_x)
    grad_y = ndimage.convolve(gray.astype(float), kernel_y)
    edge = np.sqrt(grad_x**2 + grad_y**2)
    return np.uint8(np.clip(edge, 0, 255))

def hed_edge(img, proto_path, model_path):
    net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
    blob = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(img.shape[1], img.shape[0]),
                                 mean=(104.00698793, 116.66876762, 122.67891434),
                                 swapRB=False, crop=False)
    net.setInput(blob)
    hed = net.forward()
    hed = hed[0, 0]
    hed = cv2.resize(hed, (img.shape[1], img.shape[0]))
    hed = (255 * hed).astype("uint8")
    return hed

def main(input_path, output_dir, proto_path, model_path):
    os.makedirs(output_dir, exist_ok=True)
    img = cv2.imread(input_path)
    if img is None:
        print("Error: Could not load image.")
        return

    cv2.imwrite(os.path.join(output_dir, "canny.png"), canny_edge(img))
    cv2.imwrite(os.path.join(output_dir, "sobel.png"), sobel_edge(img))
    cv2.imwrite(os.path.join(output_dir, "roberts.png"), roberts_edge(img))
    cv2.imwrite(os.path.join(output_dir, "hed.png"), hed_edge(img, proto_path, model_path))

    print("Edge detection results saved in:", output_dir)

if __name__ == "__main__":
    input_img_path = "input.jpg"  # Change to your image path
    output_folder = "output_edges"
    hed_proto = "deploy.prototxt"  # Change if path differs
    hed_model = "hed_pretrained_bsds.caffemodel"  # Change if path differs

    main(input_img_path, output_folder, hed_proto, hed_model)
