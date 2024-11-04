import numpy
import cv2

def read_image(image_path):
    return cv2.imread(image_path)

def show_image(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def text_image(image, text, x, y, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, color=(255, 255, 255), thickness=2):
    return cv2.putText(image, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

def save_image(image, image_path):
    cv2.imwrite(image_path, image)

def show_array(image_array, text=None, x=None, y=None):
    if text is not None:
        image_array = text_image(image_array, text, x, y, color=(0, 0, 0))
    cv2.imshow('image', image_array)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def slice_image(image, x, y, width, height):
    return image[y:y+height, x:x+width]

def segment_image(image, rows, cols, size=None):
    img_height, img_width = image.shape[:2]
    block_height = img_height // rows
    block_width = img_width // cols
    blocks = []
    for y in range(0, img_height - img_height % block_height, block_height):
        for x in range(0, img_width - img_width % block_width, block_width):
            block = slice_image(image, x, y, block_width, block_height)
            if size is not None:
                block = resize(block, *size)
            blocks.append(block)
    return blocks

def resize(image, width, height):
    return cv2.resize(image, (width, height))

def split_image(image, width, height):
    img_height, img_width = image.shape[:2]
    blocks = []
    for y in range(0, img_height, height):
        for x in range(0, img_width, width):
            block = slice_image(image, x, y, width, height)
            blocks.append(block)
    return blocks

def to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)