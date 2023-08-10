import cv2

def ResizeImage(width: int, height: int, imagePath: str):
    img = cv2.imread(imagePath)
    res = cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_CUBIC)
    