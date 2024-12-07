import cv2

def plot_bounding_boxes(image, bounding_boxes, color=(0,0,255)):
    """
    image (np.ndarray): Image
    bounding_boxes(np.ndarray or List[List[int]])
    """
    for bb in bounding_boxes:
        xmin, ymin, xmax, ymax = bb
        cv2.rectangle(image, (xmin,ymin), (xmax,ymax), color, 10)
    return image