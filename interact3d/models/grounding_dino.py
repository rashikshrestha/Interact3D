import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

class GroundingDINO():
    def __init__(self, device, text_prompt, box_th=0.2, text_th=0.3, obj_filter=None):
        self.device = device
        model_id = "IDEA-Research/grounding-dino-tiny"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
        self.prompt = text_prompt
        self.box_th = box_th
        self.text_th = text_th
        self.obj_filter = obj_filter

 
    def detect(self, image):
        '''
        Args:
            image (np.ndarray): OpenCV image, BGR, (H,W,3)
        
        Returns:
            List[List[int]]: N bounding boxes
        '''
        inputs = self.processor(images=image, text=self.prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=self.box_th,
            text_threshold=self.text_th,
            target_sizes=[image.shape[:2]]
        )
        results = results[0]
       
        bounding_boxes = [] 
        for label,box in zip(results['labels'],results['boxes']):
            if self.obj_filter is not None and label != self.obj_filter:
                continue
            xmin,ymin,xmax,ymax = box.cpu().numpy().astype(np.int32)
            bounding_boxes.append([int(xmin), int(ymin), int(xmax), int(ymax)])
            
        return bounding_boxes
    
if __name__=='__main__':
    from PIL import Image
    import cv2
    from interact3d.utils.utils_plot import plot_bounding_boxes
    from sam import SAM

    device = 'cuda' 
    text_prompt = 'green box.'
    img_path = '/home/rashik_shrestha/ws/Interact3D/temp/output/top_view.png'
    gdino = GroundingDINO(device, text_prompt, box_th=0.3, text_th=0.3)
    sam = SAM(device)
    
    #! Read image
    img_pil = Image.open(img_path) # PIL (RGB)
    img_cv = np.array(img_pil) # OpenCV RGB
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR) # OpenCV BGR
    
    #! Detection
    bb_dino = gdino.detect(img_cv)
    img_det = plot_bounding_boxes(img_cv.copy(), bb_dino)
    
    cv2.imwrite('green.png', img_det)
   
    mask = sam.get_segmentation_mask(img_pil, bb_dino)
    cv2.imwrite('mask.png', mask) 
    