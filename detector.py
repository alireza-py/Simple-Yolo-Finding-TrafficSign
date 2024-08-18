import cv2
import math
import numpy
import ultralytics
from dataclasses import dataclass

DEBUG = True

class Sign:
    def __init__(self) -> None:
        self.model_normal = ultralytics.YOLO('FNS.pt')
        self.out_data = {}

    def ai_detecting(self, image:numpy.ndarray) -> dict:
        if not isinstance(image, numpy.ndarray):
            return None
        
        result = self.model_normal.predict(image)

        for action in result:
            datasets = action.boxes.data.tolist()
            if not datasets:
                continue
            for data in datasets:
                class_key = data[5]
                position_x1 = int(data[0])
                position_y1 = int(data[1]) 
                position_x2 = int(data[2])
                position_y2 = int(data[3])
                class_name = action.names[class_key]
                position_one = (position_x1, position_y1)
                position_two = (position_x2, position_y2)
                self.out_data['sign_name'] = class_name
                self.out_data['sign_number'] = class_key
                self.out_data['first_position'] = position_one
                self.out_data['secound_position'] = position_two
                if DEBUG:
                    self.shape(image, class_name, position_one, position_two)
        return self.out_data

    def shape(self, 
             image:numpy.ndarray, 
             name_class:str,
             position_one:tuple, 
             position_two:tuple,) -> bool:

        difference_x = abs(position_two[0] - position_one[0])
        difference_y = abs(position_two[1] - position_one[1])
        center_x = position_one[0] + (difference_x // 2)
        center_y = position_one[1] + (difference_y // 2)
        distans_line = int(math.sqrt(difference_x**2 + difference_y**2) // 3)

        font_text = cv2.FONT_HERSHEY_SIMPLEX
        cv2.circle(image, position_one, 2, (200, 100, 100), -1)
        cv2.circle(image, position_two, 2, (100, 100, 200), -1)
        cv2.circle(image, (center_x, center_y), distans_line, (150, 100, 50), 2)
        cv2.putText(image, name_class, position_one, font_text, .7, (50, 100, 50), 2)

class Road:
    def __init__(self) -> None:
        pass