import os
import cv2
import torch
import numpy as np
import torch.nn as nn


class handRecognition():
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load('ultralytics/yolov5',
                                'custom',
                                path='./hand/model/best_2.pt', force_reload=True)

        self.model.to(self.device)


    def recognition(self, video):
        cap = cv2.VideoCapture("/home/gw6/gw/main_route/received_data/"+video)
        print("process 1 hand")
        # 이건 과정 확인
        save_video_path = "/home/gw6/gw/main_route/hand/result/"+video
        #재생할 파일의 넓이와 높이
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        #video controller
        fourcc = cv2.VideoWriter_fourcc(*'X264')
        out = cv2.VideoWriter(save_video_path, fourcc, 30.0, (int(width), int(height)))

        dict_hand = {
            "positive" : 0,
            "negative" : 0,
            "nothing" : 0,
            "stop" : 0
        }
        nope_hand = True
        while True:
            # Grab a single frame of video
            ret, frame = cap.read()
                    
            if not ret:
                print("끝 hand")
                break

            results = self.model(frame)
            df = results.pandas().xyxy[0]  # 결과를 DataFrame으로 변환
            names = df['name'].tolist()  # 'name' 열의 값만 추출

            for i in names:
                if i in dict_hand:
                    dict_hand[i] += 1
                    if dict_hand[i] >= 5:
                        nope_hand = False

            out.write(np.squeeze(results.render()))

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        print(dict_hand)
        if nope_hand == True:
            return (False, "")
        else:
            return (True, max(dict_hand,key=dict_hand.get))
        
# if __name__=='__main__':
#     hand = handRecognition()
#     hand.recognition("output_1.mp4")
