import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tt


def conv_block(in_channels, out_channels, pool=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ELU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


def conv_block(in_channels, out_channels, pool=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class ResNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.input = conv_block(in_channels, 64)

        self.conv1 = conv_block(64, 64, pool=True)
        self.res1 = nn.Sequential(conv_block(64, 32), conv_block(32, 64))
        self.drop1 = nn.Dropout(0.5)

        self.conv2 = conv_block(64, 64, pool=True)
        self.res2 = nn.Sequential(conv_block(64, 32), conv_block(32, 64))
        self.drop2 = nn.Dropout(0.5)

        self.conv3 = conv_block(64, 64, pool=True)
        self.res3 = nn.Sequential(conv_block(64, 32), conv_block(32, 64))
        self.drop3 = nn.Dropout(0.5)

        self.classifier = nn.Sequential(
            nn.MaxPool2d(6), nn.Flatten(), nn.Linear(64, num_classes)
        )

    def forward(self, xb):
        out = self.input(xb)

        out = self.conv1(out)
        out = self.res1(out) + out
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.res2(out) + out
        out = self.drop2(out)

        out = self.conv3(out)
        out = self.res3(out) + out
        out = self.drop3(out)

        return self.classifier(out)



class faceRecognition():
    def __init__(self):
        # init
        self.face_classifier = cv2.CascadeClassifier("./face/emotion/models/haarcascade_frontalface_default.xml")
        model_state = torch.load("./face/emotion/models/emotion_detection_model_state.pth")
        self.class_labels = ["Angry", "Happy", "Neutral", "Sad", "Suprise"]

        # GPU 사용 가능 여부 확인
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ResNet(1, len(self.class_labels))
        # 모델 상태 사전을 GPU로 이동
        model_state = {k: v.to(self.device) for k, v in model_state.items()}
        self.model.load_state_dict(model_state)
        # 모델을 GPU로 이동
        self.model = self.model.to(self.device)

    def recognition(self, video):
        # method 1
        cap = cv2.VideoCapture("/home/gw6/gw/main_route/received_data/" + video) # Video file source "/home/gw6/gw/main_route/received_data/output_7.mp4"
        print(video) 
        print("process face")
        # 이건 과정 확인
        save_video_path = "/home/gw6/gw/main_route/face/emotion/result/" + video
        #재생할 파일의 넓이와 높이
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        #video controller
        fourcc = cv2.VideoWriter_fourcc(*'X264')
        out = cv2.VideoWriter(save_video_path, fourcc, 30.0, (int(width), int(height)))

        emotion_class = {
            "Angry": 0,
            "Happy": 0,
            "Neutral": 0,
            "Sad": 0,
            "Suprise" : 0
        }

        nope_emotion = True # 모든 frame에서 emotion이 detection 안될 때
        while True:
            # Grab a single frame of video
            ret, frame = cap.read()
            
            if not ret:
                print("끝 face")
                break
            
            # frame = cv2.flip(frame, 1)
            labels = []
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,minSize=(100,100), flags=cv2.CASCADE_SCALE_IMAGE)
            # frame = cv2.flip(frame, 0)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                roi_gray = gray[y : y + h, x : x + w]
                roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

                if np.sum([roi_gray]) != 0:
                    roi = tt.functional.to_pil_image(roi_gray)
                    roi = tt.functional.to_grayscale(roi)
                    roi = tt.ToTensor()(roi).unsqueeze(0)
                    roi = roi.to(self.device)

                    # make a prediction on the ROI
                    tensor = self.model(roi)
                    pred = torch.max(tensor, dim=1)[1].tolist()
                    label = self.class_labels[pred[0]]

                    label_position = (x, y)
                    cv2.putText(
                        frame,
                        label,
                        label_position,
                        cv2.FONT_HERSHEY_COMPLEX,
                        2,
                        (0, 255, 0),
                        3,
                    )
                    
                    # dict에 값 넣기
                    emotion_class[label] += 1
                    # 인식 가능
                    if emotion_class[label] >= 20:
                        nope_emotion = False
                else:
                    cv2.putText(
                        frame,
                        "No Face Found",
                        (20, 60),
                        cv2.FONT_HERSHEY_COMPLEX,
                        2,
                        (0, 255, 0),
                        3,
                    )

            out.write(frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        print(emotion_class)
        if nope_emotion == True:
            return (False, "")
        else:
            return (True, max(emotion_class,key=emotion_class.get))