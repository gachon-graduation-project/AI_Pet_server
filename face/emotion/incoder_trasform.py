import os
import cv2
import torch
import numpy as np


cap = cv2.VideoCapture("./result/oooo.mp4")

w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'X264')  # 비디오 인코딩 형식 설정
out = cv2.VideoWriter(f'./result/ooooii.mp4', fourcc, 30.0, (w, h))

while True:
    # Grab a single frame of video
    ret, frame = cap.read()
    frame = cv2.flip(frame, 0)
    frame = cv2.flip(frame, 1)
    if not ret:
        print("끝")
        break

    out.write(frame)

# 비디오 저장을 완료합니다.
out.release()