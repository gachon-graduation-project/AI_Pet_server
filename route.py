from fastapi import FastAPI
from fastapi import WebSocket
import cv2
import numpy as np
import asyncio
import time
from face.emotion.video_capture import faceRecognition
from reinforcement.main_NN import Reinforcement
from hand.hand_recognition import handRecognition

# api 통신
app = FastAPI()

@app.get("/")
def root():
    return {"message" : "Hello World"}

@app.get("/home")
def home():
    return {"message" : "home"}

# 소켓 통신
   
# @app.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     await websocket.accept()
#     while True:
#         data = await websocket.receive_text()
#         await websocket.send_text(f"You entered: {data}") 


face = faceRecognition()
hand = handRecognition()
reinforcement = Reinforcement()
@app.websocket("/ws1")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    i = 0
    while True:
        # 비디오 라이터 설정
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # 비디오 인코딩 형식 설정
        out = cv2.VideoWriter(f'./received_data/output_{i}.mp4', fourcc, 15.0, (640, 480))
        print("시작")

        # 일정 시간 동안 이미지를 수집합니다.
        start_time = asyncio.get_event_loop().time()
        duration = 10  # 10초 동안 데이터 수집

        while asyncio.get_event_loop().time() - start_time < duration:
            # 바이너리 데이터를 수신합니다.
            data = await websocket.receive_bytes()

            # 바이너리 데이터를 NumPy 배열로 변환합니다.
            nparr = np.frombuffer(data, np.uint8)

            # 이미지를 디코딩합니다.
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # 비디오 파일에 프레임을 추가합니다.
            out.write(image)

        # 비디오 저장을 완료합니다.
        out.release()
        # model process

        # start_time = time.time()
        # emotion 없으면 (False, "") 있으면 (True, emotion)
        
        (state_emotion, emotion) = face.recognition(f'output_{i}.mp4')
        (state_hand, reward) = hand.recognition(f'output_{i}.mp4')
        if state_emotion == True:
            print("emotion detection 완료")
            action = reinforcement.first_process(emotion)
        else:
            print("emotion detection 실패")
            action = -1 # 행동 없다.

        if state_hand == True:
            print("hand detection 완료")
            if reward == "stop" :
                # exception
                pass
            else:
                reinforcement.second_process(reward)
        else:
            print("hand detection 실패")
        
        print(f"output_{i}.mp4 는 {action} 번째 행동")
        # elapsed_time = time.time() - start_time
        # if elapsed_time < duration:
        #     await asyncio.sleep(duration - elapsed_time)
        await websocket.send_text(f"{action}")

        # cache
        i += 1 
        if i > 9:
            i = 0

# @app.websocket("/ws2")
# async def websocket_endpoint(websocket: WebSocket):
#     await websocket.accept()
#     i = 1
#     start_time = asyncio.get_event_loop().time()
#     duration = 10
#     # 첫번째 websocket 기다리기
#     while asyncio.get_event_loop().time() - start_time < duration:
#         continue
    
#     # 첫번쨰 websocket 
#     await asyncio.sleep(1)
    
#     while True:
#         # 비디오 라이터 설정
#         fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # 비디오 인코딩 형식 설정
#         out = cv2.VideoWriter(f'./received_data/output_{i}.mp4', fourcc, 15.0, (640, 480))
#         print("시작2")
#         # 일정 시간 동안 이미지를 수집합니다.
#         start_time = asyncio.get_event_loop().time()
#         duration = 10  # 10초 동안 데이터 수집

#         while asyncio.get_event_loop().time() - start_time < duration:
#             # 바이너리 데이터를 수신합니다.
#             data = await websocket.receive_bytes()

#             # 바이너리 데이터를 NumPy 배열로 변환합니다.
#             nparr = np.frombuffer(data, np.uint8)

#             # 이미지를 디코딩합니다.
#             image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

#             # 비디오 파일에 프레임을 추가합니다.
#             out.write(image)

#         # 비디오 저장을 완료합니다.
#         out.release()
#         # model process

#         start_time = time.time()
#         # emotion 없으면 (False, "") 있으면 (True, emotion)
#         (state, emotion) = face.recognition(f'output_{i}.mp4')
#         if state == True:
#             print("detection 완료")
#             action = reinforcement.first_process(emotion)
#         else:
#             print("detection 실패")
#             action = -1 # 행동 없다.
        
#         print(f"output_{i}.mp4 는 {action} 번째 행동")
#         elapsed_time = time.time() - start_time
#         if elapsed_time < duration:
#             await asyncio.sleep(duration - elapsed_time)

#         await websocket.send_text(f"output_{i}.mp4 는 {action} 번째 행동")

        
#         # reward
#         i += 2
#         if i > 9:
#             i = 1


# 보내는 웹 소켓 하나 필요




#도커 컨테이너 전체
#output 형태 (json같이 일정 형태가 있는) 
        """
        6개의 input과 output에 대한 명세서 필요 
        face recognition : Input - 사진, 영상, output - 사진에 대한 label, status, label에 대한 확률 정도 
        hand recognition : Input - 사진, 영상, output - 사진에 대한 label, status, label에 대한 확률 정도 
        input controller : label을 받음, face, hand recognition에서 받은 output들 둘다. 
        여기서의 output - 처리 유무 판단 (이전 사진의 label과 같아서 처리를 안할 수도 있고, label이 다르다면 처리를 할 것이다.) + label (강화학습의 input값으로 넣어줄)

        input controller는 나누는게 편할 듯 

        reinforcement -> face contoller에 대한 output , action vector 



        """
