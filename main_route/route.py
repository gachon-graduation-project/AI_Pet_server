from fastapi import FastAPI
from fastapi import WebSocket
from aiortc import RTCPeerConnection, RTCSessionDescription
import json

# api 통신
app = FastAPI()
pc = RTCPeerConnection()
pc.on("datachannel", lambda channel: print("datachannel", channel))

@app.get("/")
def root():
    return {"message" : "Hello World"}

@app.get("/home")
def home():
    return {"message" : "home"}

# 소켓 통신
   
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        # await websocket.send_text(f"You entered: {data}") 
        message = json.loads(data)
        if message["type"] == "offer":
            await pc.setRemoteDescription(RTCSessionDescription(**message))
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
            await websocket.send_json(
                {"type": "answer", "sdp": pc.localDescription.sdp}
            )
        elif message["type"] == "ice-candidate":
            pass  # Handle ICE candidates









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
