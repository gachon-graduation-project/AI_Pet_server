FROM python:3.9-slim

# 작업 디렉터리 설정
WORKDIR /usr/src/app

# 필요한 라이브러리 설치
RUN pip install --upgrade pip
RUN pip install "fastapi[all]"
RUN pip install numpy pandas matplotlib scikit-learn
RUN pip install opencv-python

# 시스템 패키지 설치
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx && \
    apt-get install -y libglib2.0-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


# PyTorch 설치
RUN pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 --index-url https://download.pytorch.org/whl/cu118

# 요구 사항 파일 복사 및 설치
COPY requirements.txt ./
RUN pip install -r requirements.txt

# 애플리케이션 코드를 컨테이너 안에 복사
COPY . .
COPY /received_data .
COPY /face .
COPY /hand .
COPY /reinforcement .


# uvicorn 실행
CMD ["uvicorn", "route:app", "--host", "0.0.0.0", "--port", "12125"]
