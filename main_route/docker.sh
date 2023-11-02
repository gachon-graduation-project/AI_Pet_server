#!/bin/bash

# 컨테이너 멈추기
docker stop main_route

# 컨테이너 지우기
docker rm main_route

# 이미지 지우기
docker rmi main_route:latest

# 이미지 새로 빌드
docker build -t main_route:latest .

# docker run
docker run --name main_route -d -p 12125:12125 main_route:latest

# docker network 연결
docker network connect my-network main_route

# docker network 연결 확인
docker network inspect my-network
