                                             
docker rm redis

docker run --name redis -p 6379:6379 -d redis

docker network connect my-network redis

docker network inspect my-network