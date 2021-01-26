# Commands to build docker image for master and the latest tag

## stable tag -> <tag>:

```bash
cd spectrochempy

docker build \
   --build-arg BRANCH=0.2.4  \
   --build-arg PY_VERSION=3.9  \
   -t spectrocat/spectrochempy:stable . &&\
docker run \
   --entrypoint start.sh \
   -p 8888:8888 \
   -v /path/on/host/spectrochempy:/home/jovyan/spectrochempy \
   --name scpy_stable \
   -t \
   -i \
   spectrocat/spectrochempy:stable jupyter lab --port=8888
```

## master -> latest

```bash
cd spectrochempy

docker build \
   --build-arg BRANCH=master  \
   --build-arg PY_VERSION=3.9  \
   -t spectrocat/spectrochempy:latest . &&\
docker run \
   --entrypoint start.sh \
   -p 8889:8889 \
   -v /path/on/host/spectrochempy:/home/jovyan/spectrochempy \
   --name scpy_latest \
   -t \
   -i \
   spectrocat/spectrochempy:latest jupyter lab --port=8889
```

docker build
--build-arg BRANCH=master --build-arg PY_VERSION=3.9
-t spectrocat/spectrochempy:latest .
&& docker run
--entrypoint start.sh
-p 8889:8889
-v /Users/christian/Dropbox/SCP/spectrochempy:/home/jovyan/spectrochempy
--name scpy_latest
-t
-i
spectrocat/spectrochempy:latest jupyter lab --port=8889
