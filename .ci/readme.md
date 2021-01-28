# Commands to build docker images.

## stable (0.2.4):

```bash
cd spectrochempy

docker build --build-arg BRANCH=0.2.4 --build-arg PY_VERSION=3.9 -t spectrocat/spectrochempy:stable .

docker run --entrypoint start.sh -p 8888:8888 -v /path/on/host/spectrochempy:/home/jovyan/spectrochempy --name scpy_stable -it spectrocat/spectrochempy:stable jupyter lab --port=8888
```

## latest (master)

```bash
cd spectrochempy

docker build --build-arg BRANCH=0.2.4 --build-arg PY_VERSION=3.9 -t spectrocat/spectrochempy:stable .

docker run --entrypoint start.sh -p 8889:8889 -v /path/on/host/spectrochempy:/home/jovyan/spectrochempy --name scpy_stable -it spectrocat/spectrochempy:stable jupyter lab --port=8889

```
