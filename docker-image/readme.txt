This image is intended as a light weight, "quick start" image to use with the sample Python scripts for working with the Trace Vision API.

1. To build Docker image locally:

    make build


2. To run the Docker container, use a command like:

    docker run -v /path/to/trace-vision-api-demo:/tmp/trace-vision-api-demo -v /path/to/video/directory:/tmp/data -it --rm --name vision_api_container vision_api ../bin/bash
