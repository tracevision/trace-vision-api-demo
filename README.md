
# Trace Vision API Sample Code

These sample scripts use the Trace Vision API. They are intended to illustrate the basics of working with the API to process footage and use results.

We're barely scratching the surface here of what's possible, and look forward to seeing where you take this, whether that's creating visually stunning overlays for footage, combining metadata into detailed analytics and playlists, or anything else!


## The Trace Vision API

The Trace Vision API helps you understand and get value from videos by providing core data about key people, objects, and events shown in footage.

Full API documentation can be found at [https://api.tracevision.com/graphql/v1/docs/introduction/welcome](https://api.tracevision.com/graphql/v1/docs/introduction/welcome).

The API URL is [https://api.tracevision.com/graphql/v1/](https://api.tracevision.com/graphql/v1/).

Please note that to use the API, you will need a customer ID and API key. These can be found in the [developer portal](https://developer.tracevision.com/). You can request access to the API and developer portal [here](https://www.tracevision.com/developer-resources).


## Sample code

We provide sample code for working with the Trace Vision API to
* create a session
* upload footage
* check the session status
* retrieve results
* use the results to create tracking overlays and highlight clips


Sample code is provided in Python, and is intended to illustrate the basics of working with the API to process footage and use results.

Of course, you can use any programming language you choose to make requests to the API as well as retrieve and use results. We hope that you will integrate API calls into your own tech stack in a way that makes sense for your app or product.


## Python sample scripts

There are two scripts:
1. `create_trace_vision_session.py`, which walks through creating a session and uploading footage; and
2. `use_trace_vision_session_results.py`, which walks through checking the session status, retrieving results, and using those results to create tracking overlays and highlight clips.


Typically, you will want to use `create_trace_vision_session.py` to create a session and upload video.

Once that session is finished processing, run `use_trace_vision_session_results.py` to retrieve results and use them.

Both scripts make use of the third file, `vision_api_interface.py`, which contains an interface class with functionality to use the API and retrieve data from its responses.


### Python environnment

These scripts were developed using Python 3.10.13.

Python requirements can be found in `requirements.txt`.

We also provide a `Dockerfile` with an appropriate minimal Docker image so you can get started quickly.


### Getting started (using Docker)

#### 1. Build Docker image

Navigate to `docker-image` and build the Docker container
```sh
cd docker-image
make build
```

#### 2. Launch Docker container

You will need a local folder for the video footage you want to upload in step 3. If you don't already have one you want to use, you can make one now with a command like
```sh
mkdir local-video-footage
```

Start a Docker container using a command like
```sh
docker run -v /path/to/local-video-footage:/tmp/local-video-footage -it --rm --name vision_api_container vision_api ../bin/bash
```
Note that:
- You will need to replace `/path/to/local-video-footage` with the folder path you just created. All files in this folder will be visible and editable from inside your Docker container, where they will be located at `/tmp/local-video-footage`. For more information on mapping volumes, see [the Docker documentation](https://docs.docker.com/storage/volumes/).
- This command will start a bash terminal inside your running Docker container.


#### 3. Create a Trace Vision API session and upload video

Follow instructions in `create_trace_vision_session.py` to find footage, create appropriate metadata, create a session, and upload your video.

Note that from outside Docker, you can save your video file and session input JSON file in `/path/to/local-video-footage`. Inside your Docker container, use `/tmp/local-video-footage` in your filepaths when providing arguments to Python scripts.


#### 4. Retrieve results and create videos

Follow instructions in `use_trace_vision_session_results.py` to check your session status, retrieve results, and use metadata to cut highlights and create tracking overlays.

Note that you can save your output to `/tmp/local-video-footage` from inside Docker and it will be available outside Docker at `/path/to/local-video-footage`.
