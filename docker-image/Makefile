all: build

build:
	cp -f ../requirements.txt vision_api_demo_requirements.txt
	docker build --tag=vision_api -f ./Dockerfile .
