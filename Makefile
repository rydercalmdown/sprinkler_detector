IMAGE_NAME=10.0.0.179:32000/sprinkler-detector
VERSION=latest

.PHONY: run
run:
	@docker run --env-file settings.env -p 8000:80 $(IMAGE_NAME):$(VERSION)

.PHONY: build
build:
	@export IMAGE_NAME=$(IMAGE_NAME):$(VERSION) && cd deployment && bash install.sh

.PHONY: install
install:
	@make build

.PHONY: deploy
deploy:
	@kubectl apply -f deployment/kubernetes
	@kubectl scale deploy sprinkler-detector --replicas=0
	@kubectl scale deploy sprinkler-detector --replicas=1

.PHONY: push
push:
	@docker push $(IMAGE_NAME):$(VERSION)

.PHONY: redeploy
redeploy:
	@make build
	@make push
	@make deploy
