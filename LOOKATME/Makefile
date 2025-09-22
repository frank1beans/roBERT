
.PHONY: install convert tapt mlm label hier predict eval calibrate api docker-build docker-up pack-validate pack-build

install:
	python -m pip install --upgrade pip
	pip install -e .

convert:
	robimb convert --src data/raw/keynote_jsonl/tuo.jsonl --out data --seed 42 --lowercase

tapt:
	robimb train --task tapt --config configs/tapt.json

mlm:
	robimb train --task mlm --config configs/mlm.json

label:
	robimb train --task label --config configs/label.json

hier:
	robimb train --task hier --config configs/hier.json

predict:
	robimb predict --pack pack/current/pack.json --text "Muratura in laterizio forato sp 25 cm REI 120" --model runs/label --label-index data/metadata/label_index/cat.json

calibrate:
	robimb calibrate --model runs/label --label-index data/metadata/label_index/cat.json --val data/processed/classification/cat/val.jsonl --out runs/label/calibrator.json

eval:
	robimb eval --model runs/label --label-index data/metadata/label_index/cat.json --data data/processed/classification/cat/test.jsonl --out runs/eval/cat_test --calibrator runs/label/calibrator.json

api:
	uvicorn robimb.service.app:app --host 0.0.0.0 --port 8080

docker-build:
	docker build -f docker/Dockerfile.cpu -t robimb/api:latest .

docker-up:
	docker compose up --build -d

pack-validate:
	robimb pack-validate --pack pack/current/pack.json

pack-build:
	robimb pack-build --src pack/v0 --out pack/v1 --set-current
