# Atajos de desarrollo. Uso: make <target>
# (En Windows usa los comandos directamente o scripts\run_smoke.bat)

CFG ?= config/config.yaml
SMOKE_CFG = config/config_smoke.yaml

.PHONY: install test lint smoke synthetic preprocess baseline matrix report clean

install:        ## Instala dependencias y el paquete en modo editable
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e .

test:           ## Pruebas unitarias rápidas
	pytest -q

lint:           ## Linter
	ruff check src scripts tests

smoke:          ## Pipeline completo con datos sintéticos (CPU)
	bash scripts/run_smoke.sh

synthetic:      ## Genera solo los datos sintéticos
	python scripts/00_make_synthetic_data.py --config $(SMOKE_CFG)

preprocess:     ## Preprocesa datos reales
	python scripts/02_preprocess.py --config $(CFG)

baseline:       ## Baselines intra-dominio (ambos datasets, ResNet-50)
	python scripts/03_train_baseline.py --config $(CFG) --dataset busi    --arch resnet50
	python scripts/03_train_baseline.py --config $(CFG) --dataset bus_bra --arch resnet50

matrix:         ## Matriz de generalización cruzada (sin adaptación)
	python scripts/04_cross_domain_matrix.py --config $(CFG) --arch resnet50 --adaptation none

report:         ## Reporte consolidado
	python scripts/06_generate_report.py --config $(CFG)

clean:          ## Borra resultados y datos procesados (NO toca data/raw)
	rm -rf results/models/* results/figures/* results/reports/* data/processed/*
