# Makefile for CardioFusion

.PHONY: help install setup test clean lint format run docker-build docker-run

help:
	@echo "CardioFusion - Makefile Commands"
	@echo "================================"
	@echo "install       - Install dependencies"
	@echo "setup         - Complete project setup"
	@echo "check-data    - Verify dataset is present"
	@echo "preprocess    - Run data preprocessing"
	@echo "train         - Train all models"
	@echo "test          - Run tests"
	@echo "lint          - Run linting"
	@echo "format        - Format code with black"
	@echo "run           - Run Streamlit app"
	@echo "clean         - Clean generated files"
	@echo "docker-build  - Build Docker image"
	@echo "docker-run    - Run Docker container"

install:
	pip install --upgrade pip
	pip install -r requirements.txt

setup: install
	cp .env.example .env
	@echo "Setup complete!"
	@echo "Next step: Place your dataset in data/raw/ directory"

check-data:
	python scripts/download_data.py

preprocess:
	jupyter nbconvert --to notebook --execute notebooks/data_preprocessing.ipynb

train:
	jupyter nbconvert --to notebook --execute notebooks/baseline_models.ipynb
	jupyter nbconvert --to notebook --execute notebooks/advanced_models.ipynb

test:
	pytest tests/ -v --cov=src --cov-report=html

lint:
	flake8 src/ scripts/ tests/
	mypy src/

format:
	black src/ scripts/ tests/
	isort src/ scripts/ tests/

run:
	streamlit run src/app.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.log" -delete
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf htmlcov
	rm -rf .coverage

docker-build:
	docker build -t cardiofusion:latest .

docker-run:
	docker run -d -p 8501:8501 --name cardiofusion cardiofusion:latest

docker-stop:
	docker stop cardiofusion
	docker rm cardiofusion
