FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
    HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
RUN pip install --no-cache-dir \
    numpy==1.26.4 scipy==1.14.1 scikit-learn==1.5.2 \
    transformers==4.46.3 peft==0.13.2 accelerate==1.1.1 \
    safetensors==0.4.5 huggingface-hub==0.26.2 \
    opencv-python-headless==4.10.0.84 av==13.1.0 pillow==11.0.0 \
    pandas==2.2.3 threadpoolctl==3.5.0
WORKDIR /app
CMD ["python", "-u", "revision/experiment_queue.py", "worker", "--root", "/results"]
