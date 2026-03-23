# M70: pip install torch with --index-url (not conda ::)
# M83: ALL package versions pinned
# M84: Consolidated pip workflow
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libhdf5-dev pkg-config git curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# M83: Pinned versions
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    scipy==1.12.0 \
    pandas==2.2.0 \
    scikit-learn==1.4.0 \
    lifelines==0.29.0 \
    scikit-survival==0.22.2 \
    xgboost==2.0.3 \
    lightgbm==4.3.0 \
    shap==0.45.0 \
    anndata==0.10.5 \
    scanpy==1.10.0 \
    mlflow==2.11.0 \
    optuna==3.5.0 \
    pydantic==2.6.0 \
    pyarrow==15.0.0 \
    matplotlib==3.8.3 \
    seaborn==0.13.2 \
    plotly==5.19.0 \
    jinja2==3.1.3 \
    tqdm==4.66.2 \
    joblib==1.3.2

# M70: PyTorch via pip --index-url (not conda)
RUN pip install --no-cache-dir torch==2.2.1 --index-url https://download.pytorch.org/whl/cu121

COPY . .

# M77: Export PYTHONHASHSEED in entrypoint
ENV PYTHONHASHSEED=42
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["/app/orchestration/docker-entrypoint.sh"]
CMD ["python", "main.py", "--stages", "all"]
