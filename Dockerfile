# GPU image (vLLM backend). Mount the model at /models/docket, or pass --model <hf-id> to download it.
FROM vllm/vllm-openai:v0.31.0
WORKDIR /app
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY docket_serve ./docket_serve
RUN pip install --no-cache-dir --no-deps . && pip install --no-cache-dir fastapi uvicorn
ENV DOCKET_MODEL=/models/docket HOST=0.0.0.0 PORT=8080
EXPOSE 8080
ENTRYPOINT ["docket-serve"]
