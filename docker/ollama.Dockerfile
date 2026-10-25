# Local LLM server with the model baked into the image (no runtime download).
# Change the model with:  docker compose build --build-arg LLM_MODEL=qwen2.5:1.5b ollama
ARG OLLAMA_IMAGE=ollama/ollama:0.9.6
FROM ${OLLAMA_IMAGE}
ARG LLM_MODEL=qwen2.5:3b
ENV OLLAMA_HOST=0.0.0.0:11434
RUN ollama serve > /tmp/ollama-build.log 2>&1 & pid=$!; \
    for i in $(seq 1 60); do ollama list >/dev/null 2>&1 && break; sleep 1; done; \
    ollama pull "${LLM_MODEL}" && ollama list; status=$?; \
    kill $pid; wait $pid 2>/dev/null; exit $status
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 CMD ollama list >/dev/null || exit 1
