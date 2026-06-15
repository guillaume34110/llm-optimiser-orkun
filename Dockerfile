# WAAAGH GROT-80M web demo — turnkey CPU image for Coolify / any Docker host.
#
# Build args:
#   GROT_WEIGHTS_URL  URL to grot-80m.safetensors (e.g. a GitHub Release asset).
#                     Leave empty to bake weights in yourself or mount at runtime
#                     onto /app/orkun/demo/assets/grot-80m.safetensors.
#   ORKISH_REF        git ref of the Orkish repo to vendor (default: main).
#
# Security: runs as non-root, no shell tool, python tool under kernel rlimits
# (see orkun/demo/safe_exec.py). Give the container NO outbound network in prod.
FROM python:3.11-slim

ARG GROT_WEIGHTS_URL=""
ARG ORKISH_REF=main

RUN apt-get update && apt-get install -y --no-install-recommends git curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Orkish runtime deps: tokenizer, tool_registry.json, model arch (torch_impl/…), infer/.
RUN git clone --depth 1 --branch "${ORKISH_REF}" \
      https://github.com/guillaume34110/llm-waaagh-Orkish.git /opt/Orkish

COPY . /app

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir numpy tokenizers pyyaml safetensors

# Fetch weights at build if a URL was given (keeps the 319 MB blob out of git).
RUN mkdir -p /app/orkun/demo/assets \
 && if [ -n "$GROT_WEIGHTS_URL" ]; then \
        curl -fL "$GROT_WEIGHTS_URL" -o /app/orkun/demo/assets/grot-80m.safetensors; \
    fi

ENV ORKISH_REPO=/opt/Orkish \
    GROT_HOST=0.0.0.0 \
    GROT_PORT=8000 \
    PYTHONPATH=/app

# Non-root, read-only-friendly: the sandbox uses /tmp which stays writable.
RUN useradd -m grot && chown -R grot:grot /app /opt/Orkish
USER grot

EXPOSE 8000
CMD ["python", "-m", "orkun.demo.server"]
