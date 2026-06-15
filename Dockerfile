# WAAAGH GROT-80M web demo — turnkey CPU image for Coolify / any Docker host.
#
# Weights are NOT baked in (319 MB, kept out of git + image). The image boots
# WITHOUT a model; an operator uploads grot-80m.safetensors once via /admin onto a
# persistent volume mounted at /data — it then survives redeploys.
#
# Required env at runtime:
#   ADMIN_PASSWORD    enables /admin (login + weights upload). Without it /admin is off.
#   ADMIN_USER        admin username (default: admin).
# Optional:
#   GROT_CKPT         weights path (default below, on the persistent volume).
#   ORKISH_REF        git ref of the Orkish repo to vendor (build arg, default: main).
#
# Security: non-root, no shell tool, python tool under kernel rlimits
# (see orkun/demo/safe_exec.py). Give the container NO outbound network in prod.
FROM python:3.11-slim

ARG ORKISH_REF=main

RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Orkish runtime deps: tokenizer, tool_registry.json, model arch (torch_impl/…), infer/.
RUN git clone --depth 1 --branch "${ORKISH_REF}" \
      https://github.com/guillaume34110/llm-waaagh-Orkish.git /opt/Orkish

COPY . /app

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir numpy tokenizers pyyaml safetensors

# /data = persistent volume; weights uploaded via /admin land here and survive redeploys.
ENV ORKISH_REPO=/opt/Orkish \
    GROT_CKPT=/data/grot-80m.safetensors \
    GROT_HOST=0.0.0.0 \
    GROT_PORT=8000 \
    PYTHONPATH=/app

RUN mkdir -p /data && useradd -m grot \
 && chown -R grot:grot /app /opt/Orkish /data
USER grot

EXPOSE 8000
CMD ["python", "-m", "orkun.demo.server"]
