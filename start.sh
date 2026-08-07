#!/usr/bin/env bash
# Render (and any host that injects $PORT) start script for the Streamlit app.
# Falls back to 8501 for local runs where $PORT is not set.
exec streamlit run app/dashboard.py \
  --server.port "${PORT:-8501}" \
  --server.address 0.0.0.0 \
  --server.headless true
