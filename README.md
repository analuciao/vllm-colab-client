# vllm-colab-client

Minimal Python 3.9 client to call a vLLM model hosted on Google Colab via ngrok.

## Usage

1. Start vLLM on Colab and create an ngrok HTTPS tunnel.
2. Put the ngrok HTTPS base URL in `call_vllm_minimal.py` (NGROK_BASE).
3. Run: `python3 call_vllm_minimal.py`.

