# OpenPI Branch

Runs an OpenPI-trained x2robot policy behind the `x2robot_client` WebSocket+msgpack protocol.

Example:

```bash
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost

vla/openpi/.venv/bin/python vla/maniparena_model_server/openpi_branch/launch_openpi_server.py \
  --config pi05_x2robot_pick_banana_ee \
  --checkpoint-dir /mnt/data/checkpoint/rain/openpi/checkpoints/pi05_x2robot_pick_banana_ee/pi05_x2robot_pick_banana/39999 \
  --host 0.0.0.0 \
  --port 8000
```

Notes:
- First request will be slow due to JAX/XLA compilation; subsequent calls are much faster.
- If local connections fail with proxy errors, ensure `NO_PROXY/no_proxy` includes `127.0.0.1,localhost`.
