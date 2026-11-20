---
title: "ComfyUI on a rented GPU"
description: "Run ComfyUI and Pneumam on a remote GPU and use it from your laptop's browser."
sidebar:
  order: 2
---

ComfyUI is a web server, so the GPU can be remote and the laptop only runs the browser. The graph editor runs on the client. Generation, models and outputs stay on the server.

Supported GPUs are the ones with an extra in `pyproject.toml`: `sm120` (RTX 5080/5090, RTX PRO 6000 Blackwell), `sm90` (H100/H200), `sm89` (RTX 40 series, L4, L40) and `sm86` (RTX 30 series, A10, A40). There is no `sm80` extra, so A100 is not supported.

## Server

A lium.io RTX 5090 pod costs about $0.60 an hour and comes with CUDA 13.0, `uv` and `git` installed.

1. Clone both repositories.

   ```bash
   git clone --depth 1 https://github.com/tan0210/pneumam ~/pneumam
   git clone --depth 1 https://github.com/comfyanonymous/ComfyUI ~/ComfyUI
   ```

2. Install Pneumam with ComfyUI's dependencies. ComfyUI itself needs no install.

   ```bash
   cd ~/pneumam
   uv sync --extra comfyui --extra sm120   # or sm90, sm89, sm86
   ln -sfn ~/pneumam/comfyui ~/ComfyUI/custom_nodes/pneumam_nodes
   ```

3. Convert the model. `HF_TOKEN` is needed only if the Hugging Face repository requires it.

   ```bash
   export HF_TOKEN=...
   uv run python scripts/convert.py --out checkpoints
   mv checkpoints/anima.safetensors ~/ComfyUI/models/diffusion_models/
   mv checkpoints/wan21-vae.safetensors ~/ComfyUI/models/vae/
   ```

4. Start ComfyUI from the Pneumam folder, so it uses the environment with the kernels.

   ```bash
   uv run python ~/ComfyUI/main.py --highvram --disable-cuda-malloc --listen 127.0.0.1 --port 8188
   ```

   `--disable-cuda-malloc` is required. Without it, generation fails with "Pointer argument cannot be accessed from Triton". Keep `--listen` on `127.0.0.1`: ComfyUI has no login.

The first run is slower, because it downloads the text encoder and prepares Pneumam for the image size. The first image at each new size takes a few seconds longer too.

LoRA files go in `models/loras` and LLLite files in `models/model_patches`.

## Client

1. Open an SSH tunnel and leave it running. Use the host and port from `lium ps <pod> --format json`.

   ```bash
   ssh -L 8188:127.0.0.1:8188 -p <port> root@<host>
   ```

2. Open `http://127.0.0.1:8188` in a browser, or connect the ComfyUI Desktop app:

   - In Desktop's instance list, add a new instance and choose **On a server**.
   - Enter `http://127.0.0.1:8188` and connect. Start the tunnel and ComfyUI on the server first, or the connection times out after 15 seconds.

   Desktop shows the server's own ComfyUI, so the workflow runs on the server with its nodes and models, and nothing needs installing on your computer. Desktop doesn't check the server's ComfyUI version, and it has no console for a server instance.

   If you also run a local ComfyUI in Desktop, it uses port 8188 too. Tunnel to another local port instead, such as `ssh -L 18188:127.0.0.1:8188 ...`, and connect to `http://127.0.0.1:18188`.

3. Build the graph:

   - In Load Pneumam Model, choose `anima.safetensors` and turn on `fp8` and `cuda_graphs`.
   - Connect Load Pneumam Text Encoder to two Pneumam Text Encode nodes, one for `positive` and one for `negative`, and set `length` to 512 on both.
   - Connect Empty Latent Image to the `latent_image` input of Pneumam KSampler.
   - In Pneumam KSampler, choose `euler` with 30 steps, cfg 4.0 and `shift` 3.0.
   - Connect the `LATENT` output of Pneumam KSampler and Load Pneumam VAE (`wan21-vae.safetensors`) to Pneumam VAE Decode, and connect that to Save Image.

   `latent_image` and the Save Image node are easy to forget. The graph runs but produces nothing without them.

4. Insert Pneumam LoRA or Pneumam LLLite between Load Pneumam Model and the sampler to use them.

Generated images are saved on the server. Desktop also downloads each one to its local output folder; turn off `autoDownloadOutputs` in the instance settings to stop that. From a browser, copy them back with `scp` or `rsync`.

## Prompts

Anima takes Danbooru-style tags and short natural-language sentences. Check the Anima page on Hugging Face for the recommended tags.

```
positive: masterpiece, best quality, score_7, 1girl, solo, <description>
negative: worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, sepia, bad anatomy, bad hands, extra fingers, watermark, text
```

Stay near 1024x1024 in total area, such as 832x1216 or 1216x832. Far larger or smaller sizes cause doubled limbs and broken composition.

## FP8

The `fp8` switch makes generation about twice as fast. FP8 attention is a separate experimental option, and it is off by default. To try it, set `PNEUMAM_EXPERIMENTAL_FP8_ATTENTION=1` in the shell that starts ComfyUI and keep `fp8` on. The nodes have no switch for it. It is faster, but images may show odd artifacts.

## Speed

See [Speed](/reference/speed/) for the time per image on an RTX 5090.

## Cost and cleanup

The pod has no spend cap, so remove it when done.

```bash
lium rm <pod> -y
```

Removing the pod deletes the checkpoints and everything else on it. A new pod needs the whole setup again. Invalidate the Hugging Face token once the session ends.
