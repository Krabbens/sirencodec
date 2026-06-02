# Web Microphone Encode/Decode Demo Design

## Goal

Build a local web demo that streams microphone audio through the SirenCodec MLX `7m best` checkpoint, encodes and decodes it on the backend, and plays the reconstructed audio in the browser. The first version is a comparison/debug tool, not a production low-latency transport.

## Scope

The demo will support:

- Browser microphone capture.
- WebSocket transport of short PCM chunks to a local Python backend.
- Backend encode/decode using `runs/control315000_postlava_hp020_sem000_floor035_b8_20260514_112608/checkpoints/codec_step329999.npz`.
- Reconstructed PCM streamed back to the browser.
- UI controls for `250 ms`, `500 ms`, and `1000 ms` chunk sizes.
- Runtime stats: nominal bitrate, packed payload bitrate estimate, backend wall time, RTF, speed multiplier, queue depth, and drop count.
- Start/stop controls and clear connection/model status.

The demo will not implement cross-user sender/receiver sessions, WebRTC, entropy-coded network framing beyond the existing packed index calculation, or a truly stateful causal codec. The current model still runs full chunk encode/decode.

## Recommended Architecture

Use a small FastAPI app with:

- `GET /` serving a static demo page.
- `GET /static/...` serving JavaScript and CSS.
- `WS /ws/audio` carrying binary PCM chunks from the browser and JSON metadata/reconstructed audio messages back.

The frontend uses Web Audio APIs:

- `getUserMedia` for microphone access.
- An `AudioWorklet` where available, with a simple fallback if needed.
- Float32 PCM capture and downmix to mono in the browser. The browser includes its capture sample rate in each chunk header, and the backend resamples to 16 kHz before inference.
- A small playback scheduler that queues returned reconstructed PCM and tracks underruns.

The backend uses existing MLX inference logic, factored out from `tools/infer_mlx.py` where practical:

- Load `train_mlx.Config` with the known `7m best` architecture flags.
- Load the checkpoint once at server startup.
- For each received chunk, normalize/shape PCM as `[1, T, 1]`, run `model.forward_full`, extract reconstructed audio and RVQ indices, and compute packed bitrate stats without writing files.

## Data Flow

1. User clicks `Start`.
2. Browser asks for microphone permission and starts capture.
3. Browser batches audio into the selected chunk size.
4. Browser sends each chunk over WebSocket as a small header plus Float32 PCM payload.
5. Backend queues at most a small number of chunks. If the queue is full, it drops the oldest pending chunk and reports the drop so playback favors recent microphone audio.
6. Backend runs encode/decode and sends back reconstructed Float32 PCM plus metrics.
7. Browser schedules playback and updates status counters.
8. User can change chunk size while stopped; changing while running restarts the stream cleanly.

## Error Handling

The UI should show explicit states for:

- Microphone permission denied.
- WebSocket disconnected.
- Model loading or model load failure.
- Backend overload or dropped chunks.
- Audio playback underrun.

The backend should reject unsupported sample formats, cap message size, and close the socket cleanly on malformed messages.

## Testing

Use focused tests and manual verification:

- Unit test helper functions for chunk framing, bitrate calculations, and queue/drop behavior.
- Smoke test that the backend can load the `7m best` checkpoint and process a synthetic 1 second chunk.
- Browser manual test: start/stop works, microphone signal reaches backend, reconstructed audio plays, chunk size options update stats.
- Performance check on the same `czat.wav` or synthetic audio path to compare expected CPU/MLX speed with live chunk settings.

## Implementation Notes

Keep the first implementation local-only. FastAPI dependencies should be added intentionally, likely `fastapi`, `uvicorn`, and `websockets` or FastAPI's WebSocket support. The server command should be a simple script such as:

```bash
.venv/bin/python tools/web_stream_demo.py --host 127.0.0.1 --port 8765
```

The browser page should be utilitarian: compact controls, live counters, and a small log. It should not be a marketing page.
