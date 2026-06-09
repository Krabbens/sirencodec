# SirenCodec C++20 TFLite Inference

This project is TFLite/LiteRT-only. Deployment goes through fixed-shape
`.tflite` files and `libLiteRt.so`; the previous manual native runtime has been
removed from the C++ project.

The runtime path is:

1. `tools/export_litert_models.py` exports fixed-shape LiteRT/TFLite
   FlatBuffers from the PyTorch checkpoint.
2. `sirencodec_infer` loads the `.tflite` model(s), reads audio, runs LiteRT
   Compiled Model inference, writes WAV output and emits metrics.

Executables:

- `sirencodec_infer` - TFLite/LiteRT inference only
- `sirencodec_benchmark` - TFLite/LiteRT benchmark only
- `sirencodec_tests` - TFLite/LiteRT smoke tests
- `sirencodec_litert` and `sirencodec_litert_benchmark` - compatibility names

## Build

```bash
cmake -S cpp/sirencodec_infer -B build/sirencodec_infer -DCMAKE_BUILD_TYPE=Release
cmake --build build/sirencodec_infer -j
```

The LiteRT runner uses a RAII dynamic-loader wrapper around `libLiteRt.so`, so a
regular build does not need LiteRT headers. The library path can be passed with
`--litert-lib` or with `SIRENCODEC_LITERT_LIB`.

## Export TFLite Models

```bash
python3 -m venv .venv-tflite
.venv-tflite/bin/python -m pip install --upgrade pip
.venv-tflite/bin/python -m pip install litert-torch ai-edge-litert

.venv-tflite/bin/python tools/export_litert_models.py \
  runs/control315000_postlava_hp020_sem000_floor035_b8_20260514_112608/checkpoints/codec_step329999_cuda.pt \
  --input sample_16k.wav \
  --samples 32000 \
  --models compress_packet,decompress_packet,full_recon \
  --out-dir './artifacts/models/litert_trainclean100_2s' \
  --benchmark-runs 5 \
  --num-threads 16
```

Generated 2 s models:

- `compress_packet_1x32000x1.tflite` - waveform to RVQ indices and residual norms
- `decompress_packet_1x3x125.tflite` - packet to waveform
- `full_recon_1x32000x1.tflite` - waveform to reconstructed waveform

## Inference

Codec split:

```bash
build/sirencodec_infer/sirencodec_infer \
  --mode codec \
  --input sample_16k.wav \
  --compress-model './artifacts/models/litert_trainclean100_2s/compress_packet_1x32000x1.tflite' \
  --decompress-model './artifacts/models/litert_trainclean100_2s/decompress_packet_1x3x125.tflite' \
  --litert-lib '.venv-tflite/lib/python3.12/site-packages/ai_edge_litert/libLiteRt.so' \
  --num-threads 16 \
  --output-dir './artifacts/inference/litert_codec_sample'
```

Single full reconstruction model:

```bash
build/sirencodec_infer/sirencodec_infer \
  --mode full \
  --input sample_16k.wav \
  --full-model './artifacts/models/litert_trainclean100_2s/full_recon_1x32000x1.tflite' \
  --litert-lib '.venv-tflite/lib/python3.12/site-packages/ai_edge_litert/libLiteRt.so' \
  --num-threads 16 \
  --output-dir './artifacts/inference/litert_full_sample'
```

`sirencodec_infer` performs one inference. Use `sirencodec_benchmark` for warmed
throughput numbers.

## Benchmark

```bash
build/sirencodec_infer/sirencodec_benchmark \
  --profile all \
  --input sample_16k.wav \
  --full-model './artifacts/models/litert_trainclean100_2s/full_recon_1x32000x1.tflite' \
  --compress-model './artifacts/models/litert_trainclean100_2s/compress_packet_1x32000x1.tflite' \
  --decompress-model './artifacts/models/litert_trainclean100_2s/decompress_packet_1x3x125.tflite' \
  --litert-lib '.venv-tflite/lib/python3.12/site-packages/ai_edge_litert/libLiteRt.so' \
  --num-threads 16 \
  --preload-runs 1 \
  --benchmark-runs 5 \
  --output-dir './artifacts/benchmarks/litert_single_file_2s'
```

Dataset benchmark on 5% of LibriSpeech train-clean-100 with 2 s segments and
`--num-threads 16`:

- full reconstruction: 0.062011 s, 32.25x realtime
- compression and decompression: 0.061543 s, 32.50x realtime
- compression: 0.026084 s, 76.68x realtime
- decompression: 0.035459 s, 56.40x realtime

The thread count is passed to XNNPACK through LiteRT opaque CPU options.

## Tests

```bash
build/sirencodec_infer/sirencodec_tests \
  --input sample_16k.wav \
  --full-model './artifacts/models/litert_trainclean100_2s/full_recon_1x32000x1.tflite' \
  --compress-model './artifacts/models/litert_trainclean100_2s/compress_packet_1x32000x1.tflite' \
  --decompress-model './artifacts/models/litert_trainclean100_2s/decompress_packet_1x3x125.tflite' \
  --litert-lib '.venv-tflite/lib/python3.12/site-packages/ai_edge_litert/libLiteRt.so' \
  --num-threads 16
```
