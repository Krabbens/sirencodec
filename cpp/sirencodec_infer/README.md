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
  --input sztyks2_orig.wav \
  --models compress_packet,decompress_packet,full_recon \
  --out-dir './c++outputs/litert_export_sztyks2' \
  --benchmark-runs 5 \
  --num-threads 16
```

Generated `sztyks2` models:

- `compress_packet_1x110933x1.tflite` - waveform to RVQ indices and residual norms
- `decompress_packet_1x3x434.tflite` - packet to waveform
- `full_recon_1x110933x1.tflite` - waveform to reconstructed waveform

## Inference

Codec split:

```bash
build/sirencodec_infer/sirencodec_infer \
  --mode codec \
  --input sztyks2_orig.wav \
  --compress-model './c++outputs/litert_export_sztyks2/compress_packet_1x110933x1.tflite' \
  --decompress-model './c++outputs/litert_export_sztyks2/decompress_packet_1x3x434.tflite' \
  --litert-lib '.venv-tflite/lib/python3.12/site-packages/ai_edge_litert/libLiteRt.so' \
  --num-threads 16 \
  --output-dir './c++outputs/tflite_codec_sztyks2'
```

Single full reconstruction model:

```bash
build/sirencodec_infer/sirencodec_infer \
  --mode full \
  --input sztyks2_orig.wav \
  --full-model './c++outputs/litert_export_sztyks2/full_recon_1x110933x1.tflite' \
  --litert-lib '.venv-tflite/lib/python3.12/site-packages/ai_edge_litert/libLiteRt.so' \
  --num-threads 16 \
  --output-dir './c++outputs/tflite_full_sztyks2'
```

`sirencodec_infer` performs one inference. Use `sirencodec_benchmark` for warmed
throughput numbers.

## Benchmark

```bash
build/sirencodec_infer/sirencodec_benchmark \
  --profile all \
  --input sztyks2_orig.wav \
  --full-model './c++outputs/litert_export_sztyks2/full_recon_1x110933x1.tflite' \
  --compress-model './c++outputs/litert_export_sztyks2/compress_packet_1x110933x1.tflite' \
  --decompress-model './c++outputs/litert_export_sztyks2/decompress_packet_1x3x434.tflite' \
  --litert-lib '.venv-tflite/lib/python3.12/site-packages/ai_edge_litert/libLiteRt.so' \
  --num-threads 16 \
  --preload-runs 1 \
  --benchmark-runs 5 \
  --output-dir './c++outputs/tflite_benchmark_sztyks2'
```

Latest local C++ LiteRT benchmark with `--num-threads 16`:

- `codec_full`: 0.099067 s, 69.99x realtime
- `compress_only`: 0.034069 s, 203.51x realtime
- `decompress_only`: 0.064998 s, 106.67x realtime
- `full`: 0.102328 s, 67.76x realtime

The thread count is passed to XNNPACK through LiteRT opaque CPU options.

## Tests

```bash
build/sirencodec_infer/sirencodec_tests \
  --input sztyks2_orig.wav \
  --full-model './c++outputs/litert_export_sztyks2/full_recon_1x110933x1.tflite' \
  --compress-model './c++outputs/litert_export_sztyks2/compress_packet_1x110933x1.tflite' \
  --decompress-model './c++outputs/litert_export_sztyks2/decompress_packet_1x3x434.tflite' \
  --litert-lib '.venv-tflite/lib/python3.12/site-packages/ai_edge_litert/libLiteRt.so' \
  --num-threads 16
```
