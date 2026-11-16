# Ixion

Photo rotation detector. It predicts whether an image is upright and, at
inference time, picks the 90° rotation that makes it upright.

The model is an EfficientNet-B0 with one output neuron, `P(upright)`, trained
with binary cross-entropy on four views of each image: one upright and three
rotated.

## Requirements

Python 3.12+ and the packages in `requirements.txt`.

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

GPU training and inference need a PyTorch build matching your CUDA version.

## Dataset layout

`--data-root` points at a directory containing either corpus, or both:

```
<root>/ILSVRC2017/Data/CLS-LOC/{train,test,val}/
<root>/coco/{train2017,test2017,val2017}/
```

Only directories that exist are used; at least one corpus must be present.

## Training

```bash
python src/cli.py train \
    --data-root /path/to/datasets \
    --device cuda --batch-size 256 --num-workers 8 \
    --prefetch-factor 4 --seed 0
```

`finetune` is a short preset over ImageNet-pretrained weights:

```bash
python src/cli.py finetune \
    --data-root /path/to/datasets \
    --device cuda --batch-size 256 --num-workers 8 \
    --prefetch-factor 4 --seed 0
```

Training uses SAM over AdamW and `torch.compile` when the build environment
supports it. Checkpoints are written to `checkpoints/epoch_NNN.pt` with both
EMA and raw weights. Use EMA for validation and deployment.

## Validation

```bash
python src/cli.py validate \
    --checkpoint checkpoints/epoch_008.pt \
    --data-root /path/to/datasets \
    --device cuda --batch-size 256 --num-workers 4 \
    --prefetch-factor 4 --weights ema
```

The main metric is `group4_acc`: for each base image, the highest-scoring of
the four rotations should be the upright one. `margin` is top1 minus top2 and
can be used to abstain on ambiguous images.

## ONNX export

```bash
python src/cli.py export --checkpoint checkpoints/epoch_008.pt
```

The default output is:

```
checkpoints/onnx/ixion-efficientnet-e{epoch}-{weights}.onnx
```

Normalization, input size, rotation list, and input/output names are embedded
in the ONNX metadata. `--out` overrides the path, and `--verify` compares
PyTorch against ONNX Runtime.

Run an exported model with only `onnxruntime`, `numpy`, and `Pillow`:

```bash
python scripts/run_onnx.py model.onnx photo.jpg
python scripts/run_onnx.py model.onnx photos/ --output-dir corrected/
python scripts/run_onnx.py model.onnx photo.jpg --abstain-margin 0.15 --json
```

The runner also exposes an importable API:

```python
from run_onnx import OnnxRotationModel

model = OnnxRotationModel("model.onnx")
prediction = model.predict(image)
corrected = model.correct(image, prediction)
```

## Benchmark

```bash
python scripts/bench_onnx.py model.onnx /path/to/photos --device cuda
```

Reports images/s, views/s, and latency percentiles for preprocessing plus
inference. Image decoding is excluded from the timing. The CUDA path requires
`onnxruntime-gpu` in place of the CPU `onnxruntime` package; use `--device cpu`
for the CPU provider.

## Other commands

```bash
# Tkinter browser: shows each image corrected, with rotation probabilities
python src/cli.py browse --dir /path/to/photos --checkpoint model.pt --abstain-margin 0.15

# GradCAM grids for one or more images
python src/cli.py gradcam --checkpoint model.pt --images examples/*.jpg --out-dir gradcam_out

# DFR: retrain only the head on frozen features
python src/cli.py dfr --checkpoint model.pt --data-root /path/to/in_domain_images \
    --device cuda --batch-size 256 --num-workers 8 --prefetch-factor 4
```

## Project layout

```
src/
  cli.py         argument parsing
  dispatch.py    subcommand handler table
  stats.py       dataset mean/std with JSON caching
  data.py        transforms, rotation dataset, DataLoader
  checkpoint.py  checkpoint load/save
  inference.py   four-view scoring
  training.py    SAM training loop
  validation.py  group-4 validation
  dfr.py         head-only retraining
  gradcam.py     GradCAM rendering
  browse.py      Tkinter browser
  export.py      ONNX export
  config.py      ImageNet/COCO dataset specs
  constants.py   paths, sizes, rotations
  model/         RotationModel, attention pooling, EMA, SAM
  utils/         image I/O, file index, logging
scripts/
  run_onnx.py    standalone ONNX runner
  bench_onnx.py  ONNX throughput benchmark
```

## Inference details

Each image is resized so its shorter side is 232, center-cropped to 224,
converted to a tensor, and normalized with the checkpoint's mean/std
(ImageNet statistics for pretrained runs). The same tensor is rotated by 0°,
90°, 180°, and 270° and scored as a batch of four. The highest-scoring view
gives the correction angle; the top1-top2 probability margin controls
abstention.
