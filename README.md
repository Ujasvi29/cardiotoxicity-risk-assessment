# Cardiotoxicity Risk Assessment

Research code for an interpretable multimodal framework using clinical and cardiac/TDI data. The repository currently includes dataset exploration, quality assessment, clinical and waveform preprocessing, split preparation, and an EfficientNet-B0 image feature encoder. A verified image-to-record mapping is not available, so image features must not be treated as labeled CTRCD examples or joined to clinical records.

## Data location

Original data is excluded from Git. Place the source CSV files and optional `IMAGES/` folder in `dataset/` at the repository root, or set `CARDIOTOXICITY_DATA_ROOT` to an external directory containing them. Paths are resolved independently of the current working directory. Missing source files produce an actionable error; importing project modules does not create output directories.

```powershell
$env:CARDIOTOXICITY_DATA_ROOT = "D:\path\to\original-dataset"
```

See [setup_instructions.md](setup_instructions.md) for environment setup and [Module 9 report](docs/module_9_report.md) for image feature extraction, verification, and mapping limitations.
