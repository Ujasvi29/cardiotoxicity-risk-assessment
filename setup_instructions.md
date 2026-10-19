# Team Setup Instructions

## Cardiotoxicity Risk Assessment — Local Development Setup

These instructions allow each teammate to set up and run the
project on their own computer. No cloud services are required.

---

## Prerequisites

| Software | Version | Purpose |
|----------|---------|---------|
| Python | 3.13 or 3.14 | Runtime |
| Git | Latest | Version control |
| VS Code | Latest | Editor |
| Windows `tar` | Built-in | RAR extraction (Windows 10+) |

### Verify Prerequisites

```powershell
python --version     # Should show 3.13.x or 3.14.x
git --version        # Any recent version
code --version       # VS Code
```

---

## Step 1: Clone the Repository

```powershell
cd C:\dev
git clone https://github.com/Ujasvi29/cardiotoxicity-risk-assessment.git "cardiotoxicity risk assesment"
cd "cardiotoxicity risk assesment"
```

---

## Step 2: Create Virtual Environment

```powershell
python -m venv .venv
```

### Activate the Environment

```powershell
# PowerShell
.venv\Scripts\Activate.ps1

# If you get an execution policy error:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.venv\Scripts\Activate.ps1
```

Your prompt should now show `(.venv)`.

---

## Step 3: Install Dependencies

```powershell
# Upgrade pip first
python -m pip install --upgrade pip

# Install Phase 1 dependencies
pip install -r requirements.txt
```

### Verify Installation

```powershell
python -c "import pandas; import numpy; import matplotlib; import sklearn; print('All Phase 1 dependencies OK')"
```

---

## Step 4: Obtain the Dataset

The dataset is **not stored in the Git repository** (it is
gitignored due to size). Each teammate must download it separately.

### Option A: Download from Figshare (Official Source)

1. Go to: https://figshare.com/articles/dataset/BC_cardiotox_A_cardiotoxicity_dataset_for_breast_cancer_patients/22650748
2. Download all files
3. Place them in the `dataset/` directory at the project root

### Option B: Copy from a Teammate

If a teammate already has the dataset, copy the entire `dataset/`
folder (including the extracted `IMAGES/` subdirectory) to your
project root.

### Expected Dataset Structure

```
dataset/
├── BC_cardiotox_clinical_variables.csv          (56.8 KB, 531 rows × 27 cols)
├── BC_cardiotox_functional_variable.csv         (3.2 MB, 270 rows × 1002 cols)
├── BC_cardiotox_clinical_and_functional_variables.csv  (3.5 MB, 531 rows × 1028 cols)
├── IMAGES.rar                                   (61.6 MB)
├── IMAGES/                                      (270 PNG files after extraction)
│   ├── image (1).png
│   ├── image (2).png
│   ├── ...
│   └── image (270).png
├── README.txt
├── TDI_images_preprocessing_algorithm.R
├── TDIimage_automatic.png
└── TDIimage_manual.png
```

### Extract Images (if not already extracted)

```powershell
tar -xf "dataset\IMAGES.rar" -C "dataset\"
```

### Verify Dataset

```powershell
python -c "
import os
img_count = len([f for f in os.listdir('dataset/IMAGES') if f.endswith('.png')])
import csv
with open('dataset/BC_cardiotox_clinical_variables.csv') as f:
    rows = sum(1 for _ in csv.reader(f, delimiter=';')) - 1
print(f'Clinical rows: {rows} (expected 531)')
print(f'TDI images: {img_count} (expected 270)')
if rows == 531 and img_count == 270:
    print('Dataset verification: PASSED')
else:
    print('Dataset verification: FAILED - check file integrity')
"
```

---

## Step 5: Open in VS Code

```powershell
code "C:\dev\cardiotoxicity risk assesment"
```

### Recommended VS Code Extensions

- **Python** (Microsoft)
- **Jupyter** (Microsoft)
- **Pylance** (Microsoft)
- **GitLens** (optional, for Git history)

### Select Python Interpreter

1. Open Command Palette: `Ctrl+Shift+P`
2. Type: `Python: Select Interpreter`
3. Choose: `.venv` (the local virtual environment)

---

## Project Structure

```
cardiotoxicity risk assesment/
├── .gitignore              # Git exclusions
├── README.md               # Project overview
├── requirements.txt        # Python dependencies
├── setup_instructions.md   # This file
├── dataset/                # Raw data (gitignored)
│   ├── *.csv               # Clinical & functional CSVs
│   ├── IMAGES/             # Extracted TDI images
│   └── IMAGES.rar          # Original archive
├── data/                   # Processed data outputs
│   ├── processed/          # Cleaned datasets
│   └── splits/             # Train/val/test splits
├── notebooks/              # Jupyter notebooks (EDA)
├── src/                    # Reusable Python modules
│   ├── data/               # Data loading & preprocessing
│   ├── models/             # Model architectures
│   ├── evaluation/         # Metrics & reporting
│   └── utils/              # Shared utilities
├── configs/                # Configuration files
├── outputs/                # Generated results
│   ├── figures/            # Plots and visualizations
│   └── reports/            # Analysis reports
├── tests/                  # Unit tests
└── app/                    # Streamlit application
```

---

## Git Workflow

### Branching Strategy

```
main              ← stable, reviewed code
  └── feature/*   ← individual feature branches
```

### Daily Workflow

```powershell
# 1. Pull latest changes
git pull origin main

# 2. Create a feature branch
git checkout -b feature/your-feature-name

# 3. Make changes, then stage and commit
git add .
git commit -m "Description of changes"

# 4. Push your branch
git push origin feature/your-feature-name

# 5. Create a Pull Request on GitHub for review
```

### Commit Message Convention

```
[Module X] Brief description

- Detail 1
- Detail 2
```

Examples:
- `[Module 2] Add clinical data EDA notebook`
- `[Module 4] Implement missing value imputation`

---

## Troubleshooting

### PowerShell execution policy error
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### pip install fails on a package
```powershell
# Try upgrading pip first
python -m pip install --upgrade pip

# If a specific package fails, check its Python version support
pip install <package> --verbose
```

### PyTorch installation (Phase 2)
```powershell
# CPU-only (no NVIDIA GPU required)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Dataset CSV uses European decimal format
The CSVs use `;` as delimiter and `,` as decimal separator.
When reading with pandas, use:
```python
pd.read_csv('file.csv', sep=';', decimal=',')
```
