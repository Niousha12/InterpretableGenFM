# InterpretableGenFM

This repository explores interpretability in DNABERT-2.
The model is fine-tuned and tested on the Cell Passport dataset, a curated large-scale collection of genomic variants. In this study, two strategies are implemented:

1. Fine-tuning with Focal Loss: reduces bias toward the majority benign class and improves detection of rare pathogenic variants.

2. Anomaly Detection Framework: DNABERT-2 acts as a teacher model and a distilled student model is trained on benign data; mismatches between teacher and student highlight potential pathogenic variants.

Also, two techniques are employed for interpretability:

* Attention map analysis is used for the fine-tuned DNABERT-2 classifier.

* Gradient-based attribution is used for the anomaly detection framework.

[//]: # (This combination provides biologically meaningful explanations of which regions of DNA sequences drive predictions.)

## Run the code

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

To run the fine-tuning with focal loss:

```bash
python train.py
```

To run the anomaly detection framework:

```bash
python train_kd.py
```

### Run Interpretability Analysis
To run the interpretability analysis for the fine-tuned DNABERT-2 classifier:

```bash
python interpretability.py
```

To run the interpretability analysis for the anomaly detection framework:

```bash
python interpretability_kd.py
```
