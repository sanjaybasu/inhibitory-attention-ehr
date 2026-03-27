# Data Access

## MedAlign (Experiments 1 & 3)

MedAlign is available under a data use agreement from the Stanford AIMI Center.

Request access: https://stanfordaimi.azurewebsites.net/datasets/0aad76c8-cd50-4b54-8950-3e5d78b1a59b

Once approved, download the `MedAlign_files/` directory and set the `BASE` path in
`experiments/exp1_medalign_litm.py` and `experiments/exp3_qccs_gate.py`.

## EHRSHOT (Experiment 2)

EHRSHOT is available via PhysioNet after signing the credentialing agreement.

Request access: https://physionet.org/content/ehrshot/1.0/

Set the data path in `experiments/exp2_ehrshot_diffattn.py` accordingly.

## Trained Gate Weights

The trained QCCS gate (`qccs_gate.pt`) produced by `exp3_qccs_gate.py` is
provided in `figures/qccs_gate.pt` in this repository to enable inference
without re-training (re-training requires MedAlign access).
