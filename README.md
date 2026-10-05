# Knowledge Representation: Practical Work

Practical work (TPs) for the Knowledge Representation module, implemented in Python.

## Repository structure

```
.
├── TP1/            # TODO: topic of TP1
├── TP2/            # TODO: topic of TP2
├── TP3/            # TODO: topic of TP3
├── Rapport/        # Written report
├── requirements.txt
└── README.md
```


## Tech stack

| Library | Purpose |
|---------|---------|
| `scikit-fuzzy` | Fuzzy logic |
| `pgmpy` | Probabilistic graphical models / Bayesian networks |
| `pyds` | Dempster–Shafer theory of evidence |
| `numpy`, `scipy` | Numerical computation |
| `matplotlib` | Plots and visualizations |

## Getting started

Requires Python 3.9 or later (the pinned versions in `requirements.txt` target recent Python 3 releases).

```bash
git clone https://github.com/sarahmoussaoui/Knowledge-Representation-Practical-Work.git
cd Knowledge-Representation-Practical-Work

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running the TPs

```bash
cd TP1
python <main_file>.py           # TODO: replace with the real file name
```

Repeat for `TP2` and `TP3`.

## Report

The written report is in the [`Rapport`](Rapport) folder.
