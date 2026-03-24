# Likert Scale Survey Visualizer

**Author:** Muhammad Hassaan Naeem  
**Role:** ML Researcher | Lab Engineer @ SCAT  
**LinkedIn:** [muhammad-hassaan-naeem](https://www.linkedin.com/in/muhammad-hassaan-naeem-b51610354/)  
**Based on:** [Elsevier ASEJ 2024](https://doi.org/10.1016/j.asej.2024.102794) · [ASCE Inspire 2023](https://doi.org/10.1061/9780784485163.016)

---

## What is this?

A command-line tool that takes **any Likert-scale CSV survey dataset** and automatically generates **8 professional, publication-ready visualizations** — with zero manual setup.

Built from real research experience analyzing construction industry survey data for peer-reviewed publications.

---

## Charts Generated
<img width="3054" height="2298" alt="00_dashboard" src="https://github.com/user-attachments/assets/698c0397-3329-4c34-92c0-d93cd56ceb05" />

| File | Chart Type | What it shows |
|---|---|---|
| `00_dashboard.png` | Combined dashboard | All key charts in one figure |
| `01_mean_scores.png` | Grouped bar chart | Mean ± SD per construct |
| `02_diverging_stacked.png` | Diverging stacked bar | Full response distribution |
| `03_radar_chart.png` | Radar / Spider chart | Construct means at a glance |
| `04_item_heatmap.png` | Heatmap | Every individual item mean |
| `05_distributions.png` | Histogram + KDE | Score distribution per construct |
| `06_boxplots.png` | Box plot | Spread and outliers per construct |
| `07_demographic_breakdown.png` | Horizontal bar | Scores by respondent group |

---

## Installation

```bash
git clone https://github.com/muhammad-hassaan-naeem/likert-scale-visualizer
cd likert-scale-visualizer
pip install -r requirements.txt
```

---

## Usage

### Option 1 — Auto-detect constructs (easiest)
```bash
python likert_visualizer.py --file your_survey.csv
```
The tool will automatically group columns like `SC1, SC2, SC3` → `SC` construct.

### Option 2 — Define constructs manually (recommended)
```bash
python likert_visualizer.py --file your_survey.csv --constructs config.json
```

Edit `config.json` to match your column names:
```json
{
    "Strategic":    ["SC1", "SC2", "SC3", "SC4"],
    "Technical":    ["TC1", "TC2", "TC3", "TC4"],
    "Financial":    ["FC1", "FC2", "FC3", "FC4"],
    "Performance":  ["OP1", "OP2", "OP3", "OP4"]
}
```

### Full options
```bash
python likert_visualizer.py \
    --file      sample_survey.csv \
    --constructs config.json \
    --output    my_charts \
    --scale     5
```

| Argument | Description | Default |
|---|---|---|
| `--file` | Path to your CSV file | Required |
| `--constructs` | Path to JSON config | Auto-detect |
| `--output` | Output folder name | `output_charts` |
| `--scale` | Likert scale size | `5` |

---

## CSV Format

Your CSV should have:
- One row per respondent
- One column per survey item (Likert responses as numbers)
- Optional demographic columns (text values like "Male", "5–10 yrs")

```
Respondent,SC1,SC2,SC3,SC4,TC1,TC2,TC3,TC4,OP1,OP2,Experience,Role
1,4,5,4,3,3,4,4,3,4,4,5-10 yrs,Manager
2,3,3,4,4,4,4,3,4,3,3,2-5 yrs,Engineer
```

---

## Sample Output

Run the included sample dataset:
```bash
python likert_visualizer.py --file sample_survey.csv --constructs config.json
```

This uses synthetic data mirroring the construction industry survey from the published papers.

---

## Requirements

```
numpy>=1.21
pandas>=1.3
matplotlib>=3.4
seaborn>=0.11
scipy>=1.7
```

Install with:
```bash
pip install -r requirements.txt
```

---

## Repository Structure

```
likert-scale-visualizer/
│
├── likert_visualizer.py   # Main script — all charts
├── config.json            # Example construct definitions
├── sample_survey.csv      # Sample dataset (30 respondents, 6 constructs)
├── requirements.txt
└── README.md
```

---

## Research Background

This tool was developed from hands-on survey research experience at COMSATS University Islamabad, resulting in two peer-reviewed publications:

- **Ain Shams Engineering Journal, Elsevier · 2024** — DOI: [10.1016/j.asej.2024.102794](https://doi.org/10.1016/j.asej.2024.102794)
- **ASCE Inspire Conference Proceedings · 2023** — DOI: [10.1061/9780784485163.016](https://doi.org/10.1061/9780784485163.016)

---

## License

MIT License — free to use, modify, and share with attribution.

---

⭐ If this tool saved you time, please star the repository!
