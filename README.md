# curiosity-ltv-id


**Dependencies**

```bash
# create & activate your conda env (optional but recommended)
conda create -n curiosity-ltv-id python=3.10
conda activate curiosity-ltv-id

# install requirements one by one or via:
pip install -r requirements.txt
```

**Running the CLI**

```bash
# from project root
python main.py <CHANNEL_ID> <Y_CHANNEL_INDEX> [--split train|test]
# e.g.
python main.py M-6 5
```

**In-Notebook Exploration**

1.	Launch JupyterLab:

```bash
jupyter lab notebooks/01_explore.ipynb
```

2. Ensure the first cell prepends the project root to sys.path

3.	Run the cells top-to-bottom to reproduce the sliding‐window ARX analysis.

**Plotting Results**

In any Python REPL or notebook (with ROOT & processed_dir set):

```python
from src.validate import plot_channel_results
plot_channel_results("M-6", win_size=50, hop=10, out_dir="data/processed")
```