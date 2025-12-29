Olivetti faces analysis

Files:
- `olivetti_analysis.py`: Main script to run dataset loading, stratified splits, KMeans clustering, visualization, classifier training, and experiments using KMeans as dimensionality reduction.
- `requirements.txt`: Python dependencies.

Quick start:
1. Create a virtual environment and activate it (Windows PowerShell):

   python -m venv .venv; .\.venv\Scripts\Activate.ps1

2. Install dependencies:

   pip install -r requirements.txt

3. Run the analysis (this will download the Olivetti dataset if not present):

   python olivetti_analysis.py --outdir output

Outputs:
- `output/results.json`: summary of validation and test accuracies
- `output/kmeans_vis/`: visualizations of cluster centers and example images per cluster

Notes:
- The notebook that originally referenced this task is `L06-Olivetti faces dataset.ipynb`.
- KMeans experiments try several cluster counts; increase or reduce the list in `olivetti_analysis.py` depending on how long you want experiments to run.
