
# GNN for Molecular Property Prediction

This project leverages Graph Neural Networks (GNNs) to predict molecular properties using structured molecular data. The model is built using PyTorch and PyTorch Geometric, with various preprocessing and utility tools like RDKit and ASE.

##  Project Overview

The notebook trains a GNN model to learn molecular features and predict a target property (such as dipole moment, energy, etc.). The molecules are treated as graphs, where atoms are nodes and bonds are edges.

##  Project Structure

- `GNN_for_Molecular_property_prediction.ipynb`: The main Jupyter notebook containing code for data processing, model definition, training, and evaluation.
- `requirements.txt`: Lists all required Python packages to run the notebook.
- `README.md`: This file, explaining the architecture, setup, and usage.

##  Model Architecture

The model is based on a **Message Passing Neural Network (MPNN)** or related GNN variants from `torch_geometric`. The typical architecture includes:

- **Input Layer**: Converts molecular graphs into feature representations.
- **Graph Convolution Layers**: Learns node embeddings via message passing.
- **Pooling Layer**: Aggregates node embeddings to form a graph-level representation.
- **Fully Connected Layers**: Maps the pooled representation to property prediction.

The architecture is modular, allowing easy changes to layers or pooling mechanisms.

##  Dataset

Molecular datasets like QM9 or custom XYZ format data are used. Molecules are parsed and converted into graph structures using tools like ASE or RDKit.
link for the data set - [@link](https://www.kaggle.com/competitions/molecular-property-prediction-challenge/data)

##  How to Run

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/gnn-molecular-property.git
   cd gnn-molecular-property
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Notebook**:
   Open the `.ipynb` file in JupyterLab or VSCode and run all cells.

##  Requirements

All necessary packages are listed in `requirements.txt`. Major dependencies include:

- PyTorch
- PyTorch Geometric
- Pandas
- Scikit-learn
- TQDM
- ASE

##  Results

The model shows promising results in predicting molecular properties. Performance metrics like MAE or RMSE are computed on the validation/test sets.

##  Notes

- For large datasets or more accurate results, consider tuning hyperparameters and adding dropout/batch norm.
- CUDA support can be enabled if GPU is available.

##  Author

- **Harsh** – [@yourGitHub](https://github.com/yourGitHub)

---

This project is ideal for anyone interested in combining deep learning and chemistry using graph-based approaches. Contributions and suggestions are welcome!
