
# GNN for Molecular Property Prediction

This project leverages Graph Neural Networks (GNNs) to predict molecular properties using structured molecular data. The model is built using PyTorch and PyTorch Geometric, with various preprocessing and utility tools like RDKit and ASE.

##  Project Overview

The notebook trains a GNN model to learn molecular features and predict a target property (such as dipole moment, energy, etc.). The molecules are treated as graphs, where atoms are nodes and bonds are edges.

##  Project Structure

- `GNN_for_Molecular_property_prediction.ipynb`: The main Jupyter notebook containing code for data processing, model definition, training, and evaluation.
- `requirements.txt`: Lists all required Python packages to run the notebook.
- `README.md`: This file, explaining the architecture, setup, and usage.

##  Model Architecture

The architecture used in this project is based on **SchNet**, a powerful Graph Neural Network (GNN) tailored for molecular property prediction. It processes molecules as graphs where atoms are nodes and interactions are captured via learned continuous filters based on interatomic distances.

###  Pipeline Overview

```
[Input Molecules (XYZ via ASE)]
        ↓
[Atom & Bond Featurization]
        ↓
[Graph Construction (atoms = nodes, edges = distances)]
        ↓
[SchNet GNN]
   ├── Atom Embedding Layer
   ├── 6 × Interaction Blocks (Message Passing)
   ├── Continuous Filter Convolutions (cutoff = 10 Å)
   └── Atom-wise Readout Layer
        ↓
[Pooling → Fully Connected Layers]
        ↓
[Predicted Molecular Property]
```

###  Model Configuration

```python
model = SchNet(
    hidden_channels=128,
    num_filters=128,
    num_interactions=6,
    cutoff=10.0
).to(device)
```

- **hidden_channels**: Dimensionality of hidden atomic representations.
- **num_filters**: Number of continuous convolution filters.
- **num_interactions**: Number of interaction (message-passing) layers.
- **cutoff**: Interatomic distance threshold for considering interactions (in Ångströms).

### Learning Setup

- **Optimizer**: Adam (`lr=1e-3`)
- **Loss Function**: Mean Squared Error Loss (`MSELoss`)
- **Device**: Runs on GPU (`cuda`) if available, otherwise CPU.

This setup allows the model to effectively learn from spatial molecular data and generalize well to property prediction tasks like energy, dipole moment, and more.

##  Dataset

Molecular datasets like QM9 or custom XYZ format data are used. Molecules are parsed and converted into graph structures using tools like ASE or RDKit.
link for the data set - [link](https://www.kaggle.com/competitions/molecular-property-prediction-challenge/data)

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

- **Harsh** – [GitHub](https://github.com/harshu0117)

---

This project is ideal for anyone interested in combining deep learning and chemistry using graph-based approaches. Contributions and suggestions are welcome!
