# cell-type-classification
Cell type classification

# Description of code:


* `00_train_model.ipynb` - this notebook is currently set up to train an encoder only transformer model on randomly generated data

* `01_preprocess_sc_data.ipynb` - preprocessing of single cell data using scanpy, ranking genes, selecting top k for rank value encoding to transformer model, and appending `<CLS>` token to beginning

* `models/self_attention.py` - implements a single dot product self attention layer with no positional encoding

* `models/mlp_classifier.py` - implements a single hidden layer multi-layer perceptron classifier

* `models/transformer.py` -  implements a full transformer model with multiple self attention layers and a MLP classifier that takes the embedding of 0th token (`<CLS>`) output by the last self attention layer as the input and classifies it into cell types 

# TODO:
1. plug in single cell data into training pipeline, optimize hyperparameters/arch so that model can be trained efficiently on M1 Macbook, run evaluations on test data
2. do a full write up of preprocessing steps taken, model architecture, training set up, and model performances
3. integration of `X_previous` - my idea is to append a `<SEP>` token + rank value encoding of the cell from previous time point to the rank value encoding of the cell from current time point and then train another model. This will preserve the pairing between the cells at the two time points.