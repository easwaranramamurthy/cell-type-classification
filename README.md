# cell-type-classification
Cell type classification

# Description of code:


* `01_preprocess_sc_data.ipynb` - preprocessing of single cell data using scanpy, ranking genes, selecting top k for rank value encoding to transformer model, and appending `<CLS>` token to beginning, splitting of test set into two sets - val and test

* `models/self_attention.py` - implements a single dot product self attention layer with no positional encoding

* `models/mlp_classifier.py` - implements a single hidden layer multi-layer perceptron classifier

* `models/transformer.py` -  implements a full transformer model with multiple self attention layers and a MLP classifier that takes the embedding of 0th token (`<CLS>`) output by the last self attention layer as the input and classifies it into cell types 

* `dataset/npy_dataset.py` - implements a pytorch dataset that reads from memory mapped numpy arrays

* `02_train_model.py` - trains a full transformer model on the single cell training dataset

* WandB runs are available on a public link here - https://wandb.ai/easwaran/cell_type_classification

# TODO:
1. do a full write up of geneformer/scGPT comparison, preprocessing steps taken, model architecture, training set up, and model performances
2. integration of `X_previous` - my idea is to append a `<SEP>` token + rank value encoding of the cell from previous time point to the rank value encoding of the cell from current time point and then train another model. This will preserve the pairing between the cells at the two time points.