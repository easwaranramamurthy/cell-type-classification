# cell-type-classification
Cell type classification

# Description of code:


1. `01_preprocess_sc_data.ipynb` - preprocessing of single cell data using scanpy, ranking genes, selecting top k for rank value encoding to transformer model.
* For the base version, it selects the top 2048 genes and it appends a `<CLS>` token to the beginning.
* For the version with the previous time point, it selects the top 1024 gene indexes for the current time point and the top 12024 genes at the previous time point. It then concatenates these two index lists with a `<SEP>` token in the middle. It again appends a `<CLS>` token at the beginning of the token list.
* Finally, the preprocessing includes additional splitting of the test set into two sets (val and test) for robust evaluation/hyperparameter tuning etc.

2. `models/self_attention.py` - implements a single dot product self attention layer with no positional encoding

3. `models/mlp_classifier.py` - implements a single hidden layer multi-layer perceptron classifier.
* This is used later by the model to do cell type classification on the `<CLS>` token embedding output by the final layer.

4. `models/transformer.py` -  implements a full transformer model with multiple self attention layers.

5. `dataset/npy_dataset.py` - implements a pytorch dataset that reads from memory mapped numpy arrays.

6. `02_train_model.py` - jointly trains a full transformer model and an mlp model which classifiers cell types using the `<CLS>` token embedding

7. WandB runs are available on a public link here - https://wandb.ai/easwaran/cell_type_classification

8. TODO: `03_compile_evaluation_metrics.ipynb` - notebook for compilation of evaluation metrics and comparing all trained models. 


# Comparing Geneformer and scGPT:

The main differences between Geneformer and scGPT are:

## 1. Tokenization strategy:
* Geneformer uses rank value encoding - it normalizes raw gene expression counts by the total counts in the cell and then normalizes each gene by its median gene expression value across all cells. Each cell is represented as an unordered list of indexes of its top 2048 genes.

* scGPT first selects M highly variable genes and uses an unordered list of the indexes of each gene as the main input to the model. It then does log1p transformation and bins the genes into B bins for each cell. It then feeds in the bin number as an additional context token. This context token gets its own embedding and is aggregated with the embedding of the gene index before feeding into the attention layer. It can also incorporate additional context tokens such as batch number, modality, perturbation condition etc. It also incorporates special tokens like `<cls>` to pool gene embeddings and do cell level classification and <pad> to represent short inputs.

### Pros and Cons:
Geneformer's normalization approach makes sure that housekeeping genes that are always highly expressed in most cell types have lower influence on the gene rankings and consequently upweights genes that have more information about cellular identity. In contrast, scGPT may upweight housekeeping genes which carry limited value in defining cell state.

However, since Geneformer only operates on ranks, it discards quantitative information contained in scRNA-seq cell x gene matrices which are important for many tasks. The binning strategy in scGPT ensures that some quantitative information about a gene's expression is seen by the model even though it is still unable to use all of the quantitative information in the cell x gene matrix. scGPT is also able to incorporate additional context tokens such as batch, modality etc. which can help the model to better learn real determinants of cell state rather than experimental artifacts.

Geneformer needs additional fine-tuning on top of the model embeddings for cell level tasks, whereas the incorporation of additional tokens like `<cls>` means that scGPT can be directly trained to do these tasks.

## 2. Encoder only vs GPT style autoregressive generation
* Geneformer uses an encoder only architecture with 6 attention layers. During training, 15% of the genes are masked and the model is asked to re-identify the masked genes by taking into account all unmasked genes in the input. 
* scGPT uses a decoder module as well and is trained in an autoregressive manner to predict new genes that are unknown from genes that are known or already predicted + additional tokens like `<cls>`. A random number of genes are chosen as unknown although it is unclear from the paper whether the random sample is generated per cell or across the corpus.

### Pros and Cons:
The causal masking in scGPT and unidirectional nature of generation is not ideal for single cell data where there is no ideal ordering of genes. Such sort of generation would be useful if genes are generated in a ranked manner but the order of input to scGPT is randomized and hence, it doesn't seem necessary.

## 3. Corpus:
* Geneformer is trained on 30M single cells from a broad range of human tissues so it can learn gene network dynamics across a large number of tissues/cell types.
* scGPT is trained on 10M single cells from blood/bone marrow but the incorporation of context tokens allows it to be trained quickly on new cell types/tissues.

## 4. Other differences:
* scGPT uses flash attention to speed up the attention computation
* (writing more)


# TODO:
1. compilation of evaluation metrics into a notebook.