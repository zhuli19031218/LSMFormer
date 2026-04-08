# LSMFormer: Landslide Susceptibility Mapping 

This repository contains the official implementation of the LSMFormer model for Landslide Susceptibility Mapping.

## Repository Structure

- `main_whole.py`: Main entry point for training and evaluation.
- `train_whole.py`: Core training loop and evaluation logic.
- `utils_whole.py`: Helper functions, data scaling, evaluation metrics, and plotting.
- `CL_GAF_probAtten_Res_for_RC.py`: Core model definitions (BYOL, Encoders, etc.).
- `model/`: Additional network architectures utilized by the codebase.
- `cache_data/`: Directory for storing preprocessed data caches.
- `logs/`: Directory for test results, trained weights, and training metrics (viewable via TensorBoard).

## Requirements

The codebase assumes the following main dependencies:
- Python 3.8+
- PyTorch
- pandas, numpy
- scikit-learn
- imbalanced-learn (`imblearn`)
- plotly, matplotlib, seaborn
- thop
- missingno
- torchsummary

You can typically install necessary packages via:
```bash
pip install torch pandas numpy scikit-learn imbalanced-learn plotly matplotlib seaborn thop missingno torchsummary
```

## Dataset

Expected dataset paths are defined in `main_whole.py` (e.g., `train_data`, `test_data`, `neg_data`). You need to prepare your own landslide dataset mimicking the expected columnar structure, or adjust the path logic in the main file accordingly.

## Quick Start

Execute the pre-training, fine-tuning, and testing sequence by simply running:

```bash
python main_whole.py
```

The script execution involves:
1. **SMOTE Oversampling**: Preparing balanced datasets.
2. **Contrastive Learning Pre-training**: Utilizing self-supervised mechanisms on the representation layer.
3. **Fine-Tuning**: Supervised tuning and generating final susceptibility scores.
4. **Evaluation**: Generating ROC curves and metrics reports saved into the timestamped folders in the `logs/` directory.
