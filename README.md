# CycleVI

## Overview
This repository contains the source code for the CycleVI method, as presented in [our preprint](https://doi.org/10.1101/2025.11.04.686009)

## Dependencies

* scvi-tools
* anndata
* PyTorch

## Structure

* [CycleVI_model](CycleVI_model.py): model implementation.
* [Tutorial](Tutorial.ipynb): a python notebook with ans example on how to run the model.
* [Tutorial_colab](Tutorial_colab.ipynb): same tutorial, but ready to run on Google Colab.

## Usage
To use CycleVI, simply import the model from the ``CycleVI_model.py`` file.
```python
from CycleVI_model import CycleVI
```

Instructions on how to use the model are present in the ``Tutorial.ipynb`` notebook.

## Feedback
For questions and comments, feel free to contact [Gustavo S. Jeuken](mailto:g.stolfjeuken@vu.nl).

## License
BSD 3-Clause License

## Citation
If you use this model in a publication, please cite [our preprint](https://doi.org/10.1101/2025.11.04.686009)

>CycleVI: Isolating cell cycle variation with an interpretable deep generative model
>
>Pia Mozdzanowski, Marcel Tarbier, Gustavo S. Jeuken
>
>bioRxiv 2025.11.04.686009; doi: https://doi.org/10.1101/2025.11.04.686009
