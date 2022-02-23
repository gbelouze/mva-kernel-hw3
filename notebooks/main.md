---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: kernel-hw3
  language: python
  name: kernel-hw3
---

# Support Vector Machines

```{code-cell} ipython3
import numpy as np
import pickle as pkl
from scipy import optimize
import matplotlib.pyplot as plt

from hw3.utils import plotClassification, plotRegression
```

```{code-cell} ipython3
%load_ext autoreload
%autoreload 2
```

## Loading the data

The file 'classification_datasets' contains 3 small classification datasets:

    - dataset_1: mixture of two well separated gaussians
    - dataset_2: mixture of two gaussians that are not separeted
    - dataset_3: XOR dataset that is non-linearly separable.

Each dataset is a hierarchical dictionary with the following structure:

        dataset = {'train': {'x': data, 'y':label}
                    'test': {'x': data, 'y':label}
                  }
The data $x$ is an $N$ by $2$ matrix, while the label $y$ is a vector of size $N$.

```{code-cell} ipython3
with open('../data/classification_datasets', 'rb') as file:
    datasets = pkl.load(file)
fig, ax = plt.subplots(1,3, figsize=(20, 5))
for i, (name, dataset) in enumerate(datasets.items()):

    plotClassification(dataset['train']['x'], dataset['train']['y'], ax=ax[i])
    ax[i].set_title(name)
```

## III- Kernel SVC
### 1- Implementing the Gaussian Kernel
Implement the method 'kernel' of the class RBF below, which takes as input two data matrices $X$ and $Y$ of size $N\times d$ and $M\times d$ and returns a gramm matrix $G$ of shape $N\times M$ whose components are $k(x_i,y_j) = \exp(-\Vert x_i-y_i\Vert^2/(2\sigma^2))$. (The fastest solution does not use any for loop!)

```{code-cell} ipython3
from hw3.models.kernel import RBF
from hw3.models.kernel import Linear
```


### 2- Implementing the classifier
Implement the methods 'fit' and 'separating_function' of the class KernelSVC below to learn the Kernel Support Vector Classifier.

```{code-cell} ipython3
from hw3.models.classify import KernelSVC
```

### 2- Fitting the classifier

Run the code block below to fit the classifier and report its output.

```{code-cell} ipython3
fig, ax = plt.subplots(1,3, figsize=(20, 5))

C = 10000.
kernel = Linear()
model = KernelSVC(C=C, kernel=kernel)
train_dataset = datasets['dataset_1']['train']
model.fit(train_dataset['x'], train_dataset['y'])
plotClassification(train_dataset['x'], train_dataset['y'], model, label='Training', ax = ax[0])

C = 10.
model = KernelSVC(C=C, kernel=kernel)
train_dataset = datasets['dataset_2']['train']
model.fit(train_dataset['x'], train_dataset['y'])
plotClassification(train_dataset['x'], train_dataset['y'], model, label='Training', ax = ax[1])

sigma = 1.5
C=100.
kernel = RBF(sigma)
model = KernelSVC(C=C, kernel=kernel)
train_dataset = datasets['dataset_3']['train']
model.fit(train_dataset['x'], train_dataset['y'])
plotClassification(train_dataset['x'], train_dataset['y'], model, label='Training', ax=ax[2])

plt.savefig("../reports/figures/classifier.png", bbox_inches="tight")
```

# Kernel Regression

+++

## Loading the data

```{code-cell} ipython3
file = open('../data/regression_datasets', 'rb')
datasets = pkl.load(file)
file.close()
train_set = datasets['dataset_1']['train']
train_set = datasets['dataset_1']['test']
plotRegression(train_set['x'], train_set['y'],Y_clean= train_set['y_clean'])
```

## Kernel  Support Vector Regression
### 1- Implementing the regressor
Implement the method 'fit' of the classes KernelSVR below to perform Kernel Support Vector Regression.

```{code-cell} ipython3
from hw3.models.classify import KernelSVR
```

### 2- Fitting the regressor

Run the code block below to fit the regressor and report its output.

```{code-cell} ipython3
sigma = 0.2
C = 10.
kernel = RBF(sigma)
model = KernelSVR(C,kernel, eta= .1, epsilon = 1e-6)
model.fit(train_set['x'].reshape(-1,1),train_set['y'])
plotRegression(
    train_set['x'],
    train_set['y'],
    Y_clean=train_set['y_clean'],
    model=model,
    label='Train')

plt.savefig("../reports/figures/regressor.png", bbox_inches="tight")
```
