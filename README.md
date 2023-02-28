# Quantum-enhanced Support Vector Machines for Stellar Classification

![Logo](durchmusterung.png)

[TOC]

Table of Contents

1. [Introduction](#introduction)
2. [Feature Engineering](#Feature Engineering)

## Introduction

This project aims to investigate the feasibility of employing quantum-enhanced support vector machines (QSVMs) for stellar classification based on spectral data from the open source dataset\cite{Ku_KaggleStellar}. The core of the study involves designing and implementing a novel QSVM algorithm, which will be compared to traditional SVMs and Morgan–Keenan (MK) classification system, that is conventionally used in the field of stellar classification. Furthermore, the project seeks to explore the potential GPU acceleration techniques for the task of QSVM model training. Ultimately, this study will attempt to demonstrate the potential of quantum computing in enhancing the precision and efficiency of machine-learning approaches in astronomy.

You can find our project proposal [here](./QHack_Project_Proposal_2023.pdf).

## Feature Engineering

In order to facilitate quantum kernel learning, classical pre-processing methods can be utilized to decrease the dimensionality of the data or extract pertinent features prior to quantum kernel training. This approach can decrease the computational load and enhance the model's accuracy by decreasing noise and uncertainty associated with the quantum data. A technique for preparing the data for quantum kernel learning is to convert the quantum state into a different representation that is more suitable for machine learning methods. This may involve applying quantum circuits or other transformations to the data to extract valuable features.

There are 6 attributes provided in the original dataset. We believe only 2 of them, `B-V` and `Amag`, are relevant to the classification. They represents the B-V color index and absolute magnitude of the star, respectively. In addition,  we defined 4 more composite attributes. They are:
```math
\begin{align}
		\texttt{Amag\_SQ} = \texttt{Amag}^2\\
    \texttt{B-V\_SQ} = \texttt{B-V}^2\\
    \texttt{B-V+Amag} = \texttt{B-V} + \texttt{Amag}\\
    \texttt{B-V-Amag} = \texttt{B-V} - \texttt{Amag}
\end{align}
```


The relation between those 6 attributes and the classification are shown in the figures below.

![relation1](./assets/relation1.png)

![relation2](./assets/relation2.png)

![relation3](./assets/relation3.png)



## Quantum circuit

We employed a Second-order Pauli-Z evolution circuit, which utilizes two-qubit ZZ interaction to encode classical data into a quantum state. The circuit was repeated twice and entanglement was maximized. The QuantumKernel class was then used to compute the kernel function. The number of qubits is 6, which is equal to the number of features.

## Results

Below, we present the results of four measures - accuracy, F1 index, specificity, and sensitivity - to demonstrate the effectiveness of our QSVM model. For comparison, we also include the corresponding measures obtained from the classical SVM (CSVM) model in the same figures.

From the figures we can find that QSVM has the trend to be overall more accurate than the CSVM. When the size of dataset is small, all measures fluctuate dramatically. 
<div align=center><img src="./assets/accuracy_qsvm.png" alt="accuracy_qsvm" width="500" />
<div align=center><img src="./assets/f1_qsvm.png" alt="f1_qsvm" width="500" />
<div align=center><img src="./assets/specificty_qsvm.png" alt="specificty_qsvm" width="500" />
<div align=center><img src="./assets/sensitivty_qsvm.png" alt="sensitivty_qsvm" width="500" />





## Reference



