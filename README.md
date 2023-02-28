# Quantum-enhanced Support Vector Machines for Stellar Classification

![Logo](durchmusterung.png)

[TOC]

## Introduction

This project aims to investigate the feasibility of employing quantum-enhanced support vector machines (QSVMs) for stellar classification based on spectral data from the open source dataset\cite{Ku_KaggleStellar}. The core of the study involves designing and implementing a novel QSVM algorithm, which will be compared to traditional SVMs and Morgan–Keenan (MK) classification system, that is conventionally used in the field of stellar classification. Furthermore, the project seeks to explore the potential GPU acceleration techniques for the task of QSVM model training. Ultimately, this study will attempt to demonstrate the potential of quantum computing in enhancing the precision and efficiency of machine-learning approaches in astronomy.

You can find our project proposal [here](./QHack_Project_Proposal_2023.pdf).

## Quantum circuit

We employed a Second-order Pauli-Z evolution circuit, which utilizes two-qubit ZZ interaction to encode classical data into a quantum state. The circuit was repeated twice and entanglement was maximized. The QuantumKernel class was then used to compute the kernel function. The number of qubits is 6, which is equal to the number of features.



## Reference

