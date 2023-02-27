

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# General Imports
import numpy as np
import time
import pandas as pd
from functools import reduce

# Visualisation Imports
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit Imports
from sklearn.preprocessing import *
from sklearn.decomposition import *
from sklearn.model_selection import *
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Qiskit Imports
from qiskit import Aer, execute
from qiskit.circuit import QuantumCircuit, Parameter, ParameterVector
from qiskit.circuit.library import PauliFeatureMap, ZFeatureMap, ZZFeatureMap
from qiskit.circuit.library import TwoLocal, NLocal, RealAmplitudes, EfficientSU2
from qiskit.circuit.library import HGate, RXGate, RYGate, RZGate, CXGate, CRXGate, CRZGate
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit import QuantumCircuit, transpile, Aer, IBMQ
from qiskit.tools.jupyter import *
from qiskit.visualization import *
#from ibm_quantum_widgets import *
from qiskit.providers.aer import QasmSimulator
from qiskit import *
from qiskit.opflow import *
from qiskit.utils import *
from qiskit.circuit import *
from qiskit.circuit.library import *
from qiskit.algorithms.optimizers import *
from qiskit_machine_learning.neural_networks import *
from qiskit_machine_learning.algorithms.classifiers import *
from qiskit_machine_learning.algorithms.regressors import *
from qiskit_machine_learning.datasets import *

from typing import Union

from qiskit_machine_learning.exceptions import QiskitMachineLearningError

from qiskit import BasicAer
from qiskit.algorithms.state_fidelities import ComputeUncompute
from qiskit.circuit.library import ZZFeatureMap
from qiskit.primitives import Sampler
from qiskit.utils import algorithm_globals
from qiskit_machine_learning.algorithms import QSVC
#from qiskit_machine_learning.kernels import FidelityQuantumKernel

seed = 12345
algorithm_globals.random_seed = seed

# Loading your IBM Quantum account(s)
# provider = IBMQ.load_account()

## Import the sata set
stars_df = pd.read_csv('Star3642_balanced.csv')
null, nan = stars_df.isnull().sum() , stars_df.isna().sum()
stars_df['SpType'].unique

## Removing the rows with too much error
threshold = stars_df['e_Plx'].mean() + 0.5
stars_dff = stars_df[ stars_df['e_Plx'] < threshold ]

## feature engineering
stars_df_eng = stars_dff.copy()
stars_df_eng['Amag_SQ'] = stars_df_eng['Amag']**2
stars_df_eng['Vmag_SQ'] = stars_df_eng['Vmag']**2
stars_df_eng['B-V_SQ'] = stars_df_eng['B-V']**2