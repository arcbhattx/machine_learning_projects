import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix, roc_curve, auc


#load the dataset

diabetes = load_diabetes()
x,y = diabetes.data, diabetes.target

#convert target variables to binary 0 or 1 (1 for d, 0 for no)
y_binary = (y > np.median(y)).astype(int)

