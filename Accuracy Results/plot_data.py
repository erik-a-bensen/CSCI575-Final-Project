import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#read in data with indices
hmrf = pd.read_csv('hmrf_scores', index_col=0)
kmeans = pd.read_csv('kmeans_scores', index_col=0)
gmm = pd.read_csv('gmm_scores', index_col=0)

