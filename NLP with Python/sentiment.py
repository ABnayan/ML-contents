# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 17:29:01 2020

@author: P70002567
"""

import numpy as np
import math
import re
import pandas as pd
from bs4 import BeautifulSoup

#pip install google-colab
#from google.colab import drive

#try:
#    %tensorflow_version 2.x
#except Exception:
#    pass

import tensorflow as tf

from tensorflow.keras import layers

#pip install tensorflow_datasets
import tensorflow_datasets as tfds

#drive.mount("/content/drive")


cols = ["sentiment", "id", "date", "query", "user", "text"]
train_data = pd.read_csv(
    "C:\Users\P70002567\Desktop\NLP with Python\data/train.csv",
    header=None,
    names=cols,
    engine="python",
    encoding="latin1")
