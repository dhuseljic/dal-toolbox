# %%
import sys
import json
import numpy as np
import pylab as plt
import pandas as pd
# %%
def load_json(path):
    with open(path, 'r') as f:
        data = json.loads(f.read())
    return data

def extract_results(data):
    return

def plot_table(table_contents):
    return
# %%
data = load_json("")