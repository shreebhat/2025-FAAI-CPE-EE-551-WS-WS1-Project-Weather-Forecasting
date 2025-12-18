import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path.cwd().parent
sys.path.append(str(PROJECT_ROOT))


import weather.viz as vz

class DataAnalyzer:
    # Analyzes the data

    def analyze(self, data):
        
        print(data.info())

        x = data["DATE"]
        y = data["DailyAverageDryBulbTemperature"]
        #y = Y["y_next"]
        #print(modeldata.info())
        #x = modeldata[indVar].map(pd.Timestamp.toordinal)
        #y = modeldata[target]
        vz.dotPlot(x, y)
        