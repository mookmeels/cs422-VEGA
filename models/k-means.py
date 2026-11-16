

import polars as pl
from polars import read_csv, selectors, col


from sklearn import model_selection

# say df is a dataframe of all the info

# goal: 

### training section

dataPath = ".../" 

FEATURE_COUNT = 1000

df = read_csv(dataPath, separator=",", has_header=False) # contains image vectors


@dataclass
class Point:
    trainingIndex: int
    distance: float
    className: str

def find_distance(p1, p2, ind, p2Class):
    distance = 0
    for i in range(FEATURE_COUNT):
        distance += pow(p1[i] - p2[i], 2)
    
    p = Point(distance=pow(distance, 0.5), className=p2Class, trainingIndex=ind)
    return p





### verification section