import gower
import pandas as pd

df = pd.DataFrame([[1,2.6,'A'],[12,5,'X'],[4,7,'A']])

print(gower.gower_matrix(df))