import pandas as pd
import time

current_time = time.clock()
@profile
def read():
    d = pd.read_csv('pandas100000.csv')

    d['ts']=pd.to_datetime(d['ts'])
    d = d.set_index(d['ts'])
    _=d.iloc[200:500]
read()
print('Total inserting cost: %.6f s' % (time.clock() - current_time))
