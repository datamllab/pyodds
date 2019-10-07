import datetime
import pandas as pd
import numpy as np
import time

epochs = 10000
todays_date = datetime.datetime.now().date()

columns = ['ts','a','b', 'c','d','e']
trange = pd.date_range('2019-10-01', periods=epochs, freq='1min')

ts = pd.DataFrame(columns=columns)
current_time = time.clock()
for i in range(epochs):
    ts=ts.append({'ts':trange[i],'a':np.random.uniform(low=-4, high=4),'b':np.random.uniform(low=-4, high=4),'c':np.random.uniform(low=-4, high=4),'d':np.random.uniform(low=-4, high=4),'e':np.random.uniform(low=-4, high=4)},ignore_index=True)
ts.to_csv('pandas10000.csv')
print ('Total inserting cost: %.6f s' %(time.clock() - current_time))
