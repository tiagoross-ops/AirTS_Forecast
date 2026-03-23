from HUMAN_MADE_FILES.Data_Loading import *
from HUMAN_MADE_FILES.environmental_data_conversion import *

monthly_list = []
for month in range(3,13):
    year = 2004
    day_start = 1
    day_end = 31
    directory = era5_land_importing_by_date(year, month, day_start, day_end)
    data = environmental_data_conversion_era5_to_dfs(directory)
    monthly_list.append((month, data))
monthly_data_dict = dict(monthly_list)
#print(monthly_data_dict)

#with open('Data_file.txt','w',encoding='utf-32') as file:
#    for month, data in monthly_dict.items():
#        file.write(f'Month {month}')
#        file.write(data)

with pd.HDFStore("era5_monthly.h5", mode="w", complevel=6, complib="blosc") as store:
    for month, data_dict in monthly_data_dict.items():
        for var, data in data_dict.items():
            key = f"month_{month}_{var}"   # e.g. "
            store.put(key, data, format="table")          # format="table" allows querying

    store.close()
