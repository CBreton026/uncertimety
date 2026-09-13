# uncertimety
Scripts, notebooks and data to question common lifetime assumptions for residential buildings

To reproduce the results:
- Run dataprep.py, which outputs 'data/clean/fulldata.parquet'
- Run interpolate.py, which outputs 'data/clean/reconciled.parquet'
- Using both files, run the dmfa-lifetime.ipynb notebook.
