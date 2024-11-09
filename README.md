# Cluster-implement
Cluster implement without library (like scikit-learn)

### requirements installation
```bash
pip install argparse numpy matplotlib scikit-learn
```
```bash
pip install numpy matplotlib scikit-learn
```
```bash
[notice] A new release of pip is available: 24.2 -> 24.3.1
[notice] To update, run: C:\Users\sy020\AppData\Local\Programs\Python\Python312\python.exe -m pip install --upgrade pip

C:\Users\sy020\Documents\GitHub\Cluster-implement\ai-assignment1-skeleton>cd C:\Users\sy020\AppData\Local\Programs\Python\Python312\
C:\Users\sy020\AppData\Local\Programs\Python\Python312>python.exe -m pip install --upgrade pip
```

#### Run

- k-means, GMM
```bash
C:\Users\sy020\Documents\GitHub\Cluster-implement\ai-assignment1-skeleton>c:\users\sy020\appdata\local\programs\python\python312\python.exe "c:\Users\sy020\Documents\GitHub\Cluster-implement\ai-assignment1-skeleton\clustering.py" --data_type 0 --cluster_method 0 --cluster_num 3
```
- DBSCAN
```bash
C:\Users\sy020\Documents\GitHub\Cluster-implement\ai-assignment1-skeleton>c:\users\sy020\appdata\local\programs\python\python312\python.exe "c:\Users\sy020\Documents\GitHub\Cluster-implement\ai-assignment1-skeleton\clustering.py" --data_type=1 --cluster_method=2 --eps=0.2 --min_pts=15
```
