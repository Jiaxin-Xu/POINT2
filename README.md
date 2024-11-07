# POINT2
## POINT2 Database -- POlymer INformatics Training and Testing Database based on PI1M
### 1. Data
Data source of PI1M is from:
```
https://github.com/RUIMINMA1996/PI1M
```

Data source of MSA Gas Permeability is from:
```
https://doi.org/10.1016/j.xcrp.2024.102067
```

Data source of ND Simulation is from:
```
https://doi.org/10.1016/j.mtphys.2022.100850
```

### 2. Model
Add torch-molecule (under active development) as a submodule for graph-related models:
```
cd POINT2
git submodule add https://github.com/liugangcode/torch-molecule.git
```
When you want to update:
```
cd torch-molecule
git pull origin main  # or the relevant branch
cd ..
git add torch-molecule
git commit -m "Update torch-molecule submodule"
```