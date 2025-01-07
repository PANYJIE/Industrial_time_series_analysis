This is an algorithm library dedicated to industrial time series data analysis, including 5 types of algorithms, 20 algorithms, 
experimentally verified on actual industrial production data and public datasets, and 25 algorithm instances are formed, as follows
| Type     | Algorithm                                                        |
|----------|------------------------------------------------------------------|
| Describe | MICAD,MOCAR,RBS,TSCA                                             |
| Decide   | Il_Std,Qcd,SDE_DK                                                |
| Dianosse | MCFMAAE,MGAHGM                                                   |
| Forecast | MSNET,PID4LaTe,STD_Phy,STDNet,TDG4MSF,CGRAN,MMPNN,MCRN,TALS,MANO |
| Control  | PMCCL                                                            |

# Quick Install

We recommend to first setup a clean Python environment for your project with Python 3.8+ using conda.
Once your environment is set up you can install darts using pip:
``` python
pip install Industrial_time_series_analysis
```
# Dependencies

Python(>=3.8)
Torch(>=1.12.0)
Numpy(>=1.19.5)
threadpoolctl(>=3.1.0)
Scipy(>=1.6.0)
