This is an algorithm library specifically tailored for analyzing industrial time series data, encompassing five categories of algorithms, totaling 20 individual algorithms. 
These algorithms have been experimentally validated using actual industrial production data and public datasets, ultimately resulting in the formation of 25 algorithm instances，as follows.
| Type            | Algorithm                                                        |
|----------       |------------------------------------------------------------------|
| Describe        | MICAD,MOCAR,RBS,TSCA                                             |
| Decision-making | Il_Std,Qcd,SDE_DK                                                |
| Diagnosis       | MCFMAAE,MGAHGM                                                   |
| Forecast        | MSNET,PID4LaTe,STD_Phy,STDNet,TDG4MSF,CGRAN,MMPNN,MCRN,TALS,MANO |
| Control         | PMCCL                                                            |

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
