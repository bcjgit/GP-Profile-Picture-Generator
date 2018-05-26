# Github Profile Picture Generator

Python script used to generate my github profile picture. This script computes a draw from a Gaussian process of user specified dimension and displays/saves that image on request. 

## Arguments
| Argument      | Purpose       | Required  |
| ------------- |:-------------:| -----:|
| --dim         | Dimenson of GP you want to draw | Yes |
| --verbose      | Flag indicating if script should plot image      |   No |
| --save | Flag indicating if script should save image      |    No |

## Example Usage
```bash
python gen_pic.py --dim 400 --verbose 
```
This will draw a 400x400 GP and plot , but not save, the resulting image. 
