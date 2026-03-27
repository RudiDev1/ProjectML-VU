# Group 51 - Vrije Universiteit Amsterdam Machine Learning Project

## Description
This repository shows programming code for the neural netowork implementation from scratch using numpy for 2026 Machine Learning Project. It includes implementation of simmilar neural network using PyTorch. Ensure after copying repository that you are in folder of repository for command line operations. 

## Creating enviorment
##### Important: For running this repository conda is recommended for creation of the environment as the authors did the same
```Bash
conda env create -f environment.yml
```
After creating enviorment run
```Bash
conda activate MLProject
```
## Cleaning Data
Cleaned data is provided, however to ensure correct results you may opt to do it. There will be prompt for showing of price distribution of the cleaned data.
```Bash
cd data
python cleaning_data_panda.py
```
## Running NN from scratch Model
### If cleaning data was runned ensure to come back to Project's main folder
```Bash
cd ..
```
### Running the model
Important when running the model there is a prompt about training the model, and seeing epochs. 
It "n" is written the model runs from saved numpy array in nn_scratch.npz
```Bash
python main.py
```
## Running the pytorch model
```Bash
python alternative.py
```
#### Data for the project derived from kaggle
Ahmed Shahriar Sakib. (2024). USA Real Estate Dataset [Data set]. Kaggle. https://doi.org/10.34740/KAGGLE/DS/3202774
