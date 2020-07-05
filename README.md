Code for master thesis: Improving the Pseudo-threshold of the Flag-Bridge Based Steane Code
==========================================================================
The folder flagbridgeQEC contains the source code. 
The pdf file master_thesis_derks.pdf is the thesis.

To run the files several packages need to be installed. If pip is installed all the required packages can be installed by typing
```shell
$ pip install -r requirements.txt
```
This module can be installed by typing
```shell
$ pip install -r requirements.txt
```

Code organisation
==========================================================================
The folder flagbridgeqec contains 8 folders:
1. **circuits**: this folder contains the circuits. The sequential circuits are in seq_circuits.py and syndrome_extraction_circuits.py. The optimized circuits are in folders and written in qpic files. The file read_test_circuits.py read the qpic files.
2. **datasets**: contains the datasets used to find the upper bounds and train the hld decoder.
3. **neural_networks**: code to train and test the hld neural network decoder and code to plot results.
4. **runs**: results for the circuits from previous work
5. **runs_optimized**: results for the optimized circuits.
6. **sim**: code to run the circuits. Split up in code to run the optimized circuits and the sequential circuits. check_ft1.py can be used to test if the circuits are fault-tolerant.
7. **tests**: pytest for testing if the circuits are fault-tolerant and for testing the look-up table decoder.
8. **utils**: for convenience.
