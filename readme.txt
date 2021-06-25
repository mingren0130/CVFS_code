CVFS

CVFS is a python program that employs a Cross-Validated Feature Selection (CVFS) algorithm for extracting the most related features for classification problems. Currently the program is developed to verify its capability in identifying antimicrobial resistance genes from Salmonella enterica antimicrobial resistance datasets.
	
The core idea behind the CVFS program is that the datasets are going to randomly split into distinct sub-parts, in which machine learning feature selection algorithm selected the most plausible features within each sub-part. The selected feature sets from the distinct sub-parts were then intersected to find features shared by all sub-parts. This process were then repeated several times to avoid random effect, and the intersected features selected in most of the repeated runs were then extracted as the final feature set for classifying antimicrobial resistant Salmonella enterica strains.

Prerequisite Python packages:
	numpy
	pandas
	sklearn
	xgboost (requires cmake 3.1.3 or higher as of 6/25/2021)

Program Usage

	python CVFS.py -i <inputfilename>

	Preset parameter

		-s <Number of disjoint sub-parts to be randomly divided>; default 3
		-e <Number of repeated runs>; default 5
		-t <thread number>; default 4

Datasets
12 CSV files were provided to test the efficacy of the CVFS program. Please unzip the CSV files using the following command:

$ cd CVFS_code
$ for i in *.tar.gz
$ tar zxvf ${i}
$ done


Example command:
(simple run using default command; assume running on the ampicillin.csv dataset)
$ python3 CVFS.py -i ampicillin.csv
(

Contributing authors
Ming-Ren Yang and Yu-Wei Wu
Graduate Institute of Biomedical Informatics, Taipei Medical University, Taipei, Taiwan

Version
0.1



License
MIT License

Copyright (c) 2021 Ming-Ren Yang and Yu-Wei Wu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


