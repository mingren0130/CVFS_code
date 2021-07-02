VERSION = 0.1

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.utils import shuffle
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import plot_importance
from numpy import sort
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
from sklearn import set_config 
from sklearn import metrics
from sklearn import preprocessing
import scipy as sp
import sys, getopt,os
import time
import math

inputfile = ''
outputfile = ''
ss = 0
ex = 5
se = ''
cut = ''
jobs = ''
perc = 0.6
try:
	opts, args = getopt.getopt(sys.argv[1:],"hvi:o:c:e:p:t:")  
except getopt.GetoptError:
	print('Usage: python3 CVFS.py\n  -i <input filename> (please use .csv files) \n  -o <output file>\n  [-c <Number of disjoint sub-parts; default 2>]\n  [-e <Number of repeated runs>; default 5] \n  [-p<Proportion of repeated runs for extracting common features>; default 0.6]\n  [-t <Thread number>; default 4]\n  [-v <Display version number>]')
	sys.exit(2)                             
                                                                 
for opt, arg in opts:
	if(opt == '-h'):
		print('Usage: python3 CVFS.py\n  -i <input filename> (please use .csv files) \n  -o <output file>\n  [-c <Number of disjoint sub-parts; default 2>]\n  [-e <Number of repeated runs>; default 5] \n  [-p<Proportion of repeated runs for extracting common features>; default 0.6]\n  [-t <Thread number>; default 4]\n  [-v <Display version number>]')
		sys.exit(0)
	elif(opt == '-v'):
		print("CVFS.py version", VERSION)
		print("Developed by Ming-Ren Yang and Yu-Wei Wu at Taipei Medical University, 2021")
		sys.exit(0)
	elif opt == '-i':
		inputfile = arg
	elif opt == '-o':
		outputfile = arg
	elif opt == '-c':
		cut = arg  
		cut =int(cut)
	#elif opt == '-s':
		#ss = arg
		#ss =int(ss)
	elif opt == '-p':
		perc = arg  
		perc =float(perc)
	elif opt == '-e':
		ex = arg  
		ex =int(ex)
	elif opt == '-t':
		jobs = arg  
		jobs =int(jobs)
	elif opt == '-r':
		se = arg  
		if (se=='c'):
			select='classification'
if (inputfile=="" or outputfile==""):
	print('Usage: python3 CVFS.py\n  -i <input filename> (please use .csv files) \n  -o <output file>\n  [-c <Number of disjoint sub-parts>; default 2]\n  [-e <Number of repeated runs>; default 5] \n  [-p<Proportion of repeated runs for extracting common features>; default 0.6]\n  [-t <Thread number>; default 4]')
	sys.exit(0)
if (perc > 1 or perc <= 0):
	print("Print specify a number between 0 and 1 for proportions of repeated runs.");
	sys.exit()
if (se==""):
	se='c'
if (jobs==""):
	jobs=4
if (cut==""):
	cut=2
ss = math.ceil(float(ex)*perc)

print("Number of disjoint sub-parts = ",cut)
print("Number of repeated runs = ",ex)
print("Proportion of features shared by repeated runs =", perc, "(Features need to appear in at least", ss, "repeated runs)")
print("Thread number = ",jobs)
if os.path.isfile(inputfile):
	print("File", inputfile, "found.")
else:
	print("File", inputfile, "not exist. Please indicate the correct filename.")
	sys.exit(0)
if(isinstance(cut, int)==False or cut <= 0):
	print("Incorrect disjoint sub-part number [", cut ,"]. Need to be >= 0.", sep="")
	sys.exit(0)
if(isinstance(ex, int)==False or ex <= 0):
	print("Incorrect repeated run number [", ex ,"]. Need to be >= 0.", sep="")
	sys.exit(0)
if(isinstance(jobs, int)==False):
	print("Incorrect thread number [", jobs ,"]. Need to be >= 0.", sep="")
	sys.exit(0)
if (ex<ss):
	print("select cannot exceed executions")
	sys.exit(0)
print("Loading file")
#df = pd.read_csv(inputfile,dtype={'genome_id':str})
df = pd.read_csv(inputfile)
print("Loading file Ok")
le = preprocessing.LabelEncoder()
data=df.iloc[0:,3:]
data=data[~data['resistant_phenotype'].isin(['Intermediate'])]
XX = data.iloc[0:,1:]
if(se=='c'):
	#kk=df["resistant_phenotype"].unique()
	#data["resistant_phenotype"] = data["resistant_phenotype"].str.replace(kk[0],"1")
	#data["resistant_phenotype"] = data["resistant_phenotype"].str.replace(kk[1],"0")
	data["resistant_phenotype"] = le.fit_transform(data["resistant_phenotype"])
	data=data.loc[:,~((data==0).all())]
elif(se=='r'):
	df['new_value']='NULL'
	for line in range(df.shape[0]):
		line2=line+1
		st=df.iloc[line:line2,3:4].values
		if(st=='>'):
			df.iloc[line:line2,-1:]=(math.log2((df.iloc[line:line2,4:5]).sum()*2))
		elif(st=='>='):
			df.iloc[line:line2,-1:]=(math.log2((df.iloc[line:line2,4:5]).sum()*2))
		elif(st=='<'):
			df.iloc[line:line2,-1:]=(math.log2((df.iloc[line:line2,4:5]).sum()/2))
		elif(st=='<='):
			df.iloc[line:line2,-1:]=(math.log2((df.iloc[line:line2,4:5]).sum()/2))
		else:
			df.iloc[line:line2,-1:]=(math.log2((df.iloc[line:line2,4:5]).sum()))
	df=df.loc[:,~((df==0).all())] 
cat=[]
cuut=[]
cot=[]
for k in range(ex):
	kk="datagroup_"+str(k)
	cat.append(kk)   
cc=pd.DataFrame()
kk=0
for k in cat:    
	data_n =shuffle(data)
	if(se=='c'):
		data1=data_n.loc[data_n.resistant_phenotype==1]
		data0=data_n.loc[data_n.resistant_phenotype==0]
		data1_lens=len(data1)/cut
		data1_lens=int(data1_lens)
		data0_lens=len(data0)/cut
		data0_lens=int(data0_lens)
		cmt=[]
		for k in range(cut):
			dk="datagroup_"+str(k)
			cmt.append(dk)
		jj=0
		for j in range(cut):
    			if((jj+1)==cut):
        			cmt[jj]=pd.concat([data0.iloc[int(data0_lens)*jj:,0:],data1.iloc[int(data1_lens)*jj:,0:]], axis=0)
    			else:
        			cmt[jj]=pd.concat([data0.iloc[int(data0_lens)*jj:int(data0_lens)*(jj+1),0:],data1.iloc[int(data1_lens)*jj:int(data1_lens)*(jj+1),0:]], axis=0)
    			cot.append(cmt[jj])  
    			jj=jj+1
		#datagroup1= pd.concat([data0.iloc[0:int(data0_lens),0:],data1.iloc[0:int(data1_lens),0:]], axis=0)
		#datagroup2= pd.concat([data0.iloc[int(data0_lens):int(data0_lens)*2,0:],data1.iloc[int(data1_lens):int(data1_lens)*2,0:]], axis=0)
		#datagroup3= pd.concat([data0.iloc[int(data0_lens)*2:,0:],data1.iloc[int(data1_lens)*2:,0:]], axis=0)
	if(se=='r'):
		data0_lens=len(df)/cut3
		data0_lens=int(data0_lens)
		datagroup1= df.iloc[0:int(data0_lens),0:]
		datagroup2= df.iloc[int(data0_lens):int(data0_lens)*2,0:]
		datagroup3= df.iloc[int(data0_lens)*2:,0:]
	for k in range(cut):
    		kkk="datagroup"+str(k)
    		cuut.append(kkk) 	
	c=pd.DataFrame()
	j=0
	for i in cot:
		if(se=='c'):
			X = i.iloc[0:,1:]
			y = i['resistant_phenotype']
		elif(se=='r'):
			X = df.iloc[0:,7:-1]
			y = df['new_value']
		if(se=='c'):
			model = XGBClassifier(max_depth=10,n_estimators=500,n_jobs=jobs,use_label_encoder=False,eval_metric="auc")
			model.fit(X, y)
			feature_important = model.get_booster().get_score(importance_type='gain')
		elif(se=='r'):
			xg_reg = xgb.XGBRegressor(objective ='reg:linear',max_depth = 10,n_estimators = 500,n_jobs=jobs)
			xg_reg.fit(X, y)
			feature_important = xg_reg.get_booster().get_score(importance_type='gain')
		keys = list(feature_important.keys())
		values = list(feature_important.values())
		temp1=str(cuut[j])
		temp2 = pd.DataFrame( index=keys,data=values,columns=["score"]).sort_values(by = "score", ascending=False)
		temp2=temp2.T
		for iu in range(0,temp2.shape[1],1):
			c.at[temp2.columns.values[iu],temp1]=1
		j=j+1
	c=c.T
	c=c.fillna(0)
	datagroupxor=pd.DataFrame()
	for line in range(c.shape[1]):
		line2=line+1
		s=c.iloc[0:,line:line2].sum()
		s=int(s)
		if(s>=cut):
			datagroupxor.at[c.columns.values[line],'datagroup']=1
	datagroupxor=datagroupxor.T
	ky=str(cat[kk])
	#for iu in range(0,datagroupxor.shape[1],1):
	for iu in range(0,datagroupxor.shape[1],1):
		cc.at[datagroupxor.columns.values[iu],ky]=1
	kk=kk+1
print("xgb is ok")
datagroupxorr=pd.DataFrame()
cc=cc.T
cc=cc.fillna(0)
for line in range(cc.shape[1]):
    line3=line+1
    sr=cc.iloc[0:,line:line3].sum()
    sr=int(sr)
    if(sr>=ss):
        datagroupxorr.at[cc.columns.values[line],'yyy']=1 
datagroupxorr=datagroupxorr.T
if(se=='c'):
	data['resistant_phenotype'] = data['resistant_phenotype'].astype(int)
	y = data['resistant_phenotype']
	if((datagroupxorr.shape[1])!=0):
		datagroupjoin=pd.DataFrame() 
		for line in range(XX.shape[1]):
			for line2 in range(datagroupxorr.shape[1]):
				if(XX.columns.values[line]==datagroupxorr.columns.values[line2]):
					datagroupjoin= pd.concat([datagroupjoin, data[XX.columns[line]]], axis=1)
		model = SVC(kernel='linear')
		scores = cross_val_score(model, datagroupjoin, y, cv=10, scoring='roc_auc')
		outF = open(outputfile, "w")
		outF.write("Extracted ")
		outF.write(str(datagroupxorr.shape[1]))
		outF.write(" features\n")
		outF.write("Classification accuracy of the dataset using extracted features is ")
		outF.write(str(round(scores.mean(),4)))
		outF.write("\n")
		for line in datagroupxorr.columns.values:
			outF.write(line)
			outF.write("\n")
		outF.close()
		#print("columns=" , datagroupxorr.shape[1])
		#print("SVM AVG score=%.4f" % round(scores.mean(),4))
		print("Extracted", datagroupxorr.shape[1], "features")
		print("Classification accuracy of the dataset using extracted features is", round(scores.mean(),4));
	else:
		print("Cannot find shared features in this run. Please adjust the parameters.");
elif(se=='r'):
	yy = df['new_value']
	if((datagroupxorr.shape[1])!=0):
		datagroupjoin=pd.DataFrame() 
		for line in range(XX.shape[1]):
			for line2 in range(datagroupxorr.shape[1]):
				if(XX.columns.values[line]==datagroupxorr.columns.values[line2]):
					datagroupjoin= pd.concat([datagroupjoin, df[XX.columns[line]]], axis=1)
		rfr = RandomForestRegressor()
		scores = cross_val_predict(rfr, datagroupjoin, yy,cv=10)
		rmse=np.sqrt(metrics.mean_squared_error(yy, scores))
		print("RMSE: %f" % (rmse))
		print('Pearsons p-value:',sp.stats.pearsonr(np.squeeze(yy), scores))
		print("MAE: %f" % mean_absolute_error(yy,scores))
		print("r2_score: %f" %r2_score(yy,scores))  
	else:
		print("No cluster in this experiment")

