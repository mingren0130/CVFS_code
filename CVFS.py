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
import math
import scipy as sp
import sys, getopt,os

inputfile = ''
ss = ''
ex = ''
se = ''
jobs= ''
try:
	opts, args = getopt.getopt(sys.argv[1:],"hi:e:s:o:r:") 
except getopt.GetoptError:
	print('Error')                             
                                                                 
for opt, arg in opts:
	if(opt == '-h'):
		print('\n  -i <inputfilename> please use .csv file\n  -h help \n -s \n  -e \n -o')
		sys.exit(0)
	elif opt == '-i':
		inputfile = arg
	elif opt == '-s':
		ss = arg
		ss =int(ss)
	elif opt == '-e':
		ex = arg  
		ex =int(ex)
	elif opt == '-o':
		jobs = arg  
		jobs =int(jobs)
	elif opt == '-r':
		se = arg  
		if (se=='c'):
			select='classification'
if (ex==""):
	ex=ss
elif (ss==""):
	ss=ex
if (se==""):
	se='c'
if (jobs==""):
	jobs=4
if ((ss=="") and (ex=="")):
	ss=3
	ex=3

print("Number of select=",ss)
print("Number of executions=",ex)
print("jobs=",jobs)
if os.path.isfile(inputfile):
	print("file exist")
else:
	print("file does not exist")
	sys.exit(0)
if(isinstance(ss, int)==False):
	print("Incorrect number is not int")
	sys.exit(0)
if(isinstance(ex, int)==False):
	print("Incorrect number is not int")
	sys.exit(0)
if(isinstance(jobs, int)==False):
	print("jobs number is not int")
	sys.exit(0)
if (ex<ss):
	print("select cannot exceed executions")
	sys.exit(0)

print("Loading file")
df = pd.read_csv(inputfile,dtype={'genome_id':str})
print("Loading file Ok")


data=df.iloc[0:,3:]
data=data[~data['resistant_phenotype'].isin(['Intermediate'])]
XX = data.iloc[0:,1:]
if(se=='c'):
	kk=df["resistant_phenotype"].unique()
	data["resistant_phenotype"] = data["resistant_phenotype"].str.replace(kk[0],"1")
	data["resistant_phenotype"] = data["resistant_phenotype"].str.replace(kk[1],"0")
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

cut3=3
cat=[]
for k in range(ex):
	kk="datagroup_"+str(k)
	cat.append(kk)    
cc=pd.DataFrame()
kk=0
for k in cat:    
	data_n =shuffle(data)
	if(se=='c'):
		data1=data_n.loc[data_n.resistant_phenotype=='1']
		data0=data_n.loc[data_n.resistant_phenotype=='0']
		data1_lens=len(data1)/cut3
		data1_lens=int(data1_lens)
		data0_lens=len(data0)/cut3
		data0_lens=int(data0_lens)
		datagroup1= pd.concat([data0.iloc[0:int(data0_lens),0:],data1.iloc[0:int(data1_lens),0:]], axis=0)
		datagroup2= pd.concat([data0.iloc[int(data0_lens):int(data0_lens)*2,0:],data1.iloc[int(data1_lens):int(data1_lens)*2,0:]], axis=0)
		datagroup3= pd.concat([data0.iloc[int(data0_lens)*2:,0:],data1.iloc[int(data1_lens)*2:,0:]], axis=0)
	if(se=='r'):
		data0_lens=len(df)/cut3
		data0_lens=int(data0_lens)
		datagroup1= df.iloc[0:int(data0_lens),0:]
		datagroup2= df.iloc[int(data0_lens):int(data0_lens)*2,0:]
		datagroup3= df.iloc[int(data0_lens)*2:,0:]	
	cut=['datagroup1_v','datagroup2_v','datagroup3_v']
	cot=[datagroup1,datagroup2,datagroup3]
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
			model = XGBClassifier(max_depth=10,n_estimators=500,n_jobs=jobs)
			model.fit(X, y)
			feature_important = model.get_booster().get_score(importance_type='gain')
		elif(se=='r'):
			xg_reg = xgb.XGBRegressor(objective ='reg:linear',max_depth = 10,n_estimators = 500,n_jobs=jobs)
			xg_reg.fit(X, y)
			feature_important = xg_reg.get_booster().get_score(importance_type='gain')
		keys = list(feature_important.keys())
		values = list(feature_important.values())
		temp1=str(cut[j])
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
		if(s>=3):
			datagroupxor.at[c.columns.values[line],'datagroup']=1
	datagroupxor=datagroupxor.T
	ky=str(cat[kk])
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
		print("columns value=",datagroupxorr.columns.values)
		print("columns=" , datagroupxorr.shape[1])
		print("SVM AVG score=%.4f" % round(scores.mean(),4))
	else:
		print("No cluster in this experiment")
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

