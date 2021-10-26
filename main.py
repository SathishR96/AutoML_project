import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import sklearn
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler,MaxAbsScaler,Normalizer
from sklearn import preprocessing
#from category_encoders.target_encoder import TargetEncoder
from pandas_profiling import ProfileReport 
from streamlit_pandas_profiling import st_profile_report
import sweetviz as sv 
import dtale
from dtale.views import startup




import os
path=os.getcwd()
temp='//temp.csv'
path=path+temp

page = st.selectbox("Choose your page", ["EDA", "Data preprocessing", "Auto ML"]) 

if page=="EDA":
	st.sidebar.title(
	"Exploratory Data Analysis APP" 
	)

	file=st.sidebar.checkbox("File Import")

	if file:

		#st.sidebar.checkbox("Choose The Format of the File",value = True)

		try:

			forma=["Xlsx","Csv","Oracle","Email"]
			upload=st.sidebar.selectbox("Choose The Format of the File ",forma)



			if upload=="Xlsx":
				uploaded_file = st.sidebar.file_uploader("Choose a file", type="xlsx")
				if uploaded_file is not None:
					df=pd.read_excel(uploaded_file)
					df=df.drop(df.columns[0], axis=1)
					if st.sidebar.button('Upload File'):
						st.dataframe(df)
						df.to_csv(path)

			elif upload=="Csv":
				uploaded_file = st.sidebar.file_uploader("Choose a file", type="csv")
				if uploaded_file is not None:
					df=pd.read_csv(uploaded_file)
					if st.sidebar.button('Upload File'):
					    #df=upload_xlsx(uploaded_file)
						st.dataframe(df)
						df.to_csv(path,index=False)

			elif upload=="Oracle":
				pass

			elif upload=="Email":
				pass

		except:
			pass


	exp=st.sidebar.checkbox("Export The File")

	if exp:

		try:

			down=["Download as CSV","Export to Oracle","Push As Email Attachment","Download as XLSX"]
			downl=st.sidebar.selectbox("choose the format of the file ",down)

			if downl=='Download as CSV':
				if st.sidebar.button('Process Download as CSV'):
					with open('temp.csv', 'r+') as f:
						st.sidebar.download_button(
						        label="Download Csv",
						        data=f,
						        mime='text/csv',
						        file_name="File.csv",
						        )

			elif downl=='Download as XLSX':
				if st.sidebar.button('Process Download as XLSX'):
					with open('temp.csv', 'r+') as f:
						st.sidebar.download_button(
						        label="Download XLSX",
						        data=f,
						        mime='text/xlsx',
						        file_name="File.xlsx",
						        )

			elif downl=='Export to Oracle':
				if st.sidebar.button('Process Export to Oracle'):
					pass

			elif downl=='Push As Email Attachment':
				if st.sidebar.button('Process Push As Email Attachment'):
					pass

		except:
			pass

	plot=st.sidebar.checkbox("Plotting")
	
	if plot:

		try:

			style=["package","Box Plot","CountPlot","Dist Plot","Pair Plot","Bar Plot","RelPlot","Boxen Plot"]
			plt=st.sidebar.selectbox("Choose The Plot type ",style)



			if plt=='CountPlot':
				df=pd.read_csv("temp.csv")
				df=df.drop(df.columns[0], axis=1)
				columns=df.columns
				sc=st.sidebar.selectbox("Choose The columns",columns)
				if sc:
					if st.sidebar.button('Process CountPlot'):
						st.write(sns.countplot(df[sc]))
						st.set_option('deprecation.showPyplotGlobalUse', False)
						st.pyplot()
			
			elif plt=='Dist Plot':
				df=pd.read_csv("temp.csv")
				df=df.drop(df.columns[0], axis=1)
				columns=df.columns
				sc=st.sidebar.selectbox("Choose The columns",columns)
				if sc:
					if st.sidebar.button('Process Dist Plot'):
						st.write(sns.displot(df[sc]))
						st.set_option('deprecation.showPyplotGlobalUse', False)
						st.pyplot()

			elif plt=='Pair Plot':
				df=pd.read_csv("temp.csv")
				df=df.drop(df.columns[0], axis=1)
				if st.sidebar.button('Process Pair Plot'):
					st.write(sns.pairplot(df))
					st.set_option('deprecation.showPyplotGlobalUse', False)
					st.pyplot()

			elif plt=='Bar Plot':
				df=pd.read_csv("temp.csv")
				df=df.drop(df.columns[0], axis=1)
				columns=df.columns
				sc1=st.sidebar.selectbox("Choose The 1st columns",columns)
				sc2=st.sidebar.selectbox("Choose The 2nd columns",columns)
				if sc1:
					if sc2:
						if st.sidebar.button('Process Bar plot'):
							st.write(sns.barplot(x=sc1,y=sc2,data=df))
							st.set_option('deprecation.showPyplotGlobalUse', False)
							st.pyplot()

			elif plt=='RelPlot':
				df=pd.read_csv("temp.csv")
				df=df.drop(df.columns[0], axis=1)
				columns=df.columns
				sc1=st.sidebar.selectbox("Choose The 1st columns",columns)
				sc2=st.sidebar.selectbox("Choose The 2nd columns",columns)
				if sc1:
					if sc2:
						if st.sidebar.button('Process RelPlot'):
							st.write(sns.relplot(x=sc1,y=sc2,data=df))
							st.set_option('deprecation.showPyplotGlobalUse', False)
							st.pyplot()

			elif plt=='Boxen Plot':
				df=pd.read_csv("temp.csv")
				df=df.drop(df.columns[0], axis=1)
				columns=df.columns
				sc1=st.sidebar.selectbox("Choose The 1st columns",columns)
				sc2=st.sidebar.selectbox("Choose The 2nd columns",columns)
				if sc1:
					if sc2:
						if st.sidebar.button('Process Boxen Plot'):
							st.write(sns.boxenplot(x=sc1,y=sc2,data=df))
							st.set_option('deprecation.showPyplotGlobalUse', False)
							st.pyplot()

			elif plt=='Box Plot':
				df=pd.read_csv("temp.csv")
				df=df.drop(df.columns[0], axis=1)
				columns=df.columns
				sc1=st.sidebar.selectbox("Choose The 1st columns",columns)
				sc2=st.sidebar.selectbox("Choose The 2nd columns",columns)
				if sc1:
					if sc2:
						if st.sidebar.button('Process Box Plot'):
							st.write(sns.boxplot(x=sc1,y=sc2,data=df))
							st.set_option('deprecation.showPyplotGlobalUse', False)
							st.pyplot()
		except:
			pass

	package=st.sidebar.checkbox("EDA Libraries")
	
	if package:

		try:

			package_list=["Pandas Profiling","SweetViz","Dtale"]
			plt=st.sidebar.selectbox("Choose The Plot type ",package_list)
			
			if plt=='Pandas Profiling':
				df=pd.read_csv("temp.csv")
				df=df.drop(df.columns[0], axis=1)
				columns=df.columns
				if st.sidebar.button('Process Pandas Profiling'):
					profile = ProfileReport(df)
					st_profile_report(profile)

			elif plt=='SweetViz':
				df=pd.read_csv("temp.csv")
				df=df.drop(df.columns[0], axis=1)
				columns=df.columns
				if st.sidebar.button('Process SweetViz'):
					report = sv.analyze(df)
					report.show_html()
					st_display_sweetviz("SWEETVIZ_REPORT.html")

			elif plt=='Dtale':
				df=pd.read_csv("temp.csv")
				df=df.drop(df.columns[0], axis=1)
				columns=df.columns
				if st.sidebar.button('Process Dtale'):
					startup(data_id="1", data=df)
					from dtale.app import get_instance
					# remember we set the data_id of our previous dataframe was "1"
					df = get_instance("1").data
					#<iframe src="/dtale/main/1" />
					#<a href="/dtale/main/1" target="_blank">Dataframe 1</a>
		except:
			pass



# Data Preprocessing Codes
elif page=="Data preprocessing":
	st.sidebar.title(
	"DATA PREPROCESSOR APP" 
	)	

	file=st.sidebar.checkbox("File Import")

	if file:

		#st.sidebar.checkbox("Choose The Format of the File",value = True)

		try:

			forma=["Xlsx","Csv","Oracle","Email"]
			upload=st.sidebar.selectbox("Choose The Format of the File ",forma)



			if upload=="Xlsx":
				uploaded_file = st.sidebar.file_uploader("Choose a file", type="xlsx")
				if uploaded_file is not None:
					df=pd.read_excel(uploaded_file)
					df=df.drop(df.columns[0], axis=1)
					if st.sidebar.button('Upload File'):
						st.dataframe(df)
						df.to_csv(path)

			elif upload=="Csv":
				uploaded_file = st.sidebar.file_uploader("Choose a file", type="csv")
				if uploaded_file is not None:
					df=pd.read_csv(uploaded_file)
					if st.sidebar.button('Upload File'):
					    #df=upload_xlsx(uploaded_file)
						st.dataframe(df)
						df.to_csv(path,index=False)

			elif upload=="Oracle":
				pass

			elif upload=="Email":
				pass

		except:
			pass




	mt=st.sidebar.checkbox("Missing Value Treatment")

	if mt:

		try:

			mvt=["Mean Imputation","Median Imputation ","Mode Imputation"]
			treatment=st.sidebar.selectbox("Choose an Option to Perform ", mvt)

			if treatment=='Mean Imputation':
				if st.sidebar.button('Process Mean'):
					#st.write(path)
					df=pd.read_csv("temp.csv")
					df=(df.fillna(df.mean()))
					df.fillna(df.select_dtypes(include='object').mode())
					df=df.drop(df.columns[0], axis=1)
					st.dataframe(df)
					df.to_csv(path)

			elif treatment=='Median Imputation':
				if st.sidebar.button('Process Median'):
					df=pd.read_csv("temp.csv")
					df=(df.fillna(df.median()))
					df=df.fillna(df.select_dtypes(include='object').mode())
					#df=df.drop(df.columns[0], axis=1)
					st.dataframe(df)
					df.to_csv(path,index=False, sep=',')

			elif treatment=='Mode Imputation':
				if st.sidebar.button('Process Mode'):
					df=pd.read_csv("temp.csv")
					df=(df.fillna(df.mode()))
					df=df.fillna(df.select_dtypes(include='object').mode())
					st.dataframe(df)
					df.to_csv(path)
		except:
			pass

	FE=st.sidebar.checkbox("Feature Encoding")

	if FE:

		try:

			encode=["One Hot Encoding","Dummy Encoding","Label Encoding","Hash Encoding","Frequency Encoding"]
			encoding=st.sidebar.selectbox("Choose an Option to Perform ", encode)

			if encoding=="One Hot Encoding":
				if st.sidebar.button('Process One Hot Encoding'):
					df=pd.read_csv("temp.csv")
					df=pd.get_dummies(df)
					df=df.drop(df.columns[0], axis=1)
					st.dataframe(df)
					df.to_csv(path)
					

			elif encoding=="Dummy Encoding":
				if st.sidebar.button('Process Dummy Encoding'):
					df=pd.read_csv("temp.csv")
					df=pd.get_dummies(df,drop_first=True)
					df=df.drop(df.columns[0], axis=1)
					st.dataframe(df)
					df.to_csv(path)

			elif encoding=="Label Encoding":
				if st.sidebar.button('Process Label Encoding'):
					df=pd.read_csv("temp.csv")
					df2=df.select_dtypes(include='object')
					for i in df2.columns:
						df2[i] = df2[i].astype('category')
						df2[i] = df2[i].cat.codes
					df1=df.drop(df.select_dtypes(include="object"),axis=1)
					df=pd.concat([df2, df1], axis=1)
					df=df.drop(df.columns[0], axis=1)
					st.dataframe(df)
					df.to_csv(path)

			elif encoding=="Hash Encoding":
				if st.sidebar.button('Process Hash Encoding'):
					pass

			elif encoding=="Frequency Encoding":
				if st.sidebar.button('Process Frequency Encoding'):
					df=pd.read_csv("temp.csv")
					"""df2=df.select_dtypes(include='object')
					for i in df2.columns:
						df2[i] = df2[i].astype('category')
						mean1_i=df2[i].groupby(df2[i])['y'].mean()
						mean0_i=1-mean_i
						pr=mean1_i/mean0_i
						pr=pr.to_dict()
						df2[i]=df2[i].map(pr)
					df1=df.drop(df.select_dtypes(include="object"),axis=1)
					df=pd.concat([df2, df1], axis=1)
					df=df.drop(df.columns[0], axis=1)"""
					df['target']=df[input("enter: ")]
					df['target']=pd.DataFrame(df['target'])
					df=pd.concat([df,df['target']],axis=1)
					te_df=df.copy()

					for col in te_df.select_dtypes(include='O').columns:
					    te=TargetEncoder()
					    te_df[col]=te.fit_transform(te_df[col],te_df.target)
					st.dataframe(te_df)
					df.to_csv(path)
		except:
			pass

	OT=st.sidebar.checkbox("Outlier Treatment")

	if OT:

		try:
			out=["Inter Quantile Range Method","Extreme Value Analysis"]
			outlier=st.sidebar.selectbox("Choose an Option to Perform ", out)

			if outlier=='Inter Quantile Range Method':
				if st.sidebar.button('Process IQR'):
					df=pd.read_csv("temp.csv")
					df=df.drop(df.columns[0], axis=1)
					df = df.select_dtypes(include=np.number)
					Q1 = df.quantile(0.25)
					Q3 = df.quantile(0.75)
					IQR = Q3 - Q1
					df= df[~((df < (Q1 - 1.5*IQR))| (df > (Q3 + 1.5*IQR))).any(axis=1)]
					st.dataframe(df)
					df.to_csv(path,index=False)

			elif outlier=='Extreme Value Analysis':
				if st.sidebar.button('Process Z Score Method'):
					df=pd.read_csv("temp.csv")
					df=df.drop(df.columns[0], axis=1)
					df = df.select_dtypes(include=np.number)
					std=np.std(df)
					mean=np.mean(df)
					df=df[((df-mean)/std).any(axis=1)]
					st.dataframe(df)
					df.to_csv(path,index=False)
		except:
			pass

	FS=st.sidebar.checkbox("Feature Scaling")

	if FS:

		try:

			std=["Standard Scaler","MinMax Scaler","Robust Scaler","MaxAbs Scaler"]
			scaling=st.sidebar.selectbox("Choose an Option to Perform ", std)

			if scaling=='Standard Scaler':
				if st.sidebar.button('Process Standard Scaler'):
					df=pd.read_csv("temp.csv")
					df=df.drop(df.columns[0], axis=1)
					X = df.select_dtypes(include=np.number)
					mean_X = np.mean(X)
					std_X = np.std(X)
					df = (X - np.mean(X))/np.std(X)
					st.dataframe(df)
					df.to_csv(path,index=False)

			elif scaling=='MinMax Scaler':
				if st.sidebar.button('Process MinMax Scaler'):
					df=pd.read_csv("temp.csv")
					df=df.drop(df.columns[0], axis=1)
					df = df.select_dtypes(include=np.number)
					xmin=np.min(df)
					xmax=np.max(df)
					df = (df -xmin) / (xmax -xmin)
					st.dataframe(df)
					df.to_csv(path,index=False)

			elif scaling=='Robust Scaler':
				if st.sidebar.button('Process Robust Scaler'):
					df=pd.read_csv("temp.csv")
					df=df.drop(df.columns[0], axis=1)
					df = df.select_dtypes(include=np.number)
					q3=df.quantile(0.75)-df.quantile(0.25)
					df =(df - np.median(df))/q3
					st.dataframe(df)
					df.to_csv(path,index=False)


			elif scaling=='MaxAbs Scaler':
				if st.sidebar.button('Process MaxAbs Scaler'):
					df=pd.read_csv("temp.csv")
					df=df.drop(df.columns[0], axis=1)
					df = df.select_dtypes(include=np.number) 
					df = df /np.max(abs(df))
					st.dataframe(df)
					df.to_csv(path,index=False)
		except:
			pass



	exp=st.sidebar.checkbox("Export The File")

	if exp:

		try:

			down=["Download as CSV","Export to Oracle","Push As Email Attachment","Download as XLSX"]
			downl=st.sidebar.selectbox("choose the format of the file ",down)

			if downl=='Download as CSV':
				if st.sidebar.button('Process Download as CSV'):
					with open('temp.csv', 'r+') as f:
						st.sidebar.download_button(
						        label="Download Csv",
						        data=f,
						        mime='text/csv',
						        file_name="File.csv",
						        )

			elif downl=='Download as XLSX':
				if st.sidebar.button('Process Download as XLSX'):
					with open('temp.csv', 'r+') as f:
						st.sidebar.download_button(
						        label="Download XLSX",
						        data=f,
						        mime='text/xlsx',
						        file_name="File.xlsx",
						        )

			elif downl=='Export to Oracle':
				if st.sidebar.button('Process Export to Oracle'):
					pass

			elif downl=='Push As Email Attachment':
				if st.sidebar.button('Process Push As Email Attachment'):
					pass

		except:
			pass

elif page=="Auto ML":
	

	st.sidebar.title(
	"AUTO ML" 
	)

	file=st.sidebar.checkbox("File Import")

	if file:

		#st.sidebar.checkbox("Choose The Format of the File",value = True)

		try:

			forma=["Xlsx","Csv","Oracle","Email"]
			upload=st.sidebar.selectbox("Choose The Format of the File ",forma)



			if upload=="Xlsx":
				uploaded_file = st.sidebar.file_uploader("Choose a file", type="xlsx")
				if uploaded_file is not None:
					df=pd.read_excel(uploaded_file)
					df=df.drop(df.columns[0], axis=1)
					if st.sidebar.button('Upload File'):
						st.dataframe(df)
						df.to_csv(path)

			elif upload=="Csv":
				uploaded_file = st.sidebar.file_uploader("Choose a file", type="csv")
				if uploaded_file is not None:
					df=pd.read_csv(uploaded_file)
					if st.sidebar.button('Upload File'):
					    #df=upload_xlsx(uploaded_file)
						st.dataframe(df)
						df.to_csv(path,index=False)

			elif upload=="Oracle":
				pass

			elif upload=="Email":
				pass

		except:
			pass

	ml=st.sidebar.checkbox("Auto ML")

	if ml:

		try:
			algorithms=["Navie Bayes","Decision Tree","Logistic Classification","Support Vector Machine","Random Forest"]
			upload=st.sidebar.selectbox("Choose The Algorithms to Perform ",algorithms)
			if upload=="Navie Bayes":
				df=pd.read_excel(uploaded_file)
				df=df.drop(df.columns[0], axis=1)
				columns=df.columns
				target=st.sidebar.selectbox("Choose The columns as Target",columns)
				if target:
					if st.sidebar.button('Process Navie Bayes'):
						y=df[target]
						x=df
						st.write("Target")
						st.dataframe(y)
						st.dataframe(x)
						model=GaussianNB()
						model.fit(x,y)
						st.write("Your Model is Trained ")
						st.write("Proceed With Testing / Validating the Model")
						for i in x.columns:
							lis=x[i].unique().tolist()
							bucket=st.selectbox("choose the options",lis)
							for i in bucket:
								st.dataframe(i)
			if upload=="Decision Tree":
				df=pd.read_excel(uploaded_file)
				df=df.drop(df.columns[0], axis=1)
				columns=df.columns
				target=st.sidebar.selectbox("Choose The columns as Target",columns)
				if target:
					if st.sidebar.button('Process Navie Bayes'):
						y=df[target]
						x=df
						st.write("Target")
						st.dataframe(y)
						st.dataframe(x)
						
		
		except:
			pass
