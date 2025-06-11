import streamlit as st
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer
from eda import DataFrame_Loader,AttributeInformation,EDA_Data,Data_Base_Modelling
def main():
	st.title("🔬 DataWizard: EDA & Machine Learning Made Easy")

	st.markdown('**Select activity from the side menu**')

	st.markdown("""
### 🔍 What this app offers:

**1. General EDA**
- View data types, column names, and missing values
- Perform aggregation and tabulation
- Analyze numerical and categorical variables
- Drop null values
- Cross tabulation
- Pearson and Spearman correlation
- **Univariate Analysis**: Histograms, distplots, countplots
- **Bivariate Analysis**: Scatter plots, bar plots, violin plots
- **Multivariate Analysis**: Heatmaps, pairplots, word clouds

**2. EDA for Linear Models**
- Generate QQ plots
- Detect outliers
- Visualize distributions with distplots
- Perform Chi-Square tests

**3. Machine Learning Model Building**
- Train-test split
- Build baseline models:
    - Logistic Regression
    - Decision Tree
    - Random Forest
    - Naive Bayes
    - XGBoost Classifier
""")

	st.info("Sample dataset for testing -" """https://www.kaggle.com/datasets/CooperUnion/cardataset""")
  
	activities = ["General EDA","EDA For Linear Models","Machine Learning Model Building"]	
	choice = st.sidebar.selectbox("Select Activities",activities)
	# num_df = None 
	if choice == 'General EDA':
		st.title("Exploratory Data Analysis")

		data = st.file_uploader("Upload a dataset (only csv type supported)", type=["csv"])
		if data is not None:
			df = load.read_csv(data)
			st.dataframe(df.head())
			st.success("Data frame loaded successfully")
			

			if st.checkbox("Show dtypes"):
				st.write(dataframe.show_dtypes(df))

			if st.checkbox("Show Columns"):
				st.write(dataframe.show_columns(df))

			if st.checkbox("Show Missing"):
				st.write(dataframe.show_missing(df))

			if st.checkbox("column information"):
				st.write(info.column_information(df))

			if st.checkbox("Aggregation Tabulation"):
				st.write(dataframe.tabulation(df))

			if st.checkbox("Num Count Summary"):
				st.write(info.num_count_summary(df))

			if st.checkbox("Statistical Summary"):
				st.write(info.statistical_summary(df))		

# 			if st.checkbox("Show Selected Columns"):
# 				selected_columns = st.multiselect("Select Columns",all_columns)
# 				new_df = df[selected_columns]
# 				st.dataframe(new_df)

                
			if st.checkbox("Show Selected Columns"):
				selected_columns = st.multiselect("Select Columns",dataframe.show_columns(df))
				new_df = df[selected_columns]
				st.dataframe(new_df)

			if st.checkbox("Numerical Variables"):
				st.session_state['num_df'] = dataframe.numerical_variables(df)
				numer_df = pd.DataFrame(st.session_state['num_df'])
				st.dataframe(numer_df)

			if st.checkbox("Categorical Variables"):
				new_df = dataframe.categorical_variables(df)
				catego_df = pd.DataFrame(new_df)
				st.dataframe(catego_df)

			if st.checkbox("DropNA"):
				if 'num_df' in st.session_state:
					imp_df = dataframe.drop_na(st.session_state['num_df'])
					st.dataframe(imp_df)
				else:
					st.warning("Please select 'Numerical Variables' first.")


			if st.checkbox("Missing after DropNA"):
				st.write(dataframe.show_missing(imp_df))
               

			all_columns_names = dataframe.show_columns(df)
			all_columns_names1 = dataframe.show_columns(df)            
			selected_columns_names = st.selectbox("Select Column 1 for Cross Tabulation",all_columns_names)
			selected_columns_names1 = st.selectbox("Select Column 2 for Cross Tabulation",all_columns_names1)
			if st.button("Generate Cross Tab"):
				st.dataframe(pd.crosstab(df[selected_columns_names],df[selected_columns_names1]))


			all_columns_names3 = dataframe.show_columns(df)
			all_columns_names4 = dataframe.show_columns(df)            
			selected_columns_name3 = st.selectbox("Select Column 1 for Pearson Correlation (Numerical Columns)",all_columns_names3)
			selected_columns_names4 = st.selectbox("Select Column 2 for Pearson Correlation (Numerical Columns)",all_columns_names4)
			if st.button("Generate Pearson Correlation"):
				df=pd.DataFrame(dataframe.show_pearsonr(imp_df[selected_columns_name3],imp_df[selected_columns_names4]),index=['Pvalue', '0'])
				st.dataframe(df)  

			spearmanr3 = dataframe.show_columns(df)
			spearmanr4 = dataframe.show_columns(df)            
			spearmanr13 = st.selectbox("Select Column 1 for Spearman Correlation (Categorical Columns)",spearmanr4)
			spearmanr14 = st.selectbox("Select Column 2 for Spearman Correlation (Categorical Columns)",spearmanr4)
			if st.button("Generate Spearman Correlation"):
				df=pd.DataFrame(dataframe.show_spearmanr(catego_df[spearmanr13],catego_df[spearmanr14]),index=['Pvalue', '0'])
				st.dataframe(df)

			st.subheader("UNIVARIATE ANALYSIS")
			
			all_columns_names = dataframe.show_columns(df)         
			selected_columns_names = st.selectbox("Select Column for Histogram ",all_columns_names)
			if st.checkbox("Show Histogram for Selected Variable"):
				st.write(dataframe.show_hist(df[selected_columns_names]))
				st.pyplot()		

			all_columns_names = dataframe.show_columns(df)         
			selected_columns_names = st.selectbox("Select Columns for Distplot ",all_columns_names)
			if st.checkbox("Show DisPlot for Selected Variable"):
				st.write(dataframe.show_displot(df[selected_columns_names]))
				st.pyplot()

			all_columns_names = dataframe.show_columns(df)         
			selected_columns_names = st.selectbox("Select Columns for CountPlot ",all_columns_names)
			if st.checkbox("Show CountPlot for Selected Variable"):
				st.write(dataframe.show_countplot(df[selected_columns_names]))
				st.pyplot()

			st.subheader("BIVARIATE ANALYSIS")

			Scatter1 = dataframe.show_columns(df)
			Scatter2 = dataframe.show_columns(df)            
			Scatter11 = st.selectbox("Select Column 1 for Scatter Plot (Numerical Columns)",Scatter1)
			Scatter22 = st.selectbox("Select Column 2 for Scatter Plot (Numerical Columns)",Scatter2)
			if st.button("Generate Plotly Scatter Plot"):
				st.pyplot(dataframe.plotly_scatter(df,df[Scatter11],df[Scatter22]))
                
			bar1 = dataframe.show_columns(df)
			bar2 = dataframe.show_columns(df)            
			bar11 = st.selectbox("Select Column 1 for Bar Plot ",bar1)
			bar22 = st.selectbox("Select Column 2 for Bar Plot ",bar2)
			if st.button("Generate Plotly Histogram Plot"):
				st.pyplot(dataframe.plotly_histogram(df,df[bar11],df[bar22]))                

			violin1 = dataframe.show_columns(df)
			violin2 = dataframe.show_columns(df)            
			violin11 = st.selectbox("Select Column 1 for Violin Plot",violin1)
			violin22 = st.selectbox("Select Column 2 for Violin Plot",violin2)
			if st.button("Generate Plotly Violin Plot"):
				st.pyplot(dataframe.plotly_violin(df,df[violin11],df[violin22]))  

			st.subheader("MULTIVARIATE ANALYSIS")

			if st.checkbox("Show Histogram"):
				st.write(dataframe.show_hist(df))
				st.pyplot()

			if st.checkbox("Show HeatMap"):
				st.write(dataframe.show_heatmap(df))
				st.pyplot()

			if st.checkbox("Show PairPlot"):
				st.write(dataframe.show_pairplot(df))
				st.pyplot()

			if st.button("Generate Word Cloud"):
				st.write(dataframe.show_wordcloud(df))
				st.pyplot()

	elif choice == 'EDA For Linear Models':
		st.title("EDA For Linear Models")
		data = st.file_uploader("Upload a dataset (only csv type supported)", type=["csv"])
		if data is not None:
			df = load.read_csv(data)
			st.dataframe(df.head())
			st.success("Data frame loaded successfully")


			all_columns_names = dataframe.show_columns(df)         
			selected_columns_names = st.selectbox("Select Columns for qqplot ",all_columns_names)
			if st.checkbox("Show qqplot for Selected Variable"):
				st.write(dataframe.show_qqplot(df[selected_columns_names]))
				st.pyplot()

			all_columns_names = dataframe.show_columns(df)         
			selected_columns_names = st.selectbox("Select Columns for Outliers ",all_columns_names)
			if st.checkbox("Show Outliers for Selected Variable"):
				st.write(dataframe.detect_outliers(df[selected_columns_names]))

			# all_columns_names = show_columns(df)         
			# selected_columns_names = st.selectbox("Select target ",all_columns_names)
			# if st.checkbox("Anderson Normality Test"):
			# 	st.write(Anderson_test(df[selected_columns_names]))	

			if st.checkbox("Show Distplot for Selected Columns"):
				selected_columns_names = st.selectbox("Select Columns for Distplot ",all_columns_names)
				st.dataframe(dataframe.show_displot(df[selected_columns_names]))
				st.pyplot()

			con1 = dataframe.show_columns(df)
			con2 = dataframe.show_columns(df)            
			conn1 = st.selectbox("Select 1st Columns for chi square test",con1)
			conn2 = st.selectbox("Select 2st Columns for chi square test",con2)
			if st.button("Generate chi square test"):
				st.write(dataframe.check_cat_relation(df[conn1],df[conn2],0.5))
			

	elif choice == 'Machine Learning Model Building':
		st.title("Machine Learning Model Building for Classification Problem")
		data = st.file_uploader("Upload a dataset (only csv type supported)", type=["csv"])
		if data is not None:
			df = load.read_csv(data)
			st.dataframe(df.head())
			st.success("Data frame loaded successfully")

			if st.checkbox("Select your Variables  (Target Variable should be at last)"):
				selected_columns_ = st.multiselect("Select Columns for separation ",dataframe.show_columns(df))
				sep_df = df[selected_columns_]
				st.dataframe(sep_df)

			if st.checkbox("Show Indpendent Data"):
				x = sep_df.iloc[:,:-1]
				st.dataframe(x)

			if st.checkbox("Show Dependent Data"):
				y = sep_df.iloc[:,-1]
				st.dataframe(y)

			if st.checkbox("Dummy Variable"):
				df = dataframe.get_dummies(df)  # uses improved method
				st.write(f"Shape after encoding: {df.shape}")
				x = df.iloc[:, :-1]
				y = df.iloc[:, -1]
				st.dataframe(df)
    
			if st.checkbox("Imputer (imputation transformer) "):
				df = model.imputer(df)
				x = df.iloc[:, :-1]
				y = df.iloc[:, -1]
				st.dataframe(df)

			if st.checkbox("Compute Principle Component Analysis"):
				y = df.iloc[:, -1]  # Save target column before PCA
				df = dataframe.apply_pca(df.iloc[:, :-1])  # Apply PCA only on features
				x = df  # PCA-transformed features
				st.dataframe(df)


			if st.checkbox("DropNA"):
				num_df = dataframe.numerical_variables(df)
				imp_df = dataframe.drop_na(num_df)
				st.dataframe(imp_df)


			st.subheader("TRAIN TEST SPLIT")


			if st.checkbox("Select X Train"):
				from sklearn.model_selection import train_test_split
				x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
				st.dataframe(x_train)

			if st.checkbox("Select x_test"):
				from sklearn.model_selection import train_test_split
				x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
				st.dataframe(x_test)

			if st.checkbox("Select y_train"):
				from sklearn.model_selection import train_test_split
				x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
				st.dataframe(y_train)

			if st.checkbox("Select y_test"):
				from sklearn.model_selection import train_test_split
				x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
				st.dataframe(y_test)

			st.subheader("MODEL BUILDING")
			st.write("Build your BaseLine Model")

			if st.checkbox("Logistic Regression "):
				x = model.logistic_regression(x_train,y_train,x_test,y_test)
				st.write(x)

			if st.checkbox("Decision Tree "):
				x = model.decision_tree(x_train,y_train,x_test,y_test)
				st.write(x)

			if st.checkbox("Random Forest "):
				x = model.random_forest(x_train,y_train,x_test,y_test)
				st.write(x)

			if st.checkbox("Naive_Bayes "):
				x = model.naive_bayes(x_train,y_train,x_test,y_test)
				st.write(x)

			if st.checkbox("XGB Classifier "):
				x = model.xgb_classifier(x_train,y_train,x_test,y_test)
				st.write(x)


	st.markdown('<style>h1{color: red;}</style>', unsafe_allow_html=True)
	  


if __name__ == '__main__':
	load = DataFrame_Loader()
	dataframe = EDA_Data()
	info = AttributeInformation()
	model = Data_Base_Modelling()
	main()