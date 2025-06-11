import pandas as pd
import numpy as np
import seaborn as sns
from typing import Literal
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import scipy.stats as stats
from sklearn import preprocessing
from sklearn.impute import IterativeImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import plotly.express as px
from wordcloud import WordCloud
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from scipy.stats import pearsonr, spearmanr, anderson
from scipy import stats
from statsmodels.graphics.gofplots import qqplot
import statsmodels.api as sm
from sklearn.impute import SimpleImputer
from scipy.stats import chi2_contingency

class DataFrame_Loader:


    def __init__(self):

        print("Loading DataFrame")

    def read_csv(self,data):
        self.df = pd.read_csv(data)
        return self.df
    
class EDA_Data:

    def __init__(self):
        print("EDA_Dataframe_Analysis object created")

    def show_dtypes(self, df):
        return df.dtypes

    def show_columns(self, df):
        return df.columns

    def show_missing(self, df):
        return df.isna().sum()

    def show_hist(self, df):
        return df.hist()

    def tabulation(self, df):
        table = pd.DataFrame(df.dtypes, columns=['Dtype'])
        table = table.reset_index().rename(columns={'index': 'Name'})
        table['No of Missing'] = df.isnull().sum().values
        table['No of Uniques'] = df.nunique().values
        table['% Missing'] = (df.isnull().sum().values / df.shape[0]) * 100

        for i in range(min(3, len(df))):
            table[f'Observation {i+1}'] = df.iloc[i].values

        for name in table['Name']:
            entropy_val = stats.entropy(df[name].value_counts(normalize=True), base=2)
            table.loc[table['Name'] == name, 'Entropy'] = round(float(entropy_val), 2)

        return table


    def numerical_variables(self, df):
        return df.select_dtypes(exclude="object")

    def categorical_variables(self, df):
        return df.select_dtypes(include="object")

    def drop_na(self, df):
        return df.dropna()

    def show_pearsonr(self, x, y):
        return pearsonr(x, y)

    def show_spearmanr(self, x, y):
        return spearmanr(x, y)

    def plotly_scatter(self, df, x, y):
        fig = px.scatter(df, x=x, y=y)
        fig.update_traces(marker=dict(size=10, line=dict(width=2, color='DarkSlateGrey')))
        fig.show()

    def show_displot(self, x):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.histplot(x, kde=True)

        plt.subplot(1, 2, 2)
        x.plot.box(figsize=(8, 4))

        plt.tight_layout()
        plt.show()

    def show_countplot(self, x):
        fig, ax = plt.subplots(figsize=(16, 6))
        return sns.countplot(x=x, ax=ax)

    def plotly_histogram(self, df, x, y=None):
        fig = px.histogram(df, x=x, y=y)
        fig.show()

    def plotly_violin(self, df, x, y):
        fig = px.violin(df, x=x, y=y, box=True, points="all")
        fig.show()

    def show_pairplot(self, df):
        return sns.pairplot(df)

    def show_heatmap(self, df):
        numeric_df = df.select_dtypes(include='number')
        numeric_df = numeric_df.dropna()
        plt.figure(figsize=(15, 15))
        return sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')

    def show_wordcloud(self, series):
        text = " ".join(series.astype(str))
        wordcloud = WordCloud(width=1000, height=500).generate(text)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()
        return wordcloud

    def label_encode(self, x):
        le = LabelEncoder()
        return le.fit_transform(x)

    def concat(self, *dfs, axis: Literal[0, 1] = 0):
        return pd.concat(list(dfs), axis=axis)


    def get_dummies(self, df, max_unique=100):
        cat_cols = [col for col in df.columns if df[col].dtype == 'object' or df[col].dtype.name == 'category']
        
        limited_cols = [col for col in cat_cols if df[col].nunique() <= max_unique]
        
        df_to_encode = df[limited_cols]
        dummies = pd.get_dummies(df_to_encode, drop_first=True)

        df = df.drop(columns=limited_cols)
        df = pd.concat([df, dummies], axis=1)
        
        return df


    def show_qqplot(self, x):
        return qqplot(x, line='45')

    def anderson_test(self, x):
        return anderson(x)

    def apply_pca(self, df, n_components=8):

        df_processed = pd.get_dummies(df, drop_first=True)
        df_processed = df_processed.dropna()  
        pca = PCA(n_components=n_components)
        components = pca.fit_transform(df_processed)
        return pd.DataFrame(components, columns=[f'PC{i+1}' for i in range(components.shape[1])])


    def detect_outliers(self, x):
        q1 = x.quantile(0.25)
        q3 = x.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return x[(x < lower) | (x > upper)]

    from scipy.stats import chi2_contingency


    def check_cat_relation(self, x, y, confidence_level=0.95):
        cross_tab = pd.crosstab(x, y)
        result = chi2_contingency(cross_tab)
        stat, p, dof, expected = result  # Ensures p is a float

        alpha = 1 - confidence_level
        print(f"Chi-square: {stat:.4f}, p-value: {p:.4f}")
        
        if isinstance(p, float) and p > alpha:
            print(">> Accepting Null Hypothesis: No significant relationship.")
        else:
            print(">> Rejecting Null Hypothesis: Significant relationship.")
        
        return p, alpha



class AttributeInformation:

    def __init__(self):
        print("AttributeInformation object created")

    def column_information(self, data: pd.DataFrame) -> pd.DataFrame:
        """Provides basic attribute info of the dataset"""
        data_info = pd.DataFrame(
            columns=[
                'No of observations',
                'No of Variables',
                'No of Numerical Variables',
                'No of Factor Variables',
                'No of Categorical Variables',
                'No of Logical Variables',
                'No of Date Variables',
                'No of zero variance variables'
            ]
        )

        data_info.loc[0, 'No of observations'] = data.shape[0]
        data_info.loc[0, 'No of Variables'] = data.shape[1]
        data_info.loc[0, 'No of Numerical Variables'] = data.select_dtypes(include=[np.number]).shape[1]
        data_info.loc[0, 'No of Factor Variables'] = data.select_dtypes(include='category').shape[1]
        data_info.loc[0, 'No of Categorical Variables'] = data.select_dtypes(include='object').shape[1]
        data_info.loc[0, 'No of Logical Variables'] = data.select_dtypes(include='bool').shape[1]
        data_info.loc[0, 'No of Date Variables'] = data.select_dtypes(include='datetime64').shape[1]
        data_info.loc[0, 'No of zero variance variables'] = data.loc[:, data.nunique() == 1].shape[1]

        data_info = data_info.transpose()
        data_info.columns = ['Value']
        data_info['Value'] = data_info['Value'].astype(int)

        return data_info

    def get_missing_values(self, data: pd.DataFrame) -> pd.Series:
        """Returns a Series of missing values sorted in descending order"""
        missing_values = data.isnull().sum()
        return missing_values[missing_values > 0].sort_values(ascending=False)

    def _iqr(self, x: pd.Series) -> float:
        return x.quantile(0.75) - x.quantile(0.25)

    def _outlier_count(self, x: pd.Series) -> int:
        """Calculates outliers using IQR method"""
        upper = x.quantile(0.75) + 1.5 * self._iqr(x)
        lower = x.quantile(0.25) - 1.5 * self._iqr(x)
        return ((x > upper) | (x < lower)).sum()

    def num_count_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Summary of numerical features in terms of count distributions"""
        df_num = df.select_dtypes(include=[np.number])
        summary = pd.DataFrame()

        for col in df_num.columns:
            summary.loc[col, 'Negative values count'] = (df_num[col] < 0).sum()
            summary.loc[col, 'Positive values count'] = (df_num[col] > 0).sum()
            summary.loc[col, 'Zero count'] = (df_num[col] == 0).sum()
            summary.loc[col, 'Unique count'] = df_num[col].nunique()
            summary.loc[col, 'Negative Infinity count'] = (df_num[col] == -np.inf).sum()
            summary.loc[col, 'Positive Infinity count'] = (df_num[col] == np.inf).sum()
            summary.loc[col, 'Missing Percentage'] = df_num[col].isnull().mean() * 100
            summary.loc[col, 'Count of outliers'] = self._outlier_count(df_num[col])

        return summary

    def statistical_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Returns a statistical summary including percentiles"""
        df_num = df.select_dtypes(include=[np.number])
        if df_num.empty:
            return pd.DataFrame()

        try:
            stats_df = df_num.describe().transpose()
            stats_df['10%'] = df_num.quantile(0.10)
            stats_df['90%'] = df_num.quantile(0.90)
            stats_df['95%'] = df_num.quantile(0.95)
            stats_df = stats_df.rename(columns={
                '25%': 'Q1', '50%': 'Median', '75%': 'Q3'
            })
        except Exception as e:
            print("Error in statistical_summary:", e)
            stats_df = pd.DataFrame()

        return stats_df
    


class Data_Base_Modelling:

    def __init__(self):
        print("Data_Base_Modelling object created")

    def label_encoding(self, df):
        category_cols = [col for col in df.columns if df[col].dtypes == "object"]
        label_encoder = preprocessing.LabelEncoder()
        mapping_dict = {}
        
        for col in category_cols:
            df[col] = label_encoder.fit_transform(df[col])
            
            le_name_mapping = {}
            for i, class_name in enumerate(label_encoder.classes_):
                le_name_mapping[class_name] = i
        
            mapping_dict[col] = le_name_mapping
        
        return mapping_dict

    def imputer(self, df):
        numeric_cols = df.select_dtypes(include='number')
        non_numeric_cols = df.select_dtypes(exclude='number')

        imp_mean = IterativeImputer(random_state=0)
        imputed_array = imp_mean.fit_transform(numeric_cols)

        imputed_numeric_df = pd.DataFrame(imputed_array, columns=numeric_cols.columns, index=numeric_cols.index)

        combined_df = pd.concat([imputed_numeric_df, non_numeric_cols], axis=1)

        final_df = combined_df[df.columns]

        return final_df

    def run_model(self, x_train, y_train, x_test, y_test, model, model_name="Model"):
        pipeline = Pipeline([(f'{model_name}_classifier', model)])
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        report = classification_report(y_test, predictions)
        print(f"Classification Report for {model_name}:\n")
        return report

    def logistic_regression(self, x_train, y_train, x_test, y_test):
        return self.run_model(x_train, y_train, x_test, y_test, LogisticRegression(), "LogisticRegression")

    def decision_tree(self, x_train, y_train, x_test, y_test):
        return self.run_model(x_train, y_train, x_test, y_test, DecisionTreeClassifier(), "DecisionTree")

    def random_forest(self, x_train, y_train, x_test, y_test):
        return self.run_model(x_train, y_train, x_test, y_test, RandomForestClassifier(), "RandomForest")

    def naive_bayes(self, x_train, y_train, x_test, y_test):
        return self.run_model(x_train, y_train, x_test, y_test, GaussianNB(), "NaiveBayes")

    def xgb_classifier(self, x_train, y_train, x_test, y_test):
        return self.run_model(x_train, y_train, x_test, y_test, XGBClassifier(), "XGBoost")
