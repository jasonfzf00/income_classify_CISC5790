import pandas as pd
import os as os
from sklearn.preprocessing import LabelEncoder

schema = ["age","workclass","fnlwgt","education","education-num","marital-status"
          ,"occupation","relationship","race","sex","capital-gain","capital-loss",
          "hours-per-week","native-country","income"]

#Read csv and convert "?" into NaN
df = pd.read_csv(os.getcwd()+'/data/census-income-data.csv', names=schema,sep=',\s',na_values=["?"],engine="python")

#Remove All NaN rows
df = df.dropna()
    
# util function for label encoding
def label_encode_categorial_features(df):
    categorial_feature_names = df.dtypes[df.dtypes == "object"].index.to_list() #Find catagorical features
    label_encoder = {}
    for cfn in categorial_feature_names:
        le = LabelEncoder()
        label_encoder[cfn] = le
        df[cfn] = le.fit_transform(df[cfn].to_list())
    return label_encoder

label_encoder = label_encode_categorial_features(df)

df_test = pd.read_csv(os.getcwd()+'/data/census-income.test.csv',names=schema,sep=',\s',na_values=["?"],engine="python")


print(df_test.isnull().sum())

#Save Cleaned Dataset
#df.to_csv(os.getcwd()+'/data/census-income-data-cleaned.csv',index=False)

