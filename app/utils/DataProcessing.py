# encoding:utf-8
import pandas as pd
from datetime import datetime
from sklearn.utils import shuffle
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier


# input_data_path = '../data/bank-full.csv'
# data = pd.read_csv(input_data_path,sep=';')
# print(data.info())
# # 数据缺失补全
# #2.1 查看缺失数据
# for i in data.columns:
#     if type(data[i][0]) is str: #从data.describe中发现，数值型数据没有缺失，因此只需要寻找非数值型数据
#         print("unknown value count in {} :{}\t".format(i,(data[data[i]=='unknown']['y']).count()))
#         # job:288 education:1857 contact:13020 poutcome:36959
# 2.2 缺失数据处理
    #将非数值化数据进行分类变量数值化(编码)
"""
二分类变量的编码
default,housing,loan,均为yes或者no
"""
def encode_bin_attrs(data,bin_attrs):
    for i in bin_attrs:
        data.loc[data[i] == 'no',i] = 0
        data.loc[data[i] == 'yes',i] = 1
    return data

    """
    有序分类变量编码 比如 education的排名，按照等级123 [primary,secondary,tertiary],从低到高
    """
def encode_edu_attrs(data):
    values = ['primary','secondary','tertiary']
    levels = (1,2,3)
    dict_levels = dict(zip(values,levels))
    for i in values:
        data.loc[data['education'] == i,'education'] = dict_levels[i]
    return data

    """
    无序分类变量编码 使用哑变量进行编码，一般n个分类需要设置n-1个哑变量
    job, marital, contact, month
    """

def encode_cate_attrs(data,cate_attrs):
    data = encode_edu_attrs(data)
    cate_attrs.remove('education')
    for i in cate_attrs:
        dummies_df = pd.get_dummies(data[i])
        dummies_df = dummies_df.rename(columns=lambda x: i+'_'+str(x))
        data = pd.concat([data,dummies_df],axis=1)
        data = data.drop(i,axis=1)
    return data

#归一化数值型的数据
def trans_num_attrs(data,numeric_attrs):
    bin_num = 10
    bin_age_attr = 'age'
    data[bin_age_attr] = pd.qcut(data[bin_age_attr],bin_num)
    data[bin_age_attr] = pd.factorize(data[bin_age_attr])[0] +1

    for i in numeric_attrs:
        scaler = preprocessing.StandardScaler()
        data[i] = scaler.fit_transform(data[i].reshape(-1,1))
    return data

# 使用随机森林来补全缺失的数据
def train_predict_unknown(trainX,trainY,testX):
    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest = random_forest.fit(trainX,trainY)
    test_predictY = random_forest.predict(testX).astype(int)
    return pd.DataFrame(test_predictY,index=testX.index)


#是否delete列poutcome，因为缺失太多
def fill_unknown(data,bin_attrs,cate_attrs,numeric_attrs):
    fill_attrs = []
    for i in bin_attrs+cate_attrs:
        if data[data[i] == 'unknown']['y'].count() < 500:
            #delete the col contain unknown
            data =  data[data[i] != 'unknown']
        else:
            fill_attrs.append(i)
    data = encode_cate_attrs(data,cate_attrs)
    data = encode_bin_attrs(data,bin_attrs)
    data = trans_num_attrs(data,numeric_attrs)
    data['y'] = data['y'].map({'no':0,'yes':1}).astype(int)
    data['default'] = data['default'].astype(np.int32)
    data['loan'] = data['loan'].astype(np.int32)
    data['housing'] = data['housing'].astype(np.int32)
    for i in fill_attrs:
        # we use it predict the unknown data
        test_data = data[data[i] == 'unknown']
        testX = test_data.drop(fill_attrs,axis=1)
        train_data = data[data[i] != 'unknown'].astype(np.int32)
        trainY = train_data[i]
        trainX = train_data.drop(fill_attrs,axis=1)
        test_data[i] = train_predict_unknown(trainX,trainY,testX)
        data = pd.concat([train_data,test_data])
    return data

# Preprocess the data

def preprocess_data():
    input_data_path = '../data/bank-full.csv'
    processed_data_path = '../data/bank_processed.csv'
    print("Loading data...")
    data = pd.read_csv(input_data_path,sep=';')
    print("Preprocessing data...")
    numerics_attrs = ['age','balance','day','duration','campaign','pdays','previous']
    bin_attrs = ['default','housing','loan']
    cate_attrs = ['job','marital','month','education']
    # 因为poutcome和contact的数据缺失太多超过10%，我们直接选择删除
    data = data.drop(['poutcome','contact'],axis=1)
    data = shuffle(data) # random shuffle the original data
    data = fill_unknown(data,bin_attrs,cate_attrs,numerics_attrs)
    data.to_csv(processed_data_path,index=False)

if __name__ == '__main__':
    start_time = datetime.now()
    preprocess_data()
    end_time = datetime.now()
    total_time = (end_time - start_time).seconds
    print("Cost time:{}s".format(total_time))





