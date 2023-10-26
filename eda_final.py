import tensorflow as tf
import pandas as pd
import numpy as np
import pathlib
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

def dic(path_to_folder):
    '''
       создает dic.pickle с уникальными значенями по
       всем столбцам и по всем файлам данных для
       подсчета количества
    '''
    path = path_to_folder#r'C:\Users\Admin\Desktop\MTS ML CUP\competition_data_final_pqt'
    files_parquet = pathlib.Path(path).glob('*.parquet')

    region_name = set()
    city_name = set()
    cpe_manufacturer_name = set()
    cpe_model_name = set()
    url_host = set()
    cpe_type_cd = set()
    cpe_model_os_type = set()
    price = set()
    date = set()
    part_of_day = set()
    request_cnt = set()
    user_id = set()

    i = 1
    for file in files_parquet:
        ds_df = pd.read_parquet(file)
        region_name = region_name.union(set(ds_df['region_name']))
        city_name = city_name.union(set(ds_df['city_name']))
        cpe_manufacturer_name = cpe_manufacturer_name.union(set(ds_df['cpe_manufacturer_name']))
        cpe_model_name = cpe_model_name.union(set(ds_df['cpe_model_name']))
        url_host = url_host.union(set(ds_df['url_host']))
        cpe_type_cd = cpe_type_cd.union(set(ds_df['cpe_type_cd']))
        cpe_model_os_type = cpe_model_os_type.union(set(ds_df['cpe_model_os_type']))
        price = price.union(set(ds_df['price']))
        date = date.union(set(ds_df['date']))
        part_of_day = part_of_day.union(set(ds_df['part_of_day']))
        request_cnt = request_cnt.union(set(ds_df['request_cnt']))
        user_id = user_id.union(set(ds_df['user_id']))
        print(i)
        i += 1
        del ds_df
        
    price_array = np.array(list(price))
    price_without_nan = price_array[~np.isnan(price_array)]
    price_without_nan = list(price_without_nan)
    
    dic = {'region_name':region_name,
           'city_name':city_name,
           'cpe_manufacturer_name':cpe_manufacturer_name,
           'cpe_model_name':cpe_model_name,
           'url_host':url_host,
           'cpe_type_cd':cpe_type_cd,
           'cpe_model_os_type':cpe_model_os_type,
           'price':price_without_nan,
           'date':date,
           'part_of_day':part_of_day,
           'request_cnt':request_cnt,
           'user_id':user_id}

    for key in dic_new.keys():
        dic_new[key] = list(dic_new[key])
        dic_new[key].sort()

    with open('dic.pickle', 'wb') as f:
        pickle.dump(dic, f)

def n_na(path_to_folder, file_number):
    '''
       подсчитывает количество nan по столбцам
       во всех файлах данных
    '''
    path = path_to_folder#r'C:\Users\Admin\Desktop\MTS ML CUP\competition_data_final_pqt'
    files_parquet = pathlib.Path(path).glob('*.parquet')
    files_parquet = list(files_parquet)

    i = file_number # номер файла в files_parquet для подсчета nan
    for file in files_parquet[i:]:
        ds_df = pd.read_parquet(file)
        na = ds_df.isna().sum()
        print(f'Файл номер{i} имеет {na} записей с NaN')
        i += 1
    
def intersection(path_to_folder):
    '''
       проверяет есть ли пересечение в поле user_id
       по всем файлам датасета
    '''
    path = path_to_folder#r'C:\Users\Admin\Desktop\MTS ML CUP\competition_data_final_pqt'
    files_parquet = pathlib.Path(path).glob('*.parquet')
    user_id = {}
    for i, file in enumerate(files_parquet):
        ds_df = pd.read_parquet(file)
        user_id[i] = set(ds_df['user_id'])
        print(i)
    for i in range(10):
        for j in range((i + 1), 10):
            if i != j:
                print(f'{i} - {j} - количество пересечений по user_id\
                        {len(user_id[i].intersection(user_id[j]))}')

def ds_df_pivot(path_to_folder):
    '''
       создает сводную таблицу part_of_day(0 - 3) х дни недели (0 - 6)
       с суммой входов для уникальных user_id
    '''
    path = path_to_folder#r'C:\Users\Admin\Desktop\MTS ML CUP\competition_data_final_pqt'
    files_parquet = pathlib.Path(path).glob('*.parquet')
    for i, file in enumerate(files_parquet):
        ds_df = pd.read_parquet(file)
        ds_df['date'] = pd.to_datetime(ds_df['date']).dt.weekday
        ds_df['part_of_day'].replace(to_replace=['morning', 'day', 'evening',
                                                 'night'],
                                     value=[0, 1, 2, 3], inplace=True)
        table = pd.pivot_table(data=ds_df, values='request_cnt',
                               index=['user_id', 'part_of_day'],
                               columns='date', aggfunc=np.sum,
                               fill_value=0)
        if i == 0:
            ds_df_pivot = table
        else:
            ds_df_pivot = pd.concat([ds_df_pivot, table])
    with open('df_pivot.pickle', 'wb') as f:
        pickle.dump(ds_df_pivot, f)

def make_train_and_test():
    '''
       создает тренировочный и тестовый датасеты на основе
       сводной таблицы part_of_day(0 - 3) х дни недели (0 - 6)
       с суммой входов
    '''
    with open('df_pivot.pickle', 'rb') as f:
        df_pivot = pickle.load(f)

    user_id_in_df_pivot = []
    for ind in df_pivot.index:
        user_id_in_df_pivot.append(ind[0])
    user_id_in_df_pivot = list(set(user_id_in_df_pivot))

    public_train = pd.read_parquet("public_train.pqt")
    public_train.dropna(inplace=True)
    submit_2 = pd.read_parquet("submit_2.pqt")
    #len(set(public_train['user_id']).intersection(submit_2['user_id']))
    train_np_array = np.zeros(shape=(len(public_train), 28))
    for i, user_id in enumerate(public_train['user_id']):
        if user_id in user_id_in_df_pivot:
            for j in df_pivot.loc[user_id].index:
                train_np_array[i, j*7:(j * 7)+7] = df_pivot.loc[user_id].loc[j]
        if not i % 1000:
            print(i)
    with open('train_np_array.pickle', 'wb') as f:
        pickle.dump(train_np_array, f)

    test_np_array = np.zeros(shape=(len(submit_2), 28))
    for i, user_id in enumerate(submit_2['user_id']):
        if user_id in user_id_in_df_pivot:
            for j in df_pivot.loc[user_id].index:
                test_np_array[i, j*7:(j * 7)+7] = df_pivot.loc[user_id].loc[j]
        if not i % 1000:
            print(i)
    with open('test_np_array.pickle', 'wb') as f:
        pickle.dump(test_np_array, f)

def make_df_text(path_to_folder):
    '''
        создает для каждого уникального user_id текст, состоящий
        из уникальных слов, входящих в url_host. Каждое слово входит в текст 1 раз
    '''
    path = path_to_folder#r'C:\Users\Admin\Desktop\MTS ML CUP\competition_data_final_pqt'
    files_parquet = pathlib.Path(path).glob('*.parquet')
    dic_text = {}
    for i, file in enumerate(files_parquet):
        ds_df = pd.read_parquet(file)
        ds_df['url_host'] = ds_df['url_host'].str.split(pat=".")
        for user_id in list(ds_df['user_id'].unique()):
            text = []
            for t in ds_df[ds_df['user_id'] == user_id]['url_host']:
                text.extend(t)
            text = list(set(text))
            dic_text[user_id] = " ".join(text)
        print(i)

    with open('dic_text.pickle', 'wb') as f:
        pickle.dump(dic_text, f)

    df_text = pd.DataFrame.from_dict(data=dic_text, orient='index',
                                     columns=['text'])
    df_text.sort_index(inplace=True)

    with open('df_text.pickle', 'wb') as f:
        pickle.dump(df_text, f)

def make_df_url_host_all(path_to_folder, file_number):
    '''
       создает для каждого уникального user_id текст, состоящий
       из уникальных url_host. Каждый url_host входит в текст столько раз
       сколько встречается в файле
    '''
    path = path_to_folder#r'C:\Users\Admin\Desktop\MTS ML CUP\competition_data_final_pqt'
    files_parquet = pathlib.Path(path).glob('*.parquet')
    i = file_number# счетчик с какого файла начинать обработку
    if i == 0:
        dic_text = {}
    else:
        with open('dic_url_host_all.pickle', 'rb') as f:
            dic_text = pickle.load(f)
    j = i
    for file in list(files_parquet)[i:]:
        ds_df = pd.read_parquet(file)
        for user_id in list(ds_df['user_id'].unique()):
            text = []
            for t in ds_df[ds_df['user_id'] == user_id]['url_host']:
                text.append(t)
            dic_text[user_id] = " ".join(text)
        with open('dic_url_host_all.pickle', 'wb') as f:
            pickle.dump(dic_text, f)
        print(j)
        j += 1    

    df_text = pd.DataFrame.from_dict(data=dic_text, orient='index', columns=['df_url_host_all'])
    df_text.sort_index(inplace=True)

    with open('df_url_host_all.pickle', 'wb') as f:
        pickle.dump(df_text, f)

def sammarize(path_to_folder, file_number):
    '''
       создает для каждого уникального user_id текст, состоящий
       из уникальных 'region_name', 'city_name', 'cpe_manufacturer_name',
       'cpe_model_name', 'cpe_type_cd','cpe_model_os_type', 'url_host'. Каждый входит в текст
       только один раз. Поля указывать вручню и активировать предобработку
    '''
    path = path_to_folder# r'C:\Users\Admin\Desktop\MTS ML CUP\competition_data_final_pqt'
    files_parquet = pathlib.Path(path).glob('*.parquet')
    keys = ['region_name', 'city_name', 'cpe_manufacturer_name',
            'cpe_model_name', 'cpe_type_cd','cpe_model_os_type', 'url_host']
    k = ['region_name', 'city_name', 'cpe_manufacturer_name', 'cpe_model_name',
         'url_host', 'cpe_type_cd', 'cpe_model_os_type', 'price', 'date',
         'part_of_day', 'request_cnt', 'user_id']
    i = file_number # счетчик с какого файла начинать обработку
    if i == 0:
        dic_url_host = {}
    else:
        with open('dic_url_host.pickle', 'rb') as f:
            dic_url_host = pickle.load(f)
    j = i
    for file in list(files_parquet)[i:]:
        ds_df = pd.read_parquet(file)
        #ds_df['region_name'] = ds_df['region_name'].str.replace(pat='\(Якутия\)', repl='')
        #ds_df['region_name'] = ds_df['region_name'].str.replace(pat='-', repl='')
        #ds_df['region_name'] = ds_df['region_name'].str.replace(pat='—', repl='')
        #ds_df['region_name'] = ds_df['region_name'].str.replace(pat=' ', repl='')
        #ds_df['city_name'] = ds_df['city_name'].str.replace(pat='-', repl='')
        #ds_df['city_name'] = ds_df['city_name'].str.replace(pat='—', repl='')
        #ds_df['city_name'] = ds_df['city_name'].str.replace(pat=' ', repl='')
        #ds_df['cpe_manufacturer_name'] = ds_df['cpe_manufacturer_name'].str.replace(pat=' ', repl='')
        #ds_df['cpe_model_name'] = ds_df['cpe_model_name'].str.replace(pat=' ', repl='')
        #ds_df['cpe_model_os_type'] = ds_df['cpe_model_os_type'].str.replace(pat='Apple iOS', repl='iOS')
        for user_id in list(ds_df['user_id'].unique()):
            text = ' '.join(ds_df[ds_df['user_id'] == user_id]['url_host'].unique())
            dic_url_host[user_id] = text
        with open('dic_url_host.pickle', 'wb') as f:
            pickle.dump(dic_url_host, f)
        print(j)
        j += 1

    df_url_host = pd.DataFrame.from_dict(data=dic_url_host, orient='index', columns=['url_host'])
    df_url_host.sort_index(inplace=True)
    with open('df_url_host.pickle', 'wb') as f:
        pickle.dump(df_url_host, f)

def convert_sparse_matrix_to_sparse_tensor(X):
    '''
        преобразует разряженную матрицу sklearn в разряженный тензор tf
    '''
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)

#path_to_folder = r'C:\Users\Admin\Desktop\MTS ML CUP\competition_data_final_pqt'
#dic(path_to_folder)
#n_na(path_to_folder, file_number=0)
#intersection(path_to_folder)
#ds_df_pivot(path_to_folder)
#make_train_and_test()
#make_df_text(path_to_folder)
#sammarize(path_to_folder, file_number=0)
#make_df_url_host_all(path_to_folder, file_number=0)

files_df = ['df_pivot.pickle',
            'df_city_name.pickle',
            'df_cpe_manufacturer_name.pickle',
            'df_cpe_model_name.pickle',
            'df_cpe_model_os_type.pickle',
            'df_cpe_type_cd.pickle',
            'df_region_name.pickle',
            'df_text.pickle',
            'df_url_host.pickle',
            'df_url_host_all.pickle']
submit_2 = pd.read_parquet("submit_2.pqt")
# номера индексов не совпадают с user_id в 79996 случаях
# submit_2['user_id'] совпадает с sample_submission['user_id']
#submit_2.set_index(keys='user_id', drop=False, inplace=True)
public_train = pd.read_parquet("public_train.pqt") # номера индексов совпадают с значениями столбца user_id
# public_train['user_id'] не пересекается с sample_submission['user_id'] и с submit_2['user_id']
public_train.dropna(inplace=True)
sample_submission = pd.read_csv('sample_submission.csv')

########  Векторизация и создание датасетов  ########
with open('df_pivot.pickle', 'rb') as f:
    df_pivot = pickle.load(f)
df_pivot = df_pivot.unstack(level=- 1, fill_value=0)
a1 = np.array(df_pivot, copy=True)
a1 = a1 / a1.max(axis=0)
a2 = np.array(df_pivot, copy=True)
a2 = a2.T
a2 = a2 / a2.max(axis=0)
a2 = a2.T
a3 = np.array(df_pivot, copy=True)
a3 = a3 / a3.max().max()
np_pivot = np.concatenate((a1, a2, a3), axis=1)
df_np_pivot = pd.DataFrame(data=np_pivot, index=df_pivot.index,
                           columns=None, dtype=None, copy=True)

df_test = sample_submission.copy()
df_test.drop(columns=['age', 'is_male'], inplace=True)
df_test = df_test.merge(right=df_np_pivot, how='left', left_on='user_id', right_index=True)
df_test.set_index(keys='user_id', inplace=True)

df_train = public_train.copy()
# в столбеце age есть записи со значением 0 - 29
# в столбце is_male есть записи со значением NA - 5632
df_train.drop(columns=['user_id'], inplace=True)# номера индексов совпадают с значениями столбца user_id
df_train = df_train.merge(right=df_np_pivot, how='left', left_index=True, right_index=True)

for file in files_df[1:]:
    with open(file, 'rb') as f:
        df = pickle.load(f)
    df_test = df_test.merge(right=df, how='left', left_index=True, right_index=True)
    df_train = df_train.merge(right=df, how='left', left_index=True, right_index=True)

df_test['url_host'] = df_test['url_host'].str.replace(pat='.', repl='_')
df_test['url_host'] = df_test['url_host'].str.replace(pat='-', repl='_')
df_test['url_host'] = df_test['url_host'].str.replace(pat='--', repl='_')
df_test['url_host'] = df_test['url_host'].str.replace(pat='\d+', repl='')

df_train['url_host'] = df_train['url_host'].str.replace(pat='.', repl='_')
df_train['url_host'] = df_train['url_host'].str.replace(pat='-', repl='_')
df_train['url_host'] = df_train['url_host'].str.replace(pat='--', repl='_')
df_train['url_host'] = df_train['url_host'].str.replace(pat='\d+', repl='')

df_test['df_url_host_all'] = df_test['df_url_host_all'].str.replace(pat='.', repl='_')
df_test['df_url_host_all'] = df_test['df_url_host_all'].str.replace(pat='-', repl='_')
df_test['df_url_host_all'] = df_test['df_url_host_all'].str.replace(pat='--', repl='_')
df_test['df_url_host_all'] = df_test['df_url_host_all'].str.replace(pat='\d+', repl='')

df_train['df_url_host_all'] = df_train['df_url_host_all'].str.replace(pat='.', repl='_')
df_train['df_url_host_all'] = df_train['df_url_host_all'].str.replace(pat='-', repl='_')
df_train['df_url_host_all'] = df_train['df_url_host_all'].str.replace(pat='--', repl='_')
df_train['df_url_host_all'] = df_train['df_url_host_all'].str.replace(pat='\d+', repl='')

df_test['text'] = df_test['text'].str.replace(pat='.', repl='_')
df_test['text'] = df_test['text'].str.replace(pat='-', repl='_')
df_test['text'] = df_test['text'].str.replace(pat='--', repl='_')
df_test['text'] = df_test['text'].str.replace(pat='\d+', repl='')

df_train['text'] = df_train['text'].str.replace(pat='.', repl='_')
df_train['text'] = df_train['text'].str.replace(pat='-', repl='_')
df_train['text'] = df_train['text'].str.replace(pat='--', repl='_')
df_train['text'] = df_train['text'].str.replace(pat='\d+', repl='')

#df_train['url_host'] = df_train['url_host'].str.split()
#df_train['url_host'].map(lambda x: x.sort())
#df_train['url_host'] = df_train['url_host'].str.join(' ')
#df_train['df_url_host_all'] = df_train['df_url_host_all'].str.split()
#df_train['df_url_host_all'].map(lambda x: x.sort())
#df_train['df_url_host_all'] = df_train['df_url_host_all'].str.join(' ')
df_train['text'] = df_train['text'].str.split()
df_train['text'].map(lambda x: x.sort())
df_train['text'] = df_train['text'].str.join(' ')

df_train.drop_duplicates(subset=['text'], inplace=True)
df_train.drop_duplicates(subset=df_train.keys()[2:-9], inplace=True)

X_train_age = df_train[df_train['age'] >= 19 ].iloc[:, 2:].copy()
y_train_age = df_train[df_train['age'] >= 19 ].iloc[:, 0].copy()
y_train_age[y_train_age > 70] = 71

X_1, X_2, y_1, y_2 = train_test_split(X_train_age, y_train_age, test_size=0.5,
                                      random_state=2, shuffle=True,
                                      stratify=y_train_age)
X_1_1, X_1_2, y_1_1, y_1_2 = train_test_split(X_1, y_1, test_size=0.5,
                                              random_state=2, shuffle=True,
                                              stratify=y_1)
X_2_1, X_2_2, y_2_1, y_2_2 = train_test_split(X_2, y_2, test_size=0.5,
                                              random_state=2, shuffle=True,
                                              stratify=y_2)
X_train_age_1 = pd.concat([X_1_1, X_1_2, X_2_1])
X_val_age_1 = X_2_2
y_train_age_1 = pd.concat([y_1_1, y_1_2, y_2_1])
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\age\y_train_age_1.pickle', 'wb') as f:
    pickle.dump(y_train_age_1, f)
y_val_age_1 = y_2_2
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\age\y_val_age_1.pickle', 'wb') as f:
    pickle.dump(y_val_age_1, f)
X_train_age_2 = pd.concat([X_1_2, X_2_1, X_2_2])
X_val_age_2 = X_1_1
y_train_age_2 = pd.concat([y_1_2, y_2_1, y_2_2])
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\age\y_train_age_2.pickle', 'wb') as f:
    pickle.dump(y_train_age_2, f)
y_val_age_2 = y_1_1
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\age\y_val_age_2.pickle', 'wb') as f:
    pickle.dump(y_val_age_2, f)

X_train_is_male = df_train[df_train['is_male']!= 'NA'].iloc[:, 2:].copy()
y_train_is_male = df_train[df_train['is_male']!= 'NA'].iloc[:, 1].copy()
X_1, X_2, y_1, y_2 = train_test_split(X_train_is_male, y_train_is_male,
                                      test_size=0.5,
                                      random_state=2, shuffle=True,
                                      stratify=y_train_is_male)
X_1_1, X_1_2, y_1_1, y_1_2 = train_test_split(X_1, y_1, test_size=0.5,
                                              random_state=2, shuffle=True,
                                              stratify=y_1)
X_2_1, X_2_2, y_2_1, y_2_2 = train_test_split(X_2, y_2, test_size=0.5,
                                              random_state=2, shuffle=True,
                                              stratify=y_2)
X_train_is_male_1 = pd.concat([X_1_1, X_1_2, X_2_1])
X_val_is_male_1 = X_2_2
y_train_is_male_1 = pd.concat([y_1_1, y_1_2, y_2_1])
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\is_male\y_train_is_male_1.pickle', 'wb') as f:
    pickle.dump(y_train_is_male_1, f)
y_val_is_male_1 = y_2_2
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\is_male\y_val_is_male_1.pickle', 'wb') as f:
    pickle.dump(y_val_is_male_1, f)
X_train_is_male_2 = pd.concat([X_1_2, X_2_1, X_2_2])
X_val_is_male_2 = X_1_1
y_train_is_male_2 = pd.concat([y_1_2, y_2_1, y_2_2])
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\is_male\y_train_is_male_2.pickle', 'wb') as f:
    pickle.dump(y_train_is_male_2, f)
y_val_is_male_2 = y_1_1
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\is_male\y_val_is_male_2.pickle', 'wb') as f:
    pickle.dump(y_val_is_male_2, f)

vectorizer_df_test_city_name = CountVectorizer()
X_df_test_city_name = vectorizer_df_test_city_name.fit_transform(df_test['city_name'])
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\test\X_df_test_city_name.pickle', 'wb') as f:
    pickle.dump(X_df_test_city_name, f)
X_df_train_age_1_city_name = vectorizer_df_test_city_name.transform(X_train_age_1['city_name'])
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\age\X_df_train_age_1_city_name.pickle', 'wb') as f:
    pickle.dump(X_df_train_age_1_city_name, f)
X_df_val_age_1_city_name = vectorizer_df_test_city_name.transform(X_val_age_1['city_name'])
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\age\X_df_val_age_1_city_name.pickle', 'wb') as f:
    pickle.dump(X_df_val_age_1_city_name, f)
X_df_train_age_2_city_name = vectorizer_df_test_city_name.transform(X_train_age_2['city_name'])
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\age\X_df_train_age_2_city_name.pickle', 'wb') as f:
    pickle.dump(X_df_train_age_2_city_name, f)
X_df_val_age_2_city_name = vectorizer_df_test_city_name.transform(X_val_age_2['city_name'])
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\age\X_df_val_age_2_city_name.pickle', 'wb') as f:
    pickle.dump(X_df_val_age_2_city_name, f)
X_df_train_is_male_1_city_name = vectorizer_df_test_city_name.transform(X_train_is_male_1['city_name'])
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\is_male\X_df_train_is_male_1_city_name.pickle', 'wb') as f:
    pickle.dump(X_df_train_is_male_1_city_name, f)
X_df_val_is_male_1_city_name = vectorizer_df_test_city_name.transform(X_val_is_male_1['city_name'])
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\is_male\X_df_val_is_male_1_city_name.pickle', 'wb') as f:
    pickle.dump(X_df_val_is_male_1_city_name, f)
X_df_train_is_male_2_city_name = vectorizer_df_test_city_name.transform(X_train_is_male_2['city_name'])
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\is_male\X_df_train_is_male_2_city_name.pickle', 'wb') as f:
    pickle.dump(X_df_train_is_male_2_city_name, f)
X_df_val_is_male_2_city_name = vectorizer_df_test_city_name.transform(X_val_is_male_2['city_name'])
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\is_male\X_df_val_is_male_2_city_name.pickle', 'wb') as f:
    pickle.dump(X_df_val_is_male_2_city_name, f)

X_df_test_count = df_test.iloc[:, :84]
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\test\X_df_test_count.pickle', 'wb') as f:
    pickle.dump(X_df_test_count, f)
X_df_train_age_1_count = X_train_age_1.iloc[:, :84]
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\age\X_df_train_age_1_count.pickle', 'wb') as f:
    pickle.dump(X_df_train_age_1_count, f)
X_df_val_age_1_count = X_val_age_1.iloc[:, :84]
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\age\X_df_val_age_1_count.pickle', 'wb') as f:
    pickle.dump(X_df_val_age_1_count, f)
X_df_train_age_2_count = X_train_age_2.iloc[:, :84]
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\age\X_df_train_age_2_count.pickle', 'wb') as f:
    pickle.dump(X_df_train_age_2_count, f)
X_df_val_age_2_count = X_val_age_2.iloc[:, :84]
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\age\X_df_val_age_2_count.pickle', 'wb') as f:
    pickle.dump(X_df_val_age_2_count, f)

X_df_train_is_male_1_count = X_train_is_male_1.iloc[:, :84]
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\is_male\X_df_train_is_male_1_count.pickle', 'wb') as f:
    pickle.dump(X_df_train_is_male_1_count, f)
X_df_val_is_male_1_count = X_val_is_male_1.iloc[:, :84]
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\is_male\X_df_val_is_male_1_count.pickle', 'wb') as f:
    pickle.dump(X_df_val_is_male_1_count, f)
X_df_train_is_male_2_count = X_train_is_male_2.iloc[:, :84]
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\is_male\X_df_train_is_male_2_count.pickle', 'wb') as f:
    pickle.dump(X_df_train_is_male_2_count, f)
X_df_val_is_male_2_count = X_val_is_male_2.iloc[:, :84]
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\is_male\X_df_val_is_male_2_count.pickle', 'wb') as f:
    pickle.dump(X_df_val_is_male_2_count, f)

vectorizer_df_test_cpe_manufacturer_name = CountVectorizer()
X_df_test_cpe_manufacturer_name = vectorizer_df_test_cpe_manufacturer_name.fit_transform(df_test['cpe_manufacturer_name'])
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\test\X_df_test_cpe_manufacturer_name.pickle', 'wb') as f:
    pickle.dump(X_df_test_cpe_manufacturer_name, f)
X_df_train_age_1_cpe_manufacturer_name = vectorizer_df_test_cpe_manufacturer_name.transform(X_train_age_1['cpe_manufacturer_name'])
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\age\X_df_train_age_1_cpe_manufacturer_name.pickle', 'wb') as f:
    pickle.dump(X_df_train_age_1_cpe_manufacturer_name, f)
X_df_val_age_1_cpe_manufacturer_name = vectorizer_df_test_cpe_manufacturer_name.transform(X_val_age_1['cpe_manufacturer_name'])
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\age\X_df_val_age_1_cpe_manufacturer_name.pickle', 'wb') as f:
    pickle.dump(X_df_val_age_1_cpe_manufacturer_name, f)
X_df_train_age_2_cpe_manufacturer_name = vectorizer_df_test_cpe_manufacturer_name.transform(X_train_age_2['cpe_manufacturer_name'])
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\age\X_df_train_age_2_cpe_manufacturer_name.pickle', 'wb') as f:
    pickle.dump(X_df_train_age_2_cpe_manufacturer_name, f)
X_df_val_age_2_cpe_manufacturer_name = vectorizer_df_test_cpe_manufacturer_name.transform(X_val_age_2['cpe_manufacturer_name'])
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\age\X_df_val_age_2_cpe_manufacturer_name.pickle', 'wb') as f:
    pickle.dump(X_df_val_age_2_cpe_manufacturer_name, f)
X_df_train_is_male_1_cpe_manufacturer_name = vectorizer_df_test_cpe_manufacturer_name.transform(X_train_is_male_1['cpe_manufacturer_name'])
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\is_male\X_df_train_is_male_1_cpe_manufacturer_name.pickle', 'wb') as f:
    pickle.dump(X_df_train_is_male_1_cpe_manufacturer_name, f)
X_df_val_is_male_1_cpe_manufacturer_name = vectorizer_df_test_cpe_manufacturer_name.transform(X_val_is_male_1['cpe_manufacturer_name'])
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\is_male\X_df_val_is_male_1_cpe_manufacturer_name.pickle', 'wb') as f:
    pickle.dump(X_df_val_is_male_1_cpe_manufacturer_name, f)
X_df_train_is_male_2_cpe_manufacturer_name = vectorizer_df_test_cpe_manufacturer_name.transform(X_train_is_male_2['cpe_manufacturer_name'])
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\is_male\X_df_train_is_male_2_cpe_manufacturer_name.pickle', 'wb') as f:
    pickle.dump(X_df_train_is_male_2_cpe_manufacturer_name, f)
X_df_val_is_male_2_cpe_manufacturer_name = vectorizer_df_test_cpe_manufacturer_name.transform(X_val_is_male_2['cpe_manufacturer_name'])
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\is_male\X_df_val_is_male_2_cpe_manufacturer_name.pickle', 'wb') as f:
    pickle.dump(X_df_val_is_male_2_cpe_manufacturer_name, f)

vectorizer_df_test_cpe_model_name = CountVectorizer()
X_df_test_cpe_model_name = vectorizer_df_test_cpe_model_name.fit_transform(df_test['cpe_model_name'])
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\test\X_df_test_cpe_model_name.pickle', 'wb') as f:
    pickle.dump(X_df_test_cpe_model_name, f)
X_df_train_age_1_cpe_model_name = vectorizer_df_test_cpe_model_name.transform(X_train_age_1['cpe_model_name'])
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\age\X_df_train_age_1_cpe_model_name.pickle', 'wb') as f:
    pickle.dump(X_df_train_age_1_cpe_model_name, f)
X_df_val_age_1_cpe_model_name = vectorizer_df_test_cpe_model_name.transform(X_val_age_1['cpe_model_name'])
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\age\X_df_val_age_1_cpe_model_name.pickle', 'wb') as f:
    pickle.dump(X_df_val_age_1_cpe_model_name, f)
X_df_train_age_2_cpe_model_name = vectorizer_df_test_cpe_model_name.transform(X_train_age_2['cpe_model_name'])
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\age\X_df_train_age_2_cpe_model_name.pickle', 'wb') as f:
    pickle.dump(X_df_train_age_2_cpe_model_name, f)
X_df_val_age_2_cpe_model_name = vectorizer_df_test_cpe_model_name.transform(X_val_age_2['cpe_model_name'])
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\age\X_df_val_age_2_cpe_model_name.pickle', 'wb') as f:
    pickle.dump(X_df_val_age_2_cpe_model_name, f)
X_df_train_is_male_1_cpe_model_name = vectorizer_df_test_cpe_model_name.transform(X_train_is_male_1['cpe_model_name'])
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\is_male\X_df_train_is_male_1_cpe_model_name.pickle', 'wb') as f:
    pickle.dump(X_df_train_is_male_1_cpe_model_name, f)
X_df_val_is_male_1_cpe_model_name = vectorizer_df_test_cpe_model_name.transform(X_val_is_male_1['cpe_model_name'])
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\is_male\X_df_val_is_male_1_cpe_model_name.pickle', 'wb') as f:
    pickle.dump(X_df_val_is_male_1_cpe_model_name, f)
X_df_train_is_male_2_cpe_model_name = vectorizer_df_test_cpe_model_name.transform(X_train_is_male_2['cpe_model_name'])
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\is_male\X_df_train_is_male_2_cpe_model_name.pickle', 'wb') as f:
    pickle.dump(X_df_train_is_male_2_cpe_model_name, f)
X_df_val_is_male_2_cpe_model_name = vectorizer_df_test_cpe_model_name.transform(X_val_is_male_2['cpe_model_name'])
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\is_male\X_df_val_is_male_2_cpe_model_name.pickle', 'wb') as f:
    pickle.dump(X_df_val_is_male_2_cpe_model_name, f)

vectorizer_df_test_cpe_model_os_type = CountVectorizer()
X_df_test_cpe_model_os_type = vectorizer_df_test_cpe_model_os_type.fit_transform(df_test['cpe_model_os_type'])
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\test\X_df_test_cpe_model_os_type.pickle', 'wb') as f:
    pickle.dump(X_df_test_cpe_model_os_type, f)
X_df_train_age_1_cpe_model_os_type = vectorizer_df_test_cpe_model_os_type.transform(X_train_age_1['cpe_model_os_type'])
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\age\X_df_train_age_1_cpe_model_os_type.pickle', 'wb') as f:
    pickle.dump(X_df_train_age_1_cpe_model_os_type, f)
X_df_val_age_1_cpe_model_os_type = vectorizer_df_test_cpe_model_os_type.transform(X_val_age_1['cpe_model_os_type'])
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\age\X_df_val_age_1_cpe_model_os_type.pickle', 'wb') as f:
    pickle.dump(X_df_val_age_1_cpe_model_os_type, f)
X_df_train_age_2_cpe_model_os_type = vectorizer_df_test_cpe_model_os_type.transform(X_train_age_2['cpe_model_os_type'])
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\age\X_df_train_age_2_cpe_model_os_type.pickle', 'wb') as f:
    pickle.dump(X_df_train_age_2_cpe_model_os_type, f)
X_df_val_age_2_cpe_model_os_type = vectorizer_df_test_cpe_model_os_type.transform(X_val_age_2['cpe_model_os_type'])
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\age\X_df_val_age_2_cpe_model_os_type.pickle', 'wb') as f:
    pickle.dump(X_df_val_age_2_cpe_model_os_type, f)
X_df_train_is_male_1_cpe_model_os_type = vectorizer_df_test_cpe_model_os_type.transform(X_train_is_male_1['cpe_model_os_type'])
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\is_male\X_df_train_is_male_1_cpe_model_os_type.pickle', 'wb') as f:
    pickle.dump(X_df_train_is_male_1_cpe_model_os_type, f)
X_df_val_is_male_1_cpe_model_os_type = vectorizer_df_test_cpe_model_os_type.transform(X_val_is_male_1['cpe_model_os_type'])
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\is_male\X_df_val_is_male_1_cpe_model_os_type.pickle', 'wb') as f:
    pickle.dump(X_df_val_is_male_1_cpe_model_os_type, f)
X_df_train_is_male_2_cpe_model_os_type = vectorizer_df_test_cpe_model_os_type.transform(X_train_is_male_2['cpe_model_os_type'])
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\is_male\X_df_train_is_male_2_cpe_model_os_type.pickle', 'wb') as f:
    pickle.dump(X_df_train_is_male_2_cpe_model_os_type, f)
X_df_val_is_male_2_cpe_model_os_type = vectorizer_df_test_cpe_model_os_type.transform(X_val_is_male_2['cpe_model_os_type'])
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\is_male\X_df_val_is_male_2_cpe_model_os_type.pickle', 'wb') as f:
    pickle.dump(X_df_val_is_male_2_cpe_model_os_type, f)

vectorizer_df_test_cpe_type_cd = CountVectorizer()
X_df_test_cpe_type_cd = vectorizer_df_test_cpe_type_cd.fit_transform(df_test['cpe_type_cd'])
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\test\X_df_test_cpe_type_cd.pickle', 'wb') as f:
    pickle.dump(X_df_test_cpe_type_cd, f)
X_df_train_age_1_cpe_type_cd = vectorizer_df_test_cpe_type_cd.transform(X_train_age_1['cpe_type_cd'])
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\age\X_df_train_age_1_cpe_type_cd.pickle', 'wb') as f:
    pickle.dump(X_df_train_age_1_cpe_type_cd, f)
X_df_val_age_1_cpe_type_cd = vectorizer_df_test_cpe_type_cd.transform(X_val_age_1['cpe_type_cd'])
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\age\X_df_val_age_1_cpe_type_cd.pickle', 'wb') as f:
    pickle.dump(X_df_val_age_1_cpe_type_cd, f)
X_df_train_age_2_cpe_type_cd = vectorizer_df_test_cpe_type_cd.transform(X_train_age_2['cpe_type_cd'])
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\age\X_df_train_age_2_cpe_type_cd.pickle', 'wb') as f:
    pickle.dump(X_df_train_age_2_cpe_type_cd, f)
X_df_val_age_2_cpe_type_cd = vectorizer_df_test_cpe_type_cd.transform(X_val_age_2['cpe_type_cd'])
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\age\X_df_val_age_2_cpe_type_cd.pickle', 'wb') as f:
    pickle.dump(X_df_val_age_2_cpe_type_cd, f)
X_df_train_is_male_1_cpe_type_cd = vectorizer_df_test_cpe_type_cd.transform(X_train_is_male_1['cpe_type_cd'])
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\is_male\X_df_train_is_male_1_cpe_type_cd.pickle', 'wb') as f:
    pickle.dump(X_df_train_is_male_1_cpe_type_cd, f)
X_df_val_is_male_1_cpe_type_cd = vectorizer_df_test_cpe_type_cd.transform(X_val_is_male_1['cpe_type_cd'])
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\is_male\X_df_val_is_male_1_cpe_type_cd.pickle', 'wb') as f:
    pickle.dump(X_df_val_is_male_1_cpe_type_cd, f)
X_df_train_is_male_2_cpe_type_cd = vectorizer_df_test_cpe_type_cd.transform(X_train_is_male_2['cpe_type_cd'])
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\is_male\X_df_train_is_male_2_cpe_type_cd.pickle', 'wb') as f:
    pickle.dump(X_df_train_is_male_2_cpe_type_cd, f)
X_df_val_is_male_2_cpe_type_cd = vectorizer_df_test_cpe_type_cd.transform(X_val_is_male_2['cpe_type_cd'])
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\is_male\X_df_val_is_male_2_cpe_type_cd.pickle', 'wb') as f:
    pickle.dump(X_df_val_is_male_2_cpe_type_cd, f)

vectorizer_df_test_region_name = CountVectorizer()
X_df_test_region_name = vectorizer_df_test_region_name.fit_transform(df_test['region_name'])
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\test\X_df_test_region_name.pickle', 'wb') as f:
    pickle.dump(X_df_test_region_name, f)
X_df_train_age_1_region_name = vectorizer_df_test_region_name.transform(X_train_age_1['region_name'])
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\age\X_df_train_age_1_region_name.pickle', 'wb') as f:
    pickle.dump(X_df_train_age_1_region_name, f)
X_df_val_age_1_region_name = vectorizer_df_test_region_name.transform(X_val_age_1['region_name'])
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\age\X_df_val_age_1_region_name.pickle', 'wb') as f:
    pickle.dump(X_df_val_age_1_region_name, f)
X_df_train_age_2_region_name = vectorizer_df_test_region_name.transform(X_train_age_2['region_name'])
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\age\X_df_train_age_2_region_name.pickle', 'wb') as f:
    pickle.dump(X_df_train_age_2_region_name, f)
X_df_val_age_2_region_name = vectorizer_df_test_region_name.transform(X_val_age_2['region_name'])
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\age\X_df_val_age_2_region_name.pickle', 'wb') as f:
    pickle.dump(X_df_val_age_2_region_name, f)
X_df_train_is_male_1_region_name = vectorizer_df_test_region_name.transform(X_train_is_male_1['region_name'])
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\is_male\X_df_train_is_male_1_region_name.pickle', 'wb') as f:
    pickle.dump(X_df_train_is_male_1_region_name, f)
X_df_val_is_male_1_region_name = vectorizer_df_test_region_name.transform(X_val_is_male_1['region_name'])
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\is_male\X_df_val_is_male_1_region_name.pickle', 'wb') as f:
    pickle.dump(X_df_val_is_male_1_region_name, f)
X_df_train_is_male_2_region_name = vectorizer_df_test_region_name.transform(X_train_is_male_2['region_name'])
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\is_male\X_df_train_is_male_2_region_name.pickle', 'wb') as f:
    pickle.dump(X_df_train_is_male_2_region_name, f)
X_df_val_is_male_2_region_name = vectorizer_df_test_region_name.transform(X_val_is_male_2['region_name'])
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\is_male\X_df_val_is_male_2_region_name.pickle', 'wb') as f:
    pickle.dump(X_df_val_is_male_2_region_name, f)

vectorizer_df_test_text = CountVectorizer()
X_df_test_text = vectorizer_df_test_text.fit_transform(df_test['text'])
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\test\X_df_test_text.pickle', 'wb') as f:
    pickle.dump(X_df_test_text, f)
X_df_train_age_1_text = vectorizer_df_test_text.transform(X_train_age_1['text'])
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\age\X_df_train_age_1_text.pickle', 'wb') as f:
    pickle.dump(X_df_train_age_1_text, f)
X_df_val_age_1_text = vectorizer_df_test_text.transform(X_val_age_1['text'])
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\age\X_df_val_age_1_text.pickle', 'wb') as f:
    pickle.dump(X_df_val_age_1_text, f)
X_df_train_age_2_text = vectorizer_df_test_text.transform(X_train_age_2['text'])
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\age\X_df_train_age_2_text.pickle', 'wb') as f:
    pickle.dump(X_df_train_age_2_text, f)
X_df_val_age_2_text = vectorizer_df_test_text.transform(X_val_age_2['text'])
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\age\X_df_val_age_2_text.pickle', 'wb') as f:
    pickle.dump(X_df_val_age_2_text, f)
X_df_train_is_male_1_text = vectorizer_df_test_text.transform(X_train_is_male_1['text'])
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\is_male\X_df_train_is_male_1_text.pickle', 'wb') as f:
    pickle.dump(X_df_train_is_male_1_text, f)
X_df_val_is_male_1_text = vectorizer_df_test_text.transform(X_val_is_male_1['text'])
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\is_male\X_df_val_is_male_1_text.pickle', 'wb') as f:
    pickle.dump(X_df_val_is_male_1_text, f)
X_df_train_is_male_2_text = vectorizer_df_test_text.transform(X_train_is_male_2['text'])
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\is_male\X_df_train_is_male_2_text.pickle', 'wb') as f:
    pickle.dump(X_df_train_is_male_2_text, f)
X_df_val_is_male_2_text = vectorizer_df_test_text.transform(X_val_is_male_2['text'])
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\is_male\X_df_val_is_male_2_text.pickle', 'wb') as f:
    pickle.dump(X_df_val_is_male_2_text, f)

df_train_pivot = pd.read_csv(filepath_or_buffer=r'C:\Users\Admin\Desktop\MTS ML CUP\df_url_host_list_to_label_encoder_train.csv')

df_train_pivot_age = df_train_pivot[df_train_pivot['age'] > 0].pivot_table(values='age', columns='url_host', aggfunc=np.std).T
df_train_pivot_age_count = df_train_pivot[df_train_pivot['age'] > 0].pivot_table(values='age', columns='url_host', aggfunc='count').T
df_train_pivot_age['count'] = df_train_pivot_age_count['age']
voc_age = list(df_train_pivot_age[df_train_pivot_age['count'] > 10][df_train_pivot_age['age'] < 20].index)

df_train_pivot_is_male = df_train_pivot[df_train_pivot['is_male'].notna()].pivot_table(values='is_male', columns='url_host', aggfunc=np.mean).T
df_train_pivot_is_male_count = df_train_pivot[df_train_pivot['is_male'].notna()].pivot_table(values='is_male', columns='url_host', aggfunc='count').T
df_train_pivot_is_male['count'] = df_train_pivot_is_male_count['is_male']
df_is_male = df_train_pivot_is_male[df_train_pivot_is_male['count'] > 10].copy()
is_male = np.array(df_is_male['is_male'])
index_1 = is_male < 0.51
index_2 = is_male > 0.49
index_3 = index_1 * index_2
df_is_male['level'] = index_3
voc_is_male = list(df_is_male[df_is_male['level'] == False].index)

vectorizer_df_test_url_host_age = CountVectorizer(vocabulary=voc_age)
vectorizer_df_test_url_host_is_male = CountVectorizer(vocabulary=voc_is_male)
X_df_test_url_host_age = vectorizer_df_test_url_host_age.transform(df_test['url_host'])
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\test\X_df_test_url_host_age.pickle', 'wb') as f:
    pickle.dump(X_df_test_url_host_age, f)
X_df_test_url_host_is_male = vectorizer_df_test_url_host_is_male.transform(df_test['url_host'])
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\test\X_df_test_url_host_is_male.pickle', 'wb') as f:
    pickle.dump(X_df_test_url_host_is_male, f)
    
X_df_train_age_1_url_host = vectorizer_df_test_url_host_age.transform(X_train_age_1['url_host'])
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\age\X_df_train_age_1_url_host.pickle', 'wb') as f:
    pickle.dump(X_df_train_age_1_url_host, f)
X_df_val_age_1_url_host = vectorizer_df_test_url_host_age.transform(X_val_age_1['url_host'])
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\age\X_df_val_age_1_url_host.pickle', 'wb') as f:
    pickle.dump(X_df_val_age_1_url_host, f)
X_df_train_age_2_url_host = vectorizer_df_test_url_host_age.transform(X_train_age_2['url_host'])
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\age\X_df_train_age_2_url_host.pickle', 'wb') as f:
    pickle.dump(X_df_train_age_2_url_host, f)
X_df_val_age_2_url_host = vectorizer_df_test_url_host_age.transform(X_val_age_2['url_host'])
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\age\X_df_val_age_2_url_host.pickle', 'wb') as f:
    pickle.dump(X_df_val_age_2_url_host, f)
X_df_train_is_male_1_url_host = vectorizer_df_test_url_host_is_male.transform(X_train_is_male_1['url_host'])
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\is_male\X_df_train_is_male_1_url_host.pickle', 'wb') as f:
    pickle.dump(X_df_train_is_male_1_url_host, f)
X_df_val_is_male_1_url_host = vectorizer_df_test_url_host_is_male.transform(X_val_is_male_1['url_host'])
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\is_male\X_df_val_is_male_1_url_host.pickle', 'wb') as f:
    pickle.dump(X_df_val_is_male_1_url_host, f)
X_df_train_is_male_2_url_host = vectorizer_df_test_url_host_is_male.transform(X_train_is_male_2['url_host'])
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\is_male\X_df_train_is_male_2_url_host.pickle', 'wb') as f:
    pickle.dump(X_df_train_is_male_2_url_host, f)
X_df_val_is_male_2_url_host = vectorizer_df_test_url_host_is_male.transform(X_val_is_male_2['url_host'])
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\is_male\X_df_val_is_male_2_url_host.pickle', 'wb') as f:
    pickle.dump(X_df_val_is_male_2_url_host, f)

vectorizer_df_test_url_host_all_age = TfidfVectorizer(vocabulary=voc_age)
vectorizer_df_test_url_host_all_is_male = TfidfVectorizer(vocabulary=voc_is_male)
X_df_train_age_1_url_host_all = vectorizer_df_test_url_host_all_age.fit_transform(X_train_age_1['df_url_host_all'])
X_df_train_age_1_url_host_all = convert_sparse_matrix_to_sparse_tensor(X_df_train_age_1_url_host_all)
X_df_train_age_1_url_host_all = tf.sparse.reorder(sp_input=X_df_train_age_1_url_host_all, name=None)
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\age\X_df_train_age_1_url_host_all.pickle', 'wb') as f:
    pickle.dump(X_df_train_age_1_url_host_all, f)

X_df_test_age_1_url_host_all = vectorizer_df_test_url_host_all_age.transform(df_test['df_url_host_all'])
X_df_test_age_1_url_host_all = convert_sparse_matrix_to_sparse_tensor(X_df_test_age_1_url_host_all)
X_df_test_age_1_url_host_all = tf.sparse.reorder(sp_input=X_df_test_age_1_url_host_all, name=None)
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\test\X_df_test_age_1_url_host_all.pickle', 'wb') as f:
    pickle.dump(X_df_test_age_1_url_host_all, f)

X_df_val_age_1_url_host_all = vectorizer_df_test_url_host_all_age.transform(X_val_age_1['df_url_host_all'])
X_df_val_age_1_url_host_all = convert_sparse_matrix_to_sparse_tensor(X_df_val_age_1_url_host_all)
X_df_val_age_1_url_host_all = tf.sparse.reorder(sp_input=X_df_val_age_1_url_host_all, name=None)
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\age\X_df_val_age_1_url_host_all.pickle', 'wb') as f:
    pickle.dump(X_df_val_age_1_url_host_all, f)

X_df_train_age_2_url_host_all = vectorizer_df_test_url_host_all_age.fit_transform(X_train_age_2['df_url_host_all'])
X_df_train_age_2_url_host_all = convert_sparse_matrix_to_sparse_tensor(X_df_train_age_2_url_host_all)
X_df_train_age_2_url_host_all = tf.sparse.reorder(sp_input=X_df_train_age_2_url_host_all, name=None)
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\age\X_df_train_age_2_url_host_all.pickle', 'wb') as f:
    pickle.dump(X_df_train_age_2_url_host_all, f)

X_df_test_age_2_url_host_all = vectorizer_df_test_url_host_all_age.transform(df_test['df_url_host_all'])
X_df_test_age_2_url_host_all = convert_sparse_matrix_to_sparse_tensor(X_df_test_age_2_url_host_all)
X_df_test_age_2_url_host_all = tf.sparse.reorder(sp_input=X_df_test_age_2_url_host_all, name=None)
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\test\X_df_test_age_2_url_host_all.pickle', 'wb') as f:
    pickle.dump(X_df_test_age_2_url_host_all, f)

X_df_val_age_2_url_host_all = vectorizer_df_test_url_host_all_age.transform(X_val_age_2['df_url_host_all'])
X_df_val_age_2_url_host_all = convert_sparse_matrix_to_sparse_tensor(X_df_val_age_2_url_host_all)
X_df_val_age_2_url_host_all = tf.sparse.reorder(sp_input=X_df_val_age_2_url_host_all, name=None)
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\age\X_df_val_age_2_url_host_all.pickle', 'wb') as f:
    pickle.dump(X_df_val_age_2_url_host_all, f)

X_df_train_is_male_1_url_host_all = vectorizer_df_test_url_host_all_is_male.fit_transform(X_train_is_male_1['df_url_host_all'])
X_df_train_is_male_1_url_host_all = convert_sparse_matrix_to_sparse_tensor(X_df_train_is_male_1_url_host_all)
X_df_train_is_male_1_url_host_all = tf.sparse.reorder(sp_input=X_df_train_is_male_1_url_host_all, name=None)
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\is_male\X_df_train_is_male_1_url_host_all.pickle', 'wb') as f:
    pickle.dump(X_df_train_is_male_1_url_host_all, f)

X_df_test_is_male_1_url_host_all = vectorizer_df_test_url_host_all_is_male.transform(df_test['df_url_host_all'])
X_df_test_is_male_1_url_host_all = convert_sparse_matrix_to_sparse_tensor(X_df_test_is_male_1_url_host_all)
X_df_test_is_male_1_url_host_all = tf.sparse.reorder(sp_input=X_df_test_is_male_1_url_host_all, name=None)
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\test\X_df_test_is_male_1_url_host_all.pickle', 'wb') as f:
    pickle.dump(X_df_test_is_male_1_url_host_all, f)

X_df_val_is_male_1_url_host_all = vectorizer_df_test_url_host_all_is_male.transform(X_val_is_male_1['df_url_host_all'])
X_df_val_is_male_1_url_host_all = convert_sparse_matrix_to_sparse_tensor(X_df_val_is_male_1_url_host_all)
X_df_val_is_male_1_url_host_all = tf.sparse.reorder(sp_input=X_df_val_is_male_1_url_host_all, name=None)
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\is_male\X_df_val_is_male_1_url_host_all.pickle', 'wb') as f:
    pickle.dump(X_df_val_is_male_1_url_host_all, f)

X_df_train_is_male_2_url_host_all = vectorizer_df_test_url_host_all_is_male.fit_transform(X_train_is_male_2['df_url_host_all'])
X_df_train_is_male_2_url_host_all = convert_sparse_matrix_to_sparse_tensor(X_df_train_is_male_2_url_host_all)
X_df_train_is_male_2_url_host_all = tf.sparse.reorder(sp_input=X_df_train_is_male_2_url_host_all, name=None)
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\is_male\X_df_train_is_male_2_url_host_all.pickle', 'wb') as f:
    pickle.dump(X_df_train_is_male_2_url_host_all, f)

X_df_test_is_male_2_url_host_all = vectorizer_df_test_url_host_all_is_male.transform(df_test['df_url_host_all'])
X_df_test_is_male_2_url_host_all = convert_sparse_matrix_to_sparse_tensor(X_df_test_is_male_2_url_host_all)
X_df_test_is_male_2_url_host_all = tf.sparse.reorder(sp_input=X_df_test_is_male_2_url_host_all, name=None)
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\test\X_df_test_is_male_2_url_host_all.pickle', 'wb') as f:
    pickle.dump(X_df_test_is_male_2_url_host_all, f)

X_df_val_is_male_2_url_host_all = vectorizer_df_test_url_host_all_is_male.transform(X_val_is_male_2['df_url_host_all'])
X_df_val_is_male_2_url_host_all = convert_sparse_matrix_to_sparse_tensor(X_df_val_is_male_2_url_host_all)
X_df_val_is_male_2_url_host_all = tf.sparse.reorder(sp_input=X_df_val_is_male_2_url_host_all, name=None)
with open(r'C:\Users\Admin\Desktop\MTS ML CUP\train\is_male\X_df_val_is_male_2_url_host_all.pickle', 'wb') as f:
    pickle.dump(X_df_val_is_male_2_url_host_all, f)
