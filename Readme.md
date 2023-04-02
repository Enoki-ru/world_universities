# Анализ Университетов Мира (2023)
Я всех приветствую в очередной работе, где мы будет на примере датасета, содержащим данные по многим университетам мира, рассматривать университеты России и мира в целом, чтобы понять, куда,
возможно, стоит пойти учиться, если вдруг вы захотите целенаправленно изучать что-то для себя новое в разных странах и сферах в целом.

Данные я взял уже из готовой таблицы, собранно во время анализа в 2023 году, с сайта Kaggle.com.
Посмотреть и скачать данныые можно по ссылке: https://www.kaggle.com/datasets/tariqbashir/world-university-ranking-2023

---
## Описание датасета
```
World University Rankings 2023 is based upon 1,799 universities across 104 countries and regions based on many (at least 13) performance indicators that measure teaching, research, knowledge transfer, and international outlook. Data was collected from over 2,500 institutions, including survey responses from 40,000 scholars and analysis of over 121 million citations in 15.5 million research publications. The US has the most institutions overall and in the top 200, but China has overtaken Australia for the fourth-highest number of institutions in the top 200. The University of Oxford is ranked first for this year, while the highest new entry is Italy's Humanitas University.
```

---

Комментарий: Судя по данным, даже после обработки всех значений и к приведению их к нормальному виду получается >2000 (2345 rows если быть точным) университетов, поэтому фраза `based upon 1,799 universities` заставляет меня задуматься, где это я ошибаюсь в вычислениях
## Импорт необходимых библиотек Python



```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
```


```python
db=pd.read_csv('dataset.csv',encoding='ISO-8859-1') # Без данного encoding параметра вылезет ошибка чтения файла
db
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Rank</th>
      <th>Name</th>
      <th>No. of FTE Students</th>
      <th>No. of students per staff</th>
      <th>International Students</th>
      <th>Female:Male Ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>Country/Region</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Reporter</td>
      <td>Zarqa University</td>
      <td>5,768</td>
      <td>18.1</td>
      <td>32%</td>
      <td>47:53:00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>Jordan</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Reporter</td>
      <td>Ziauddin University</td>
      <td>4,906</td>
      <td>8.8</td>
      <td>1%</td>
      <td>63:37:00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>Pakistan</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4872</th>
      <td>NaN</td>
      <td>United Kingdom</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4873</th>
      <td>2</td>
      <td>Harvard University</td>
      <td>21,887</td>
      <td>9.6</td>
      <td>25%</td>
      <td>50:50:00</td>
    </tr>
    <tr>
      <th>4874</th>
      <td>NaN</td>
      <td>United States</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4875</th>
      <td>1</td>
      <td>University of Oxford</td>
      <td>20,965</td>
      <td>10.6</td>
      <td>42%</td>
      <td>48:52:00</td>
    </tr>
    <tr>
      <th>4876</th>
      <td>NaN</td>
      <td>United Kingdom</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>4877 rows × 6 columns</p>
</div>



## Комментарий к датасету
Вы только посмотрите на это безумие! NaN Строки, в одном столбце находятся разные параметры, нужно срочно приводить всё к нормальному виду.

---
## Нормализация таблицы


```python
db.drop(labels=0, inplace=True)
db.reset_index(inplace=True, drop=True)
def find_irregularity(db,full=False):
    ''' 
    Проверим, вся ли таблица имеет такой вид, или она неоднородна
    '''
    prev=-1
    sum=0
    for index, row in db.iterrows():
        # print(index,row['Rank'])
        if pd.isna(row['Rank']):
            if index-prev!=2:
                sum+=1
                if sum<3 or full==True: #Чтобы вывод не забивался поставил ограничение на вывод: sum<10
                    print('-------------------------')
                    print(f'Внимание, появилась неоднородность в данных')
                    for i in range(index-2,index+3):
                        print(i,db['Rank'].iloc[i],db['Name'].iloc[i], index-prev)
            prev=index
    print(f"---------------------\nКол-во неодноровностей:{sum}")
    if sum==0:
        print(f"Congrats, all clear!")
find_irregularity(db)
```

    -------------------------
    Внимание, появилась неоднородность в данных
    38 Reporter Western Caspian University 1
    39 nan Azerbaijan 1
    40 nan Explore 1
    41 Reporter ?Wellspring University 1
    42 nan Nigeria 1
    -------------------------
    Внимание, появилась неоднородность в данных
    67 Reporter National University of Uzbekistan named after Mirzo Ulugbek 1
    68 nan Uzbekistan 1
    69 nan Explore 1
    70 Reporter University of Uyo 1
    71 nan Nigeria 1
    ---------------------
    Кол-во неодноровностей:186
    

## Вывод:
Как видите, тут есть неоднородность в данных. Помимо того, что в одном столбце находятся и название университетов, и их страны. Так еще есть некая строка Explore, которая меняет последовательность, не давая мне правильно и быстро переделать таблицу, не портя ее структуру.
Если внимательно изучить таблицу, то можно предположить, что строки Explore нам вообще не нужны, они ничего нам не дают от слова совсем. Уберем их отсюда.


```python
db=db[db['Name']!='Explore']
db.reset_index(inplace=True, drop=True)
find_irregularity(db, full=True)
```

    -------------------------
    Внимание, появилась неоднородность в данных
    2346 10011200 St Marianna University School of Medicine 1
    2347 nan Japan 1
    2348 nan Not accredited 1
    2349 10011200 ?tefan cel Mare University of Suceava 1
    2350 nan Romania 1
    ---------------------
    Кол-во неодноровностей:1
    

Как видите, осталась только одна неоднородность, причем очень странная. Посмотрим на нее поближе


```python
db.iloc[2345:2351]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Rank</th>
      <th>Name</th>
      <th>No. of FTE Students</th>
      <th>No. of students per staff</th>
      <th>International Students</th>
      <th>Female:Male Ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2345</th>
      <td>NaN</td>
      <td>United Kingdom</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2346</th>
      <td>10011200</td>
      <td>St Marianna University School of Medicine</td>
      <td>833</td>
      <td>0.9</td>
      <td>1%</td>
      <td>39 : 61</td>
    </tr>
    <tr>
      <th>2347</th>
      <td>NaN</td>
      <td>Japan</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2348</th>
      <td>NaN</td>
      <td>Not accredited</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2349</th>
      <td>10011200</td>
      <td>?tefan cel Mare University of Suceava</td>
      <td>9,239</td>
      <td>24.8</td>
      <td>14%</td>
      <td>59:41:00</td>
    </tr>
    <tr>
      <th>2350</th>
      <td>NaN</td>
      <td>Romania</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



Как видно, есть по сути пустая строка с ненаписанными данными тк университет, скорее всего, Не аккредитован.
> Unaccredited Universities is a list of colleges, universities, and other institutions that do not have the equivalent of regional academic accreditation. Some of these institutions may have legal authority to enroll students and grant degrees, but do not have regional academic accreditation for various reasons.

Найти не аккредитованные университеты можно по ссылке: https://www.scholaro.com/unaccredited-universities/

В нашем случае, просто избавимся от всех таких строк, если они есть (вроде одна только)


```python
db=db[db['Name']!='Not accredited']
db.reset_index(inplace=True, drop=True)
find_irregularity(db, full=True) 
```

    ---------------------
    Кол-во неодноровностей:0
    Congrats, all clear!
    


```python
countries=[]
indexes=[]
for index, row in db.iterrows():
    if index%2==1:
        countries.append(row['Name'])
        indexes.append(index)
db.drop(labels=indexes, inplace=True)
db.reset_index(inplace=True, drop=True)
```


```python
db.insert(2,'Country',countries)
db
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Rank</th>
      <th>Name</th>
      <th>Country</th>
      <th>No. of FTE Students</th>
      <th>No. of students per staff</th>
      <th>International Students</th>
      <th>Female:Male Ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Reporter</td>
      <td>Zarqa University</td>
      <td>Jordan</td>
      <td>5,768</td>
      <td>18.1</td>
      <td>32%</td>
      <td>47:53:00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Reporter</td>
      <td>Ziauddin University</td>
      <td>Pakistan</td>
      <td>4,906</td>
      <td>8.8</td>
      <td>1%</td>
      <td>63:37:00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Reporter</td>
      <td>Zhytomyr Polytechnic State University</td>
      <td>Ukraine</td>
      <td>3,869</td>
      <td>15.4</td>
      <td>1%</td>
      <td>34 : 66</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Reporter</td>
      <td>Yusuf Maitama Sule University, Kano</td>
      <td>Nigeria</td>
      <td>12,880</td>
      <td>33.0</td>
      <td>0%</td>
      <td>48:52:00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Reporter</td>
      <td>York St John University</td>
      <td>United Kingdom</td>
      <td>6,315</td>
      <td>18.6</td>
      <td>12%</td>
      <td>65:35:00</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2340</th>
      <td>5</td>
      <td>Massachusetts Institute of Technology</td>
      <td>United States</td>
      <td>11,415</td>
      <td>8.2</td>
      <td>33%</td>
      <td>40 : 60</td>
    </tr>
    <tr>
      <th>2341</th>
      <td>3</td>
      <td>Stanford University</td>
      <td>United States</td>
      <td>16,164</td>
      <td>7.1</td>
      <td>24%</td>
      <td>46:54:00</td>
    </tr>
    <tr>
      <th>2342</th>
      <td>3</td>
      <td>University of Cambridge</td>
      <td>United Kingdom</td>
      <td>20,185</td>
      <td>11.3</td>
      <td>39%</td>
      <td>47:53:00</td>
    </tr>
    <tr>
      <th>2343</th>
      <td>2</td>
      <td>Harvard University</td>
      <td>United States</td>
      <td>21,887</td>
      <td>9.6</td>
      <td>25%</td>
      <td>50:50:00</td>
    </tr>
    <tr>
      <th>2344</th>
      <td>1</td>
      <td>University of Oxford</td>
      <td>United Kingdom</td>
      <td>20,965</td>
      <td>10.6</td>
      <td>42%</td>
      <td>48:52:00</td>
    </tr>
  </tbody>
</table>
<p>2345 rows × 7 columns</p>
</div>



---
## Ошибки в тексте

Вы могли уже заметить, что некоторые значения выглядят немного странно. Давайте я вам перечислю, что нас не может устраивать в данных:

1. Данные из столбцов, к примеру No. of FTE Students написаны с запятой а не с точкой, поэтому числа не воспринимаются как числа. Давайте убедимся в этом, проверив тип данных у всех значений.


```python
db.dtypes
```




    Rank                          object
    Name                          object
    Country                       object
    No. of FTE Students           object
    No. of students per staff    float64
    International Students        object
    Female:Male Ratio             object
    dtype: object



Как видите, только у столбца 'No. of students per staff' нет проблем (пока что). Давайте менять данные, приводя их к нормальному виду!

p.s. На самом деле, стоило бы руки оторвать тому человеку, который эти данные в таком виде собрал. Уж прогнать их через Power Query можно было бы, чтобы не было у нас уже проблем с обработкой.




```python
for index, row in db.iterrows():
    db['No. of FTE Students'][index]=db['No. of FTE Students'][index].replace(',', '.')
    db['No. of FTE Students'][index]=float(db['No. of FTE Students'][index])

print(db['No. of FTE Students'])
```

    C:\Users\enoki\AppData\Local\Temp\ipykernel_7600\2364009046.py:3: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      db['No. of FTE Students'][index]=db['No. of FTE Students'][index].replace(',', '.')
    C:\Users\enoki\AppData\Local\Temp\ipykernel_7600\2364009046.py:4: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      db['No. of FTE Students'][index]=float(db['No. of FTE Students'][index])
    

    0        5.768
    1        4.906
    2        3.869
    3        12.88
    4        6.315
             ...  
    2340    11.415
    2341    16.164
    2342    20.185
    2343    21.887
    2344    20.965
    Name: No. of FTE Students, Length: 2345, dtype: object
    


```python

```


```python

```


```python

```
