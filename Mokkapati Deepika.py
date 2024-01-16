#!/usr/bin/env python
# coding: utf-8

# In[1]:


# https://data.worldbank.org/indicator/AG.LND.FRST.K2


# In[2]:


import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import errors
import scipy.optimize as opt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as skmet
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


def worldbank_data(filename):
    """
    Read World Bank data from a CSV file.

    Args:
    - filename: The path to the CSV file.

    Returns:
    - df: The original DataFrame.
    - dft: Transposed DataFrame.
    """

    df = pd.read_csv(filename, skiprows=4)
    dft = df.transpose()
    dft.columns = dft.iloc[0]
    
    return df, dft


# In[4]:


df, dft = worldbank_data('API_AG.LND.FRST.K2_DS2_en_csv_v2_6302271.csv')


# In[5]:


df.head()


# In[15]:


sub = df[['Country Name', 'Indicator Name'] + list(map(str, range(2001, 2022)))]
sub = sub.dropna()
subx = sub[["Country Name", "2021"]].copy()
subx.head()


# In[16]:


subx["Growth"] = 100.0 * (sub["2021"] - sub["2001"]) / (sub["2001"])
subx.describe()


# In[18]:


def remove_outliers(data_frame, column_names):
    """
    Remove outliers from specified columns in the DataFrame.

    Returns:
    - Dataframe with outliers removed.
    """
    Q1 = data_frame[column_names].quantile(0.25)
    Q3 = data_frame[column_names].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    cleaned_data = data_frame[~((data_frame[column_names] < lower_bound) | (data_frame[column_names] > upper_bound)).any(axis=1)]
    return cleaned_data

subx = remove_outliers(subx, ["2021", "Growth"])


# In[27]:


subx = subx.dropna()


# In[28]:


plt.figure(figsize=(8, 5))
scatter_plot = plt.scatter(subx["2021"], subx["Growth"], 20, label="Countries", c = 'red')
plt.xlabel("Forest area (sq. km) in 2021")
plt.ylabel("Growth (%) Since 2001")
plt.title("Forest area (sq. km) in 2021 vs. Growth (%) Since 2001")
plt.legend()
plt.show()


# In[29]:


def normalize_data(data_frame, features):
    """
    Function to normalize the specified features using StandardScaler.
    
    """
    scaler = StandardScaler()
    subset_features = data_frame[features]
    scaler.fit(subset_features)
    normalized_data = scaler.transform(subset_features)
    normalized_df = pd.DataFrame(normalized_data, columns=features)

    return normalized_df, scaler

norm, scaler = normalize_data(subx, ['2021', 'Growth'])


# In[36]:


def silhouette_score(xy, n):
    """
    Calculates silhouette score.

    Returns:
    Silhouette score.
    """
    kmeans = KMeans(n_clusters=n, init='k-means++', n_init=20)
    kmeans.fit(xy)
    labels = kmeans.labels_
    score = skmet.silhouette_score(xy, labels)
    return score


# In[37]:


for i in range(2, 15):
    score = silhouette_score(norm, i)
    print(f"The silhouette score for {i: 3d} is {score: 7.4f}")


# In[39]:


kmeans = KMeans(n_clusters=3, init='k-means++', n_init=20)
kmeans.fit(norm)
labels = kmeans.labels_
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
xkmeans, ykmeans = centroids[:, 0], centroids[:, 1]


# In[48]:


plt.figure(figsize=(8, 5))
sns.scatterplot(x=subx["2021"], y=subx["Growth"], hue=labels, palette="Set2", s=30)
plt.scatter(xkmeans, ykmeans, marker="x", c="black", s=50)
plt.xlabel("Forest area (sq. km) in 2021")
plt.ylabel("Growth (%) Since 2001")
plt.title("Forest area (sq. km) in 2021 vs. Growth (%) Since 2001")
plt.legend()
plt.show()


# In[58]:


world = dft.loc['1960':'2022', ['World']].reset_index().rename(columns={'index': 'Year', 'World': 'Forest area (sq. km)'})
world = world.apply(pd.to_numeric, errors='coerce')
world = world.dropna()
world.describe()


# In[59]:


plt.figure(figsize=(8, 5))
sns.lineplot(data=world, x='Year', y='Forest area (sq. km)', color = 'black')
plt.xlabel("Year")
plt.ylabel("Global Forest area (sq. km)")
plt.title("Global Forest area (sq. km) between 1960-2022")
plt.show()


# In[64]:


def exponential(t, n0, g):
    """Calculates exponential function with scale factor n0 and growth rate g."""
    # makes it easier to get a guess for initial parameters
    t = t - 1990
    f = n0 * np.exp(g*t)
    return f


# In[70]:


param, covar = opt.curve_fit(exponential, world["Year"], world["Forest area (sq. km)"], p0=(1e7, 0.1))
world["fit"] = exponential(world["Year"], *param)
plt.figure(figsize=(8, 5))
sns.lineplot(data=world, x="Year", y="Forest area (sq. km)", label="Forest area (sq. km)")
sns.lineplot(data=world, x="Year", y="fit", label="Exponential Fit")
plt.xlabel("Year")
plt.ylabel("Global Forest area (sq. km)")
plt.title("Global Forest area (sq. km) between 1990-2022")
plt.legend()
plt.show()


# In[68]:


years = np.arange(2021, 2032, 1)
predictions = exponential(years, *param)
confidence_range = errors.error_prop(years, exponential, param, covar)


# In[71]:


plt.figure(figsize=(8, 5))
sns.lineplot(x= world["Year"], y= world["Forest area (sq. km)"], label="Forest area (sq. km)")
sns.lineplot(x=years, y=predictions, label="Prediction", color='red')
plt.fill_between(years, predictions - confidence_range, predictions + confidence_range, color='green', alpha=0.4, label="Confidence Range")
plt.xlabel("Year")
plt.ylabel("Global Forest area (sq. km)")
plt.title("Global Forest area (sq. km) between 1990-2030")
plt.legend()
plt.show()


# In[ ]:




