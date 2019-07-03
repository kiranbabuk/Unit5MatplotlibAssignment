#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Dependencies and Setup
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Importing requirement specific features
from scipy.stats import sem
import seaborn as sn
from math import trunc

# Hide warning messages in notebook
import warnings
warnings.filterwarnings('ignore')

# File to Load (Remember to Change These)
mouse_drug_data_to_load = "data/mouse_drug_data.csv"
clinical_trial_data_to_load = "data/clinicaltrial_data.csv"

# Read the Mouse and Drug Data and the Clinical Trial Data
mouse_drug_df = pd.read_csv(mouse_drug_data_to_load)
clinical_trial_df = pd.read_csv(clinical_trial_data_to_load)

# Combine the data into a single dataset
clinical_mouse = pd.merge(clinical_trial_df, mouse_drug_df, how='inner')

# Display the data table for preview
#mouse_drug_df -- checking sample
#clinical_trial_df -- checking sample
clinical_mouse.head()


# In[2]:


# Store the Mean Tumor Volume Data Grouped by Drug and Timepoint 
tumor_volume_df = clinical_mouse.loc[:,['Drug', 'Timepoint', 'Tumor Volume (mm3)']]
#tumor_volume_df.head() --checking sample

mean_sem_tv = tumor_volume_df.groupby(['Drug', 'Timepoint']).agg({"Tumor Volume (mm3)" :["mean", "sem"]})
mean_sem_tv


# In[3]:


# Create lists of the tumor volume means for each of the four drugs being converted to dataframe
cap_tvmean_list = mean_sem_tv.loc['Capomulin'].loc[:, 'Tumor Volume (mm3)'].loc[:,'mean'].tolist()
cef_tvmean_list = mean_sem_tv.loc['Ceftamin'].loc[:, 'Tumor Volume (mm3)'].loc[:,'mean'].tolist()
inf_tvmean_list = mean_sem_tv.loc['Infubinol'].loc[:, 'Tumor Volume (mm3)'].loc[:,'mean'].tolist()
ket_tvmean_list = mean_sem_tv.loc['Ketapril'].loc[:, 'Tumor Volume (mm3)'].loc[:,'mean'].tolist()
pro_tvmean_list = mean_sem_tv.loc['Propriva'].loc[:, 'Tumor Volume (mm3)'].loc[:,'mean'].tolist()
naf_tvmean_list = mean_sem_tv.loc['Naftisol'].loc[:, 'Tumor Volume (mm3)'].loc[:,'mean'].tolist()
plc_tvmean_list = mean_sem_tv.loc['Placebo'].loc[:, 'Tumor Volume (mm3)'].loc[:,'mean'].tolist()
ram_tvmean_list = mean_sem_tv.loc['Ramicane'].loc[:, 'Tumor Volume (mm3)'].loc[:,'mean'].tolist()
ste_tvmean_list = mean_sem_tv.loc['Stelasyn'].loc[:, 'Tumor Volume (mm3)'].loc[:,'mean'].tolist()
zon_tvmean_list = mean_sem_tv.loc['Zoniferol'].loc[:, 'Tumor Volume (mm3)'].loc[:,'mean'].tolist()
     

# Create lists of the tumor volume sems for each of the four drugs being converted to dataframe
cap_tvsem_list = mean_sem_tv.loc['Capomulin'].loc[:, 'Tumor Volume (mm3)'].loc[:,'sem'].tolist()
cef_tvsem_list = mean_sem_tv.loc['Ceftamin'].loc[:, 'Tumor Volume (mm3)'].loc[:,'sem'].tolist()
inf_tvsem_list = mean_sem_tv.loc['Infubinol'].loc[:, 'Tumor Volume (mm3)'].loc[:,'sem'].tolist()
ket_tvsem_list = mean_sem_tv.loc['Ketapril'].loc[:, 'Tumor Volume (mm3)'].loc[:,'sem'].tolist()
pro_tvsem_list = mean_sem_tv.loc['Propriva'].loc[:, 'Tumor Volume (mm3)'].loc[:,'sem'].tolist()
naf_tvsem_list = mean_sem_tv.loc['Naftisol'].loc[:, 'Tumor Volume (mm3)'].loc[:,'sem'].tolist()
plc_tvsem_list = mean_sem_tv.loc['Placebo'].loc[:, 'Tumor Volume (mm3)'].loc[:,'sem'].tolist()
ram_tvsem_list = mean_sem_tv.loc['Ramicane'].loc[:, 'Tumor Volume (mm3)'].loc[:,'sem'].tolist()
ste_tvsem_list = mean_sem_tv.loc['Stelasyn'].loc[:, 'Tumor Volume (mm3)'].loc[:,'sem'].tolist()
zon_tvsem_list = mean_sem_tv.loc['Zoniferol'].loc[:, 'Tumor Volume (mm3)'].loc[:,'sem'].tolist()

meanlist_df = pd.DataFrame({"Capomulin": cap_tvmean_list, 
                        "Ceftamin" : cef_tvmean_list,
                        "Infubinol": inf_tvmean_list,
                        "Ketapril": ket_tvmean_list,
                        "Naftisol": naf_tvmean_list,
                        "Placebo": plc_tvmean_list,
                        "Propriva": pro_tvmean_list,
                        "Ramicane": ram_tvmean_list,
                        "Stelasyn": ste_tvmean_list,
                        "Zoniferol": zon_tvmean_list})

semlist_df = pd.DataFrame({"Capomulin": cap_tvsem_list, 
                        "Ceftamin" : cef_tvsem_list,
                        "Infubinol": inf_tvsem_list,
                        "Ketapril": ket_tvsem_list,
                        "Naftisol": naf_tvsem_list,
                        "Placebo": plc_tvsem_list,
                        "Propriva": pro_tvsem_list,
                        "Ramicane": ram_tvsem_list,
                        "Stelasyn": ste_tvsem_list,
                        "Zoniferol": zon_tvsem_list})

meanlist_df


# In[4]:


# Scatter plot showing how tumor volume changes over time for each treatment
ax = plt.subplot(111)

# Set the x axis from 0 to 45 in increments of 5
x_axis = np.arange(0, 50, 5)

# Set the plot title and axes titles
plt.title("Tumor Response to Treatment")
plt.xlabel("Time (days)")
plt.ylabel("Tumor Volume (mm3)")

# Plot the 'mean' list vs. the established x axis with error 
ax.errorbar(x_axis, cap_tvmean_list, yerr=cap_tvsem_list, fmt="red", marker="o", label="Capomulin")
ax.errorbar(x_axis, inf_tvmean_list, yerr=inf_tvsem_list, fmt="blue", marker="^", label="Infubinol")
ax.errorbar(x_axis, ket_tvmean_list, yerr=ket_tvsem_list, fmt="green", marker="s", label="Ketapril")
ax.errorbar(x_axis, plc_tvmean_list, yerr=plc_tvsem_list, fmt="grey", marker="d", label="Placebo")

# Add the legend and gridlines
ax.legend(loc=2)

tick_locations = [value for value in x_axis]
ax.set_xticks(tick_locations, minor=False)
ax.grid('on', which='major', axis='both', linestyle='dotted', linewidth=0.5)

plt.xlim(0, max(x_axis)+2)
        
# Show the resulting scatter plot
plt.show()


# In[5]:


# Convert to DataFrame
Timepoint_response =  clinical_mouse.groupby(['Drug','Timepoint']).mean()[['Metastatic Sites']]

# Preview DataFrame
Timepoint_response.head()


# In[6]:


Metastatic_sites = pd.pivot_table(Timepoint_response, index='Timepoint', columns='Drug', values='Metastatic Sites', aggfunc = np.mean)
Metastatic_sites


# In[7]:


Metastatic = Metastatic_sites.index
plt.figure(figsize=(12,8))

plt.plot(Metastatic, Metastatic_sites['Capomulin'], marker ='o', linestyle='--', label="Capomulin")
plt.plot(Metastatic, Metastatic_sites['Ceftamin'], marker ='^', linestyle='--', label="Ceftamin")
plt.plot(Metastatic, Metastatic_sites['Infubinol'], marker ='s', linestyle='--', label="Infubinol")
plt.plot(Metastatic, Metastatic_sites['Ketapril'], marker ='p', linestyle='--', label="Ketapril")
plt.plot(Metastatic, Metastatic_sites['Naftisol'], marker ='+', linestyle='--', label="Naftisol")
plt.plot(Metastatic, Metastatic_sites['Placebo'], marker ='d', linestyle='--', label="Placebo")
plt.plot(Metastatic, Metastatic_sites['Propriva'], marker ='4', linestyle='--', label="Propriva")
plt.plot(Metastatic, Metastatic_sites['Ramicane'], marker ='*', linestyle='--', label="Ramicane")
plt.plot(Metastatic, Metastatic_sites['Stelasyn'], marker ='h', linestyle='--', label="Stelasyn")
plt.plot(Metastatic, Metastatic_sites['Zoniferol'], marker ='1', linestyle='--', label="Zoniferol")
plt.gca().set(xlabel = 'Treatment Duration (Days)', ylabel = 'Met. Sites',title = 'Metastatic Spread During Treatment',xlim = (0,max(Metastatic)))
plt.legend(loc = 'best', frameon=True)
plt.grid()
plt.show()


# In[8]:


mouse_response =  clinical_mouse.groupby(['Drug','Timepoint']).count()[['Mouse ID']]
mouse_response.head()


# In[9]:


Survival_pivot = pd.pivot_table(mouse_response, index='Timepoint', columns='Drug', values='Mouse ID', aggfunc = np.mean)
Survival_pivot


# In[10]:


Survival_percentage = Survival_pivot.copy()
Survival_percentage = round(Survival_percentage.apply(lambda c: c / c.max() * 100, axis=0),2)
Survival_percentage


# In[11]:


Survival_rate =  Survival_percentage.index

plt.figure(figsize=(12,8))

plt.plot(Survival_rate, Survival_percentage['Capomulin'], marker ='o', linestyle='--', label="Capomulin")
plt.plot(Survival_rate, Survival_percentage['Ceftamin'], marker ='^', linestyle='--', label="Ceftamin")
plt.plot(Survival_rate, Survival_percentage['Infubinol'], marker ='s', linestyle='--', label="Infubinol")
plt.plot(Survival_rate, Survival_percentage['Ketapril'], marker ='p', linestyle='--', label="Ketapril")
plt.plot(Survival_rate, Survival_percentage['Naftisol'], marker ='+', linestyle='--', label="Naftisol")
plt.plot(Survival_rate, Survival_percentage['Placebo'], marker ='d', linestyle='--', label="Placebo")
plt.plot(Survival_rate, Survival_percentage['Propriva'], marker ='4', linestyle='--', label="Propriva")
plt.plot(Survival_rate, Survival_percentage['Ramicane'], marker ='*', linestyle='--', label="Ramicane")
plt.plot(Survival_rate, Survival_percentage['Stelasyn'], marker ='h', linestyle='--', label="Stelasyn")
plt.plot(Survival_rate, Survival_percentage['Zoniferol'], marker ='1', linestyle='--', label="Zoniferol")
plt.gca().set(xlabel = 'Time (Days)', ylabel = 'Survival Rate(%)',title = 'Survival During Treatment',xlim = (0,max(Survival_rate)))
plt.legend(loc = 'best', frameon=True)
plt.grid()
plt.show()


# In[12]:


TumorChangePercent = (((meanlist_df.iloc[-1]-meanlist_df.iloc[0])/meanlist_df.iloc[0])*100).to_frame("% Change")
TumorChangePercent


# In[13]:


x=TumorChangePercent.index
y=TumorChangePercent['% Change']
plt.figure(figsize=(16,8))
colors = ['red' if _y >=0 else 'green' for _y in y]
ax = sn.barplot(x, y, palette=colors)
for n, (label, _y) in enumerate(zip(x, y)):
    if _y <= 0:
        ax.annotate(
            s='{:d}%'.format(trunc(_y)), xy=(n, -10), ha='center',va='center',
            xytext=(0,10), color='w', textcoords='offset points', weight='bold')
    else:
        ax.annotate(
            s='{:d}%'.format(trunc(_y)), xy=(n, 0), ha='center',va='center',
            xytext=(0,10), color='w', textcoords='offset points', weight='bold')  
plt.gca().set(xlabel='Drug', ylabel='% Tumor Volume Change', title='Tumor Change Over 45 Day Treatment')
plt.rc('grid', linestyle="--", color='black', linewidth=0.5)
plt.grid(True)
plt.show()

