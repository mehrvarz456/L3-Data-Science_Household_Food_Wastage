# This Python 3 environment comes with many helpful analytics libraries installed
# Read and Analyse Data
#-----------------------------------------------------------------------------------------
import os
import pandas as pd


# Use raw string to handle backslashes properly
file_path = r"C:\Users\admin\Python Programming\WEEK_5\global_food_wastage_dataset.csv"


output_folder = r"C:\python_workspace\outputs"
os.makedirs(output_folder, exist_ok=True)  # Create the folder if it doesn't exist

# Check if file exists
if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    print(df.head(10))
    print("Data successfully loaded.")
    
#-------------------------------------------------------------------------------#
    df.head().to_csv(os.path.join(output_folder, "data_head.csv"), index=False)
    df.describe().T.to_csv(os.path.join(output_folder, "data_statistics.csv"))
#--------------------------------------------------------------------------------#
else:
    print(f"Error: The file '{file_path}' does not exist.")

# Any results you write to the current directory are saved as output.
from PIL import Image
import requests
from io import BytesIO


from PIL import Image

# Path to the local image
image_path = r"C:\Users\admin\Python Programming\WEEK_5\hfcqdun.PNG"

# Open the image directly from the local file path
img = Image.open(image_path)


### Missing Value and Unique Counts Analysis
# describe basic statistics of data
df.describe().T

df.isnull().sum()

import missingno as msno
msno.matrix(df)

df["Country"].value_counts()


def func(df):
    for column in df.columns :
        print(f"Unique counts for column: {column}")
        print(df[column].unique())
        print()
func(df)

import seaborn as sns
import matplotlib.pyplot as plt


import geopandas as gpd
import requests
import zipfile
import io


# URL of the shapefile
#------------------------------------------------------------------------------
#url = 'https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_us_state_20m.zip'
#url2 ='https://www.naturalearthdata.com/downloads/110m-cultural-vectors/' ## new version

# Download the shapefile as a zip file
#response = requests.get(url)
#with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
#    zip_ref.extractall('path_to_extract_to')  # specify the folder where to extract

# Read the shapefile
#shapefile_path = 'path_to_extract_to/cb_2022_us_state_20m.shp'
#world = gpd.read_file(shapefile_path)

# Check the data
#print(world.head())

#-----------------------------------------------------------------------------

import zipfile
import os

# Path to the manually downloaded ZIP
#zip_path = r'C:\shapefiles_zip\cb_2022_us_state_20m.zip'

#zip_path = r'c:\Users\admin\Downloads\cb_2022_us_state_20m'  # or wherever it is

zip_path = r'c:\Users\admin\Downloads\cb_2022_us_state_20m.zip'



import zipfile
import os
import geopandas as gpd

# Path to the manually downloaded ZIP
zip_path = r'c:\Users\admin\Downloads\cb_2022_us_state_20m.zip'
extract_path = r'C:\shapefiles'

# Extract
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# Read the shapefile
shapefile_path = os.path.join(extract_path, 'cb_2022_us_state_20m.shp')
world = gpd.read_file(shapefile_path)

# Show first few rows
print(world.head())
##-----------------------------------------------------------------------
#
##-----------------------------------------------------------------------
from IPython.display import display

display(world.head())
#pip install tabulate

from tabulate import tabulate

# Show as table in console
print(tabulate(world.head(), headers='keys', tablefmt='fancy_grid'))





#---------------------------------------------------------------
plt.figure(figsize=(14,8))
sns.barplot(y="Country", x="Total Waste (Tons)", data=df, palette="magma")
plt.title("Total Food Waste by Country", fontsize=16)
plt.xlabel("Total Waste (Tons)")  
plt.ylabel("Country") 
plt.show()


plt.figure(figsize=(12,6))
sns.lineplot(x="Year", y="Total Waste (Tons)", data=df, marker="o",ci=None)
plt.title("Total Food Waste Over the Years")
plt.xlabel("Year")
plt.ylabel("Total Waste (Tons)")
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
sns.lineplot(x="Year", y="Economic Loss (Million $)", data=df, marker="o", color="darkblue")

plt.title("Economic Loss Over the Years", fontsize=16, fontweight="bold")
plt.xlabel("Year", fontsize=14)
plt.ylabel("Economic Loss (Million $)", fontsize=14)
plt.grid(True)

plt.show()


plt.figure(figsize=(12, 6))  # We adjusted the graphics size

sns.lineplot(
    x="Year", 
    y="Household Waste (%)", 
    data=df, 
    marker="o", 
    color="green", 
    linewidth=2,  

)


plt.title("Household Waste Percentage Over the Years", fontsize=16, fontweight="bold", color="darkgreen")
plt.xlabel("Year", fontsize=14, fontweight="bold", color="gray")
plt.ylabel("Household Waste (%)", fontsize=14, fontweight="bold", color="gray")


plt.grid(True, linestyle="--", alpha=0.7, color="gray")
plt.show()

sns.histplot(df["Total Waste (Tons)"], bins=30, kde=True, color="g")
plt.title("Distribution of Total Food Waste")
plt.show()


plt.figure(figsize=(14,8)) 


sns.barplot(
    y="Country", 
    x="Household Waste (%)", 
    data=df, 
    palette="magma", 
)

plt.title("Household Food Waste Percentage by Country", fontsize=16, fontweight="bold")
plt.xlabel("Household Waste (%)", fontsize=14)
plt.ylabel("Country", fontsize=14)

# Show chart
plt.show()

plt.figure(figsize=(12, 6))  


sns.barplot(
    y="Food Category", 
    x="Total Waste (Tons)", 
    data=df, 
    palette="rocket", 
)

# Title and tags
plt.title("Total Waste by Food Category", fontsize=16, fontweight="bold", color="darkblue")
plt.xlabel("Total Waste (Tons)", fontsize=14, fontweight="bold", color="gray")
plt.ylabel("Food Category", fontsize=14, fontweight="bold", color="gray")

plt.grid(True)
plt.show()

### HeatMap
#==================================================================================
from sklearn.preprocessing import LabelEncoder

df_encoded = df.copy()

for col in df.select_dtypes(include=["object"]).columns:
    df_encoded[col] = LabelEncoder().fit_transform(df[col])


plt.figure(figsize=(10, 6))
sns.heatmap(df_encoded.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()


#==================================================================================

### Outlier Analysis
#-----------------------------------------------------------------------------------
import numpy as np

def detect_outliers(df, method="IQR", threshold=1.5):
    outlier_dict = {} 
    
    for col in df.select_dtypes(include=[np.number]):  
        if method == "IQR":
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        
        elif method == "z-score":
            mean = df[col].mean()
            std = df[col].std()
            z_scores = (df[col] - mean) / std
            outliers = df[np.abs(z_scores) > threshold]

        else:
            raise ValueError("Invalid method! You should use 'IQR' or 'z-score'.")

        outlier_dict[col] = outliers[col].values

        # Visualize outliers
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=df[col])
        plt.title(f"Outliers ({col})")
        plt.show()

    return outlier_dict

outliers = detect_outliers(df, method="IQR", threshold=1.5)

for key, values in outliers.items():
    print(f"Outliers in the {key} column: {values}")
#------------------------------------------------------------------------------
from sklearn.preprocessing import StandardScaler  # to fix scale issues

from sklearn.preprocessing import StandardScaler  # for rescaling
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def detect_outliers(df, method="IQR", threshold=1.5, scale_data=True):
    outlier_dict = {}

    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Scale the data if needed
    if scale_data:
        scaler = StandardScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df[numeric_cols]), columns=numeric_cols)
    else:
        df_scaled = df[numeric_cols]

    for col in numeric_cols:
        if method == "IQR":
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        
        elif method == "z-score":
            mean = df[col].mean()
            std = df[col].std()
            z_scores = (df[col] - mean) / std
            outliers = df[np.abs(z_scores) > threshold]

        else:
            raise ValueError("Invalid method! Use 'IQR' or 'z-score'.")

        outlier_dict[col] = outliers[col].values

    ## Combined boxplot
    plt.figure(figsize=(16, 9))
    sns.boxplot(data=df_scaled, orient="h", color="lightblue")  # <-- all dark blue initially
    plt.title("Outliers Across All Numeric Columns (Scaled)", fontsize=18)
    plt.xlabel("Scaled Value", fontsize=14)
    plt.ylabel("Feature", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    return outlier_dict

# Example call
outliers = detect_outliers(df, method="IQR", threshold=1.5, scale_data=True)

# Print detected outliers
for key, values in outliers.items():
    print(f"Outliers in {key} column: {values}")
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
#=================================================================================
def create_outlier_table(df, method="IQR", threshold=1.5):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    outlier_table = pd.DataFrame(False, index=df.index, columns=numeric_cols)  # initially False

    for col in numeric_cols:
        if method == "IQR":
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outlier_condition = (df[col] < lower_bound) | (df[col] > upper_bound)
        
        elif method == "z-score":
            mean = df[col].mean()
            std = df[col].std()
            z_scores = (df[col] - mean) / std
            outlier_condition = (np.abs(z_scores) > threshold)
        
        else:
            raise ValueError("Invalid method! Use 'IQR' or 'z-score'.")

        # Mark True if it's an outlier
        outlier_table[col] = outlier_condition

    return outlier_table

# Create the outlier table
outlier_table = create_outlier_table(df, method="IQR", threshold=1.5)

# View it
outlier_table.head(10)  # Show first 10 rows

#=============

# STEP 1: Create the outlier table
#outlier_table = create_outlier_table(df, method="IQR", threshold=1.5)

# STEP 2: Check if ANY outliers exist at all
#if outlier_table.any().any():
#    print("Outliers detected! Showing first few rows where outliers exist:")

    # Show only rows where there is at least one outlier
#    df_outliers_only = df[outlier_table.any(axis=1)]
#    print(df_outliers_only)
#else:
#    print("No strong outliers detected based on current method and threshold.")

#outlier_table = create_outlier_table(df, method="IQR", threshold=1.0)



#==================================================================================

#----------------------------------------------------------------------------------
# ML Modelling , Tuning and Evaluation
# -----------------------

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import xgboost as xgb
X = df.drop('Total Waste (Tons)', axis=1)
y = df['Total Waste (Tons)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# One Hot Encoding
X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)
# Standard Scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

import xgboost as xgb

xg_model = xgb.XGBRegressor(random_state=42)
xg_model.fit(X_train_scaled, y_train)

# Making predictions with test data of XGBoost model
y_pred_xg = xg_model.predict(X_test_scaled)

# MSE ve R² calculations
#mse_xg = mean_squared_error(y_test, y_pred_xg)
#r2_xg = r2_score(y_test, y_pred_xg)

#print(f"XGBoost MSE: {mse_xg}")
#print(f"XGBoost R²: {r2_xg}")

#rf_model = RandomForestRegressor(random_state=42)
#rf_model.fit(X_train_scaled, y_train)


#y_pred_rf = rf_model.predict(X_test_scaled)

#mse_rf = mean_squared_error(y_test, y_pred_rf)
#r2_rf = r2_score(y_test, y_pred_rf)

#print(f"Random Forest MSE: {mse_rf}")
#print(f"Random Forest R²: {r2_rf}")


#rf_model = RandomForestRegressor(random_state=42)
#rf_model.fit(X_train_scaled, y_train)


#y_pred_rf = rf_model.predict(X_test_scaled)

#mse_rf = mean_squared_error(y_test, y_pred_rf)
#r2_rf = r2_score(y_test, y_pred_rf)

#print(f"Random Forest MSE: {mse_rf}")
#print(f"Random Forest R²: {r2_rf}")

#----------------------------------------------------------------------
#metrics_file = os.path.join(output_folder, "model_metrics.txt")
#with open(metrics_file, "w") as f:
#    f.write(f"XGBoost MSE: {mse_xg}\n")
#    f.write(f"XGBoost R²: {r2_xg}\n")
#    f.write(f"Random Forest MSE: {mse_rf}\n")
#    f.write(f"Random Forest R²: {r2_rf}\n")
#-----------------------------------------------------------------------
#plt.savefig(os.path.join(output_folder, "total_food_waste_by_country.png"))
#-----------------------------------------------------------------------

import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# --- XGBoost Predictions ---
mse_xg = mean_squared_error(y_test, y_pred_xg)
r2_xg = r2_score(y_test, y_pred_xg)
print(f"XGBoost MSE: {mse_xg}")
print(f"XGBoost R²: {r2_xg}")

# --- Random Forest Model ---
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)

mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print(f"Random Forest MSE: {mse_rf}")
print(f"Random Forest R²: {r2_rf}")

# --- Save Metrics to File ---
metrics_file = os.path.join(output_folder, "model_metrics.txt")
with open(metrics_file, "w") as f:
    f.write(f"XGBoost MSE: {mse_xg}\n")
    f.write(f"XGBoost R²: {r2_xg}\n")
    f.write(f"Random Forest MSE: {mse_rf}\n")
    f.write(f"Random Forest R²: {r2_rf}\n")

# --- Save Plot (if one was created before) ---
plt.savefig(os.path.join(output_folder, "total_food_waste_by_country.png"))
###############################################################################
###############################################################################
import matplotlib.pyplot as plt
import numpy as np
import os

# Model names and corresponding metrics
models = ['XGBoost', 'Random Forest']
mse_values = [mse_xg, mse_rf]
r2_values = [r2_xg, r2_rf]

x = np.arange(len(models))  # label locations
width = 0.35  # width of the bars

fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx()  # Create a second y-axis for R²

# Plot MSE on the left y-axis
bar1 = ax1.bar(x - width/2, mse_values, width, label='MSE', color='red')
ax1.set_ylabel('Mean Squared Error')
ax1.set_ylim(0, max(mse_values) * 1.1)

# Plot R² on the right y-axis
bar2 = ax2.bar(x + width/2, r2_values, width, label='R² Score', color='green')
ax2.set_ylabel('R² Score')
ax2.set_ylim(0, 1.05)

# Add titles and labels
plt.title('Model Performance Comparison')
ax1.set_xticks(x)
ax1.set_xticklabels(models)

# Add value labels
for bar in bar1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height * 1.01, f'{height:,.0f}', ha='center', va='bottom')

for bar in bar2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{height:.3f}', ha='center', va='bottom')

# Create a combined legend
bars = bar1 + bar2
labels = [bar.get_label() for bar in bars]
ax1.legend(bars, ['MSE', 'R² Score'], loc='upper center')

# Save and show
plot_path = os.path.join(output_folder, 'model_comparison_metrics.png')
plt.tight_layout()
plt.savefig(plot_path)
plt.show()

#===========================================================================
#===========================================================================

import matplotlib.pyplot as plt
import numpy as np
import os

# Example data (use your own mse_xg, mse_rf, r2_xg, r2_rf)
mse_xg, mse_rf = 10217, 9528
r2_xg, r2_rf = 0.953, 0.956

# Model names and corresponding metrics
models = ['XGBoost', 'Random Forest']
mse_values = [mse_xg, mse_rf]
r2_values = [r2_xg, r2_rf]

x = np.arange(len(models))  # label locations
width = 0.35  # width of the bars

fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx()  # Create a second y-axis for R²

# Plot MSE on the left y-axis
bar1 = ax1.bar(x - width/2, mse_values, width, label='MSE', color='orange')
ax1.set_ylabel('Mean Squared Error')
ax1.set_ylim(0, max(mse_values) * 1.1)

# Plot R² on the right y-axis
bar2 = ax2.bar(x + width/2, r2_values, width, label='R² Score', color='lightgray')
ax2.set_ylabel('R² Score')
ax2.set_ylim(0, 1.05)

# Add titles and labels
plt.title('Model Performance Comparison')
ax1.set_xticks(x)
ax1.set_xticklabels(models)

# Add value labels
for bar in bar1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height * 1.01, f'{height:,.0f}', ha='center', va='bottom')

for bar in bar2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{height:.3f}', ha='center', va='bottom')

# Create a CLEANER combined legend for each model separately
from matplotlib.patches import Patch

legend_elements = [
    Patch(facecolor='orange', label='MSE (XGBoost & RF)'), 
    Patch(facecolor='lightgray', label='R² Score (XGBoost & RF)')
]

ax1.legend(handles=legend_elements, loc='upper center')

# Save and show
output_folder = "."  # example
plot_path = os.path.join(output_folder, 'model_comparison_metrics.png')
plt.tight_layout()
plt.savefig(plot_path)
plt.show()
