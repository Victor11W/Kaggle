import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
from ydata_profiling import ProfileReport
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import  cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
import lightgbm as lgb
import optuna
import os


# Désactive les erreurs Ray parasites
os.environ["RAY_IGNORE_UNHANDLED_ERRORS"] = "1"

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
test_ids = test['Id']

y= train["SalePrice"]
train = train.drop(columns=["SalePrice","Id"])
test = test.drop(columns=["Id"])
y= train["SalePrice"]
train = train.drop(columns=["SalePrice","Id"])
test = test.drop(columns=["Id"])

# MSZoning - replace NaN with mode (RL)
train["MSZoning"] = train["MSZoning"].replace(np.nan, "RL")
test["MSZoning"] = test["MSZoning"].replace(np.nan, "RL")


# LotFrontage - replace NaN with median of neighborhood + log transform
train["LotFrontage"] = train.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median())
)
test["LotFrontage"] = test.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median())
)
train["LotFrontage"] = np.log1p(train["LotFrontage"])  # log transform for skew
test["LotFrontage"] = np.log1p(test["LotFrontage"])  # log transform for skew

# Age_Built and Age_RemodAdd - create new features
train["Age_Built"] = train["YrSold"] - train["YearBuilt"]
train["Age_RemodAdd"] = train["YrSold"] - train["YearRemodAdd"]
test["Age_Built"] = test["YrSold"] - test["YearBuilt"]
test["Age_RemodAdd"] = test["YrSold"] - test["YearRemodAdd"]

# LotArea - use log transformation to reduce skewness
train["LotArea"] = np.log1p(train["LotArea"])
test["LotArea"] = np.log1p(test["LotArea"])



# MasVnrType - replace NaN with None
train["MasVnrType"] = train["MasVnrType"].replace(np.nan, "None")
test["MasVnrType"] = test["MasVnrType"].replace(np.nan, "None")

# MasVnrArea - replace NaN with 0 + log transform
train["MasVnrArea"] = train["MasVnrArea"].replace(np.nan, 0)
test["MasVnrArea"] = test["MasVnrArea"].replace(np.nan, 0)
train["MasVnrArea"] = np.log1p(train["MasVnrArea"])  # log transform for skew
test["MasVnrArea"] = np.log1p(test["MasVnrArea"])  # log transform for skew

# ExterCond - mapping
mapping = {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
train["ExterCond"] = train["ExterCond"].map(mapping)
test["ExterCond"] = test["ExterCond"].map(mapping)
# ExterQual - mapping
mapping = {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
train["ExterQual"] = train["ExterQual"].map(mapping)
test["ExterQual"] = test["ExterQual"].map(mapping)

#BstmQual - replace NaN with None + mapping
train["BsmtQual"] = train["BsmtQual"].replace(np.nan, "None")
test["BsmtQual"] = test["BsmtQual"].replace(np.nan, "None")
mapping = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
train["BsmtQual"] = train["BsmtQual"].map(mapping)
test["BsmtQual"] = test["BsmtQual"].map(mapping)

# BsmtCOnd - replace NaN with None + mapping
train["BsmtCond"] = train["BsmtCond"].replace(np.nan, "None")
test["BsmtCond"] = test["BsmtCond"].replace(np.nan, "None")
mapping = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
train["BsmtCond"] = train["BsmtCond"].map(mapping)
test["BsmtCond"] = test["BsmtCond"].map(mapping)

# BsmtHalfBath - replace NaN with 0
train["BsmtHalfBath"] = train["BsmtHalfBath"].replace(np.nan, 0)
test["BsmtHalfBath"] = test["BsmtHalfBath"].replace(np.nan, 0)

# BsmtFullBath - replace NaN with 0
train["BsmtFullBath"] = train["BsmtFullBath"].replace(np.nan, 0)
test["BsmtFullBath"] = test["BsmtFullBath"].replace(np.nan, 0)

# BsmtExposure - replace NaN with None + mapping
train["BsmtExposure"] = train["BsmtExposure"].replace(np.nan, "None")
test["BsmtExposure"] = test["BsmtExposure"].replace(np.nan, "None")
mapping = {"None": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4}
train["BsmtExposure"] = train["BsmtExposure"].map(mapping)
test["BsmtExposure"] = test["BsmtExposure"].map(mapping)

#BsmtfinType1 - replace NaN with None + mapping
train["BsmtFinType1"] = train["BsmtFinType1"].replace(np.nan, "None")
test["BsmtFinType1"] = test["BsmtFinType1"].replace(np.nan, "None")
mapping = {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6}
train["BsmtFinType1"] = train["BsmtFinType1"].map(mapping)
test["BsmtFinType1"] = test["BsmtFinType1"].map(mapping)

# BsmtFinSF2 - replace NaN with 0 + log transform + binary indicator
train["BsmtFinSF2"] = train["BsmtFinSF2"].replace(np.nan, 0)
test["BsmtFinSF2"] = test["BsmtFinSF2"].replace(np.nan, 0)
train["HasBsmtFinSF2"] = (train["BsmtFinSF2"] > 0).astype(int)
test["HasBsmtFinSF2"] = (test["BsmtFinSF2"] > 0).astype(int)
train["BsmtFinSF2"] = np.log1p(train["BsmtFinSF2"])  # log transform for skew
test["BsmtFinSF2"] = np.log1p(test["BsmtFinSF2"])  # log transform for skew

# BsmtFinSF1 - replace NaN with 0 + log transform + binary indicator
train["BsmtFinSF1"] = train["BsmtFinSF1"].replace(np.nan, 0)
test["BsmtFinSF1"] = test["BsmtFinSF1"].replace(np.nan, 0)
train["HasBsmtFinSF1"] = (train["BsmtFinSF1"] > 0).astype(int)
test["HasBsmtFinSF1"] = (test["BsmtFinSF1"] > 0).astype(int)
train["BsmtFinSF1"] = np.log1p(train["BsmtFinSF1"])  # log transform for skew
test["BsmtFinSF1"] = np.log1p(test["BsmtFinSF1"])  # log transform for skew

# BsmtFinType2 - replace NaN with None + mapping
train["BsmtFinType2"] = train["BsmtFinType2"].replace(np.nan, "None")
test["BsmtFinType2"] = test["BsmtFinType2"].replace(np.nan, "None")
mapping = {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6}
train["BsmtFinType2"] = train["BsmtFinType2"].map(mapping)
test["BsmtFinType2"] = test["BsmtFinType2"].map(mapping)

# BsmtUnfSF - replace NaN with 0 + log transform
train["BsmtUnfSF"] = train["BsmtUnfSF"].replace(np.nan, 0)
test["BsmtUnfSF"] = test["BsmtUnfSF"].replace(np.nan, 0)
train["BsmtUnfSF"] = np.log1p(train["BsmtUnfSF"])  # log transform for skew
test["BsmtUnfSF"] = np.log1p(test["BsmtUnfSF"])  # log transform for skew

# TotalBsmtSF - use log transformation to reduce skewness
# replace NaN with 0 (no basement)
train["TotalBsmtSF"] = train["TotalBsmtSF"].replace(np.nan, 0)
test["TotalBsmtSF"] = test["TotalBsmtSF"].replace(np.nan, 0)
train["TotalBsmtSF"] = np.log1p(train["TotalBsmtSF"])
test["TotalBsmtSF"] = np.log1p(test["TotalBsmtSF"])

# HeatingQC 
mapping = {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
train["HeatingQC_num"] = train["HeatingQC"].map(mapping)
test["HeatingQC_num"] = test["HeatingQC"].map(mapping)

# Electrical - replace NaN with mode (SBrkr)
train["Electrical"] = train["Electrical"].replace(np.nan, "SBrkr")
test["Electrical"] = test["Electrical"].replace(np.nan, "SBrkr")

# Exterior1st - replace NaN with mode (VinylSd)
train["Exterior1st"] = train["Exterior1st"].replace(np.nan, "VinylSd")
test["Exterior1st"] = test["Exterior1st"].replace(np.nan, "VinylSd")

# Exterior2nd - replace NaN with mode (VinylSd)
train["Exterior2nd"] = train["Exterior2nd"].replace(np.nan, "VinylSd")
test["Exterior2nd"] = test["Exterior2nd"].replace(np.nan, "VinylSd")

# First Floor SF - use log transformation to reduce skewness
train["1stFlrSF"] = np.log1p(train["1stFlrSF"])
test["1stFlrSF"] = np.log1p(test["1stFlrSF"])

# 2nd Flr SF - create binary indicator + log transform
train["2ndFlrSF"] = np.log1p(train["2ndFlrSF"])  # log transform for skew
test["2ndFlrSF"] = np.log1p(test["2ndFlrSF"])  # log transform for skew

# Low Qualfin SF - binomial indicator
train["HasLowQualFinSF"] = (train["LowQualFinSF"] > 0).astype(int)
test["HasLowQualFinSF"] = (test["LowQualFinSF"] > 0).astype(int)

# GrLivArea - use log transformation to reduce skewness
train["GrLivArea"] = np.log1p(train["GrLivArea"])
test["GrLivArea"] = np.log1p(test["GrLivArea"])

# Kitchen Qual - replace NaN with mode (TA) + mapping
mapping = {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
train["KitchenQual"] = train["KitchenQual"].replace(np.nan, "TA")
test["KitchenQual"] = test["KitchenQual"].replace(np.nan, "TA")
train["KitchenQual"] = train["KitchenQual"].map(mapping)
test["KitchenQual"] = test["KitchenQual"].map(mapping)

#FireplaceQu - replace NaN with None + mapping
mapping = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}  
train["FireplaceQu"] = train["FireplaceQu"].replace(np.nan, "None")
test["FireplaceQu"] = test["FireplaceQu"].replace(np.nan, "None")
train["FireplaceQu"] = train["FireplaceQu"].map(mapping)
test["FireplaceQu"] = test["FireplaceQu"].map(mapping)

# GarageYrBlt - create new feature GarageAge + replace NaN with maxvalue 
max_year = max(train["YearBuilt"].max(), train["YearRemodAdd"].max(), train["YrSold"].max()) + 1
train["GarageYrBlt"] = train["GarageYrBlt"].replace(np.nan, max_year)
test["GarageYrBlt"] = test["GarageYrBlt"].replace(np.nan, max_year)
train["GarageAge"] = train["YrSold"] - train["GarageYrBlt"]
test["GarageAge"] = test["YrSold"] - test["GarageYrBlt"]

# GarageFinish - replace NaN with None + mapping
train["GarageFinish"] = train["GarageFinish"].replace(np.nan, "None")
test["GarageFinish"] = test["GarageFinish"].replace(np.nan, "None")
mapping = {"Fin": 1, "RFn": 2, "Unf": 3, "None": 0}
train["GarageFinish"] = train["GarageFinish"].map(mapping)
test["GarageFinish"] = test["GarageFinish"].map(mapping)

# GarageCond - mapping + replace NaN with None
mapping = {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5, "None": 0}
train["GarageCond"] = train["GarageCond"].replace(np.nan, "None")
test["GarageCond"] = test["GarageCond"].replace(np.nan, "None")
train["GarageCond"] = train["GarageCond"].map(mapping)
test["GarageCond"] = test["GarageCond"].map(mapping)

# GarageCars - create binary indicator and log transform + replace NaN with 0
train["GarageCars"] = train["GarageCars"].fillna(0)
test["GarageCars"] = test["GarageCars"].fillna(0)


# GarageType - replace NaN with None
train["GarageType"] = train["GarageType"].replace(np.nan, "None")
test["GarageType"] = test["GarageType"].replace(np.nan, "None")

# GarageQual - replace NaN with None + mapping
mapping = {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5, "None": 0}
train["GarageQual"] = train["GarageQual"].replace(np.nan, "None")
test["GarageQual"] = test["GarageQual"].replace(np.nan, "None")
train["GarageQual"] = train["GarageQual"].map(mapping)
test["GarageQual"] = test["GarageQual"].map(mapping)

#Same with WoodDeckSF
train["WoodDeckSF_log"] = np.log1p(train["WoodDeckSF"])  # log transform for skew
test["WoodDeckSF_log"] = np.log1p(test["WoodDeckSF"])

# Optionally keep raw or transform it
train["OpenPorchSF_log"] = np.log1p(train["OpenPorchSF"])  # log transform for skew
test["OpenPorchSF_log"] = np.log1p(test["OpenPorchSF"])

train["HasScreenPorch"] = (train["ScreenPorch"] > 0).astype(int)
train["ScreenPorchLog"] = np.log1p(train["ScreenPorch"])
test["HasScreenPorch"] = (test["ScreenPorch"] > 0).astype(int)
test["ScreenPorchLog"] = np.log1p(test["ScreenPorch"])

train["GotPool"] = train["PoolQC"].notnull().astype(int)
test["GotPool"] = test["PoolQC"].notnull().astype(int)


# Fence - replace NaN with None + mapping with quality and good wood
# Fence - mapping with quality and good wood
# Fence_wo: 2 if GdWo, 1 if MnWw, else 0
# Fence_Prv: 2 if GdPrv, 1 if MnPrv, else 0
def fence_wo(val):
    if val == "GdWo":
        return 2
    elif val == "MnWw":
        return 1
    else:
        return 0
def fence_prv(val):
    if val == "GdPrv":
        return 2
    elif val == "MnPrv":
        return 1
    else:
        return 0
train["Fence_wo"] = train["Fence"].apply(fence_wo)
test["Fence_wo"] = test["Fence"].apply(fence_wo)
train["Fence_Prv"] = train["Fence"].apply(fence_prv)
test["Fence_Prv"] = test["Fence"].apply(fence_prv)

mapping = {"GdWo": 2, "MnPrv": 1, "GdPrv": 2, "MnWw": 1, "None": 0}
train["Fence"] = train["Fence"].map(mapping)
test["Fence"] = test["Fence"].map(mapping)

train["MoSold_sin"] = np.sin(2 * np.pi * train["MoSold"] / 12)
train["MoSold_cos"] = np.cos(2 * np.pi * train["MoSold"] / 12)
test["MoSold_sin"] = np.sin(2 * np.pi * test["MoSold"] / 12)
test["MoSold_cos"] = np.cos(2 * np.pi * test["MoSold"] / 12)





#Feature engineering 
train["TotalArea"] = train["GrLivArea"] + train["TotalBsmtSF"] + train["GarageArea"]
train["BathPerRoom"] = (train["FullBath"] + train["HalfBath"]) / (train["TotRmsAbvGrd"] + 1)

test["TotalArea"] = test["GrLivArea"] + test["TotalBsmtSF"] + test["GarageArea"]
test["BathPerRoom"] = (test["FullBath"] + test["HalfBath"]) / (test["TotRmsAbvGrd"] + 1)

train["OverallQual_GrLiv"] = train["OverallQual"] * train["GrLivArea"]
train["Qual_x_Bath"] = train["OverallQual"] * (train["FullBath"] + train["HalfBath"])

test["OverallQual_GrLiv"] = test["OverallQual"] * test["GrLivArea"]
test["Qual_x_Bath"] = test["OverallQual"] * (test["FullBath"] + test["HalfBath"])

neighborhood_means = pd.concat([train, y], axis=1).groupby("Neighborhood")["SalePrice"].mean()
train["Neighborhood_mean"] = train["Neighborhood"].map(neighborhood_means)
test["Neighborhood_mean"] = test["Neighborhood"].map(neighborhood_means)


# Sale type - replace NaN with mode (WD)
train["SaleType"] = train["SaleType"].replace(np.nan, "WD")
test["SaleType"] = test["SaleType"].replace(np.nan, "WD")

train = train.drop(columns=["BsmtFinType2","ExterCond","PoolArea","OpenPorchSF","WoodDeckSF","LowQualFinSF","YearRemodAdd","YearBuilt","GarageYrBlt","PoolQC","Fence","Functional","GarageArea","EnclosedPorch","3SsnPorch","ScreenPorch","MiscFeature","MiscVal","RoofMatl","Condition2","Alley", "Street", "Utilities", "MiscFeature"])
test = test.drop(columns=["BsmtFinType2","ExterCond","PoolArea","OpenPorchSF","WoodDeckSF","LowQualFinSF","YearRemodAdd","YearBuilt","GarageYrBlt","PoolQC","Fence","Functional","GarageArea","EnclosedPorch","3SsnPorch","ScreenPorch","MiscFeature","MiscVal","RoofMatl","Condition2","Alley", "Street", "Utilities", "MiscFeature"])

print(test.columns)



#je garde à contre coeur : LandSlope, Screeporch PavedDrive?
# truc potentiellement intéréssant : MiscVal kitchenAbove
# truc ou je me suis permis des dinguerie : PoolQC (et donc poolarea) 
num_cols = ["Fence_wo","Fence_Prv","GarageQual","KitchenQual","BsmtFinType1","BsmtFinSF2","MasVnrArea","ExterQual","BsmtQual","BsmtCond","FireplaceQu","GarageFinish","BsmtExposure","BsmtFinSF1","BsmtUnfSF","TotalBsmtSF","GarageCond","HeatingQC_num","1stFlrSF","2ndFlrSF","GrLivArea","BsmtFullBath","BsmtHalfBath","FullBath","HalfBath","BedroomAbvGr","KitchenAbvGr","TotRmsAbvGrd","Fireplaces","GarageAge","GarageArea_log","WoodDeckSF_log","HasScreenPorch","MoSold_sin","MoSold_cos","Age_Built","Age_RemodAdd","LotFrontage","LotArea","OverallQual","OverallCond"]
cat_cols = ["MasVnrType","Foundation","HasBsmtFinSF1","Heating","CentralAir","Electrical","HasBsmtFinSF2","HasLowQualFinSF","GarageType","PavedDrive","HasScreenPorch","GotPool","YrSold","SaleType","SaleCondition","Exterior1st","Exterior2nd","RoofStyle","MSSubClass","HouseStyle","Neighborhood","BldgType","MSSubClass","LotShape","LandContour","LotConfig","LandSlope","Condition1"]

preprocessor = ColumnTransformer(
    transformers=[
        # One-hot categorical features
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        # Pass-through numerical features
        ("num", StandardScaler(), num_cols),
    ],
    remainder="drop"  # drop unused raw columns like Name, Ticket, Cabin, etc.
)


X_train = train
y_train = y
X_test = test

test_data = X_test
train_data = X_train
train_data['SalePrice'] = y_train

label = "SalePrice"

train_ag = TabularDataset(train_data)

test_data = X_test.copy()
train_data = X_train.copy()
train_data['SalePrice'] = y_train

label = "SalePrice"

# Création du dataset AG
train_ag = TabularDataset(train_data)

# Création et entraînement du prédicteur
predictor = TabularPredictor(
        label=label,
        eval_metric="root_mean_squared_error",
        verbosity=2
    ).fit(
        train_data=train_ag,
        time_limit=2500,   # temps max en secondes
        presets="best_quality",  # meilleure qualité
        num_cpus=1,       # pas de Ray
        num_gpus=0
    )

# Leaderboard complet
lb = predictor.leaderboard(silent=False)

# On garde les 3 meilleurs modèles
top3_models = lb.sort_values("score_val").head(3)["model"].tolist()

# Test dataset en format AG
test_ag = TabularDataset(test_data)

# Génération des prédictions pour chaque modèle du top 3
for rank, model_name in enumerate(top3_models, start=1):
    preds = predictor.predict(test_ag, model=model_name)
    
    # Sauvegarde avec nom dynamique
    filename = f"submission_{rank}_{model_name}.csv"
    submission = pd.DataFrame({
        "Id": test_ids,
        "SalePrice": preds
    })
    submission.to_csv(filename, index=False)
    print(f"Fichier {filename} généré !")