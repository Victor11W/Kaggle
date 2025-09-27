import re
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Builds Titanic features deterministically with group-based imputations."""
    def __init__(self):
        # Will be learned on fit
        self.age_group_median_ = None
        self.age_global_median_ = None
        self.fare_median_ = None

    def _extract_title(self, name: str) -> str:
        # Extract title using regex and map rare titles to 'Rare'
        m = re.search(r',\s*([^\.]*)\.', name if isinstance(name, str) else "")
        if not m:
            return "Rare"
        t = m.group(1).strip()
        return t if t in ['Mr', 'Mrs', 'Miss', 'Master'] else 'Rare'

    def _cabin_initial(self, cabin: str) -> str:
        # Map first letter; treat NA or 'T' as 'U' (Unknown)
        if not isinstance(cabin, str) or len(cabin) == 0:
            return "U"
        c0 = cabin[0]
        return "U" if c0 == "T" else c0

    def fit(self, X, y=None):
        X = X.copy()

        # Compute fare median on training data
        self.fare_median_ = X["Fare"].median(skipna=True)

        # Prepare columns required for group imputation
        sex = X["Sex"].astype(str)
        pclass = X["Pclass"].astype(int)

        # Compute Age medians per (Sex, Pclass)
        grp = X.assign(Sex=sex, Pclass=pclass).groupby(["Sex", "Pclass"])["Age"]
        self.age_group_median_ = grp.median()
        self.age_global_median_ = X["Age"].median()

        return self

    def transform(self, X):
        X = X.copy()

        # Impute Fare with train median
        X["Fare"] = X["Fare"].fillna(self.fare_median_)

        # Impute Age by (Sex, Pclass) median; fallback to global median
        key = pd.MultiIndex.from_frame(
            pd.DataFrame({"Sex": X["Sex"].astype(str), "Pclass": X["Pclass"].astype(int)})
        )
        age_fill = self.age_group_median_.reindex(key).values
        # Where group median is NaN, fallback to global median
        age_fill = np.where(np.isnan(age_fill), self.age_global_median_, age_fill)
        # Only fill NaNs
        X["Age"] = X["Age"].fillna(pd.Series(age_fill, index=X.index))

        # Derived features: Age bins
        bins = [0, 12, 19, 35, 55, np.inf]
        labels = ["child", "teen", "young", "adult", "senior"]
        X["AgeBin"] = pd.cut(X["Age"], bins=bins, labels=labels, right=True)

        
        # Family features
        X["FamilySize"] = X["SibSp"].fillna(0) + X["Parch"].fillna(0) + 1
        X["isAlone"] = (X["FamilySize"] == 1).astype(int)

        # Fare log
        X["Farelog"] = np.log1p(X["Fare"].astype(float))

        # Title from Name
        X["Title"] = X["Name"].apply(self._extract_title)

        # Cabin initial (collapse 'T' into 'U')
        X["cabinInitial"] = X["Cabin"].apply(self._cabin_initial)

        # Return with all original columns still present; downstream will select/encode
        return X
