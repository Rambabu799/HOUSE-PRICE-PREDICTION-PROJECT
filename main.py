'''
In this file I am going to load the data and develop XGBoost Regressor
Dataset : data.csv  (King County, WA — House Sales, 4600 rows)

Dropped Columns (irrelevant identifiers):
    date, street, statezip, country

City Encodings (alphabetical index):
    Algona:0            Auburn:1            Beaux Arts Village:2
    Bellevue:3          Black Diamond:4     Bothell:5
    Burien:6            Carnation:7         Clyde Hill:8
    Covington:9         Des Moines:10       Duvall:11
    Enumclaw:12         Fall City:13        Federal Way:14
    Inglewood-Finn Hill:15  Issaquah:16     Kenmore:17
    Kent:18             Kirkland:19         Lake Forest Park:20
    Maple Valley:21     Medina:22           Mercer Island:23
    Milton:24           Newcastle:25        Normandy Park:26
    North Bend:27       Pacific:28          Preston:29
    Ravensdale:30       Redmond:31          Renton:32
    Sammamish:33        SeaTac:34           Seattle:35
    Shoreline:36        Skykomish:37        Snoqualmie:38
    Snoqualmie Pass:39  Tukwila:40          Vashon:41
    Woodinville:42      Yarrow Point:43

Waterfront : No:0, Yes:1
View       : None:0 → Excellent:4
Condition  : Poor:1 → Excellent:5
Renovated  : No:0, Yes:1  (engineered from yr_renovated)
'''

import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb


class HousePriceModel:

    def __init__(self, path):

        # ── Load Dataset ──────────────────────────────────────────
        self.df = pd.read_csv(path)

        # ── Drop Irrelevant Columns ───────────────────────────────
        self.df.drop(columns=['date', 'street', 'statezip', 'country'],
                     inplace=True)

        # ── Data Cleaning ─────────────────────────────────────────

        # Remove rows where price is zero (data errors)
        self.df = self.df[self.df['price'] > 0]

        # Remove top 1% price outliers
        q99 = self.df['price'].quantile(0.99)
        self.df = self.df[self.df['price'] <= q99]

        # Remove unrealistic bedroom counts
        self.df = self.df[self.df['bedrooms'] < 20]

        # ── City Encoding (alphabetical → integer) ────────────────
        city_list = sorted(self.df['city'].unique())
        city_encoding = {city: idx for idx, city in enumerate(city_list)}
        self.df['city'] = self.df['city'].map(city_encoding)

        # ── Feature Engineering ───────────────────────────────────

        # Age of the house
        self.df['house_age'] = 2024 - self.df['yr_built']

        # Whether the house was ever renovated
        self.df['renovated'] = (self.df['yr_renovated'] > 0).astype(int)

        # Years since last renovation (or house age if never renovated)
        self.df['years_since_renovated'] = self.df.apply(
            lambda r: 2024 - r['yr_renovated']
            if r['yr_renovated'] > 0 else r['house_age'],
            axis=1
        )

        # Total usable square footage
        self.df['total_sqft'] = self.df['sqft_living'] + self.df['sqft_basement']

        # Bedroom-to-bathroom ratio (comfort indicator)
        self.df['bed_bath_ratio'] = self.df['bedrooms'] / (self.df['bathrooms'] + 1)

        # Living area relative to lot size
        self.df['living_to_lot_ratio'] = (
            self.df['sqft_living'] / (self.df['sqft_lot'] + 1)
        )

        # Combined luxury score (waterfront is premium)
        self.df['luxury_score'] = self.df['waterfront'] * 5 + self.df['view']

        # Drop original year columns (replaced by engineered features)
        self.df.drop(columns=['yr_built', 'yr_renovated'], inplace=True)

        # ── Features and Target ───────────────────────────────────

        # Move 'price' to last column so iloc[:, :-1] / iloc[:, -1] works
        cols = [c for c in self.df.columns if c != 'price'] + ['price']
        self.df = self.df[cols]

        X = self.df.iloc[:, :-1]
        y = self.df.iloc[:, -1]

        # ── Split Data ────────────────────────────────────────────
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

    # ── Training ──────────────────────────────────────────────────
    def train(self):

        self.model = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )

        # Train Model
        self.model.fit(self.X_train, self.y_train)

        # Training Prediction
        train_pred = self.model.predict(self.X_train)

        print("\nTraining R2 Score :",
              r2_score(self.y_train, train_pred))

        print("Training RMSE     : $",
              round(np.sqrt(mean_squared_error(self.y_train, train_pred)), 2))

    # ── Testing ───────────────────────────────────────────────────
    def test(self):

        test_pred = self.model.predict(self.X_test)
        print("\nTesting R2 Score  :",
              r2_score(self.y_test, test_pred))

        print("Testing RMSE      : $",
              round(np.sqrt(mean_squared_error(self.y_test, test_pred)), 2))

    # ── Custom Prediction ─────────────────────────────────────────
    def predict(self):

        '''
        Feature Order:
            bedrooms, bathrooms, sqft_living, sqft_lot, floors,
            waterfront, view, condition, sqft_above, sqft_basement,
            city, house_age, renovated, years_since_renovated,
            total_sqft, bed_bath_ratio, living_to_lot_ratio, luxury_score

        Sample House:
            3 bed | 2 bath | 1800 sqft living | 7500 sqft lot
            1 floor | No waterfront | No view | Condition = 3
            Built 2005 | Never renovated | Seattle (city=35)
        '''

        sample = [[
            3,        # bedrooms
            2,        # bathrooms
            1800,     # sqft_living
            7500,     # sqft_lot
            1,        # floors
            0,        # waterfront          (No=0, Yes=1)
            0,        # view                (0 → 4)
            3,        # condition           (1 → 5)
            1440,     # sqft_above
            360,      # sqft_basement
            35,       # city                (Seattle=35)
            19,       # house_age           (2024 - 2005)
            0,        # renovated           (No=0, Yes=1)
            19,       # years_since_renovated
            2160,     # total_sqft          (sqft_living + sqft_basement)
            1.0,      # bed_bath_ratio      (3 / (2+1))
            0.24,     # living_to_lot_ratio (1800 / 7501)
            0,        # luxury_score        (waterfront*5 + view)
        ]]

        result = self.model.predict(sample)

        print("\nPredicted Price : $", round(result[0], 2))

    # ── Save Model ────────────────────────────────────────────────
    def save_model(self):

        with open("House_Price_Model.pkl", "wb") as f:
            pickle.dump(self.model, f)

        print("\nModel Saved Successfully")


# ── Main ──────────────────────────────────────────────────────────
obj = HousePriceModel("data.csv")

obj.train()

obj.test()

obj.predict()

obj.save_model()