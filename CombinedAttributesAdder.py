import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

room_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True) -> None:
        # Constructor method that initializes the transformer
        # It takes an optional boolean parameter add_bedrooms_per_room, defaulting to True
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X: pd.DataFrame, y: None = None):
        # Fit method - doesn't do anything as there is no fitting required for this transformer
        return self

    def transform(self, X: pd.DataFrame, y: None = None):
        # Transform method - adds additional attributes to the dataset
        # Extracting relevant columns from the input DataFrame X
        rooms_per_household = X[:, room_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]

        # Checking if the add_bedrooms_per_room flag is True
        if self.add_bedrooms_per_room:
            bedroom_per_room = X[:, bedrooms_ix] / X[:, households_ix]
            # Concatenating the new attributes with the original dataset
            return np.c_[X, rooms_per_household, population_per_household, bedroom_per_room]
        else:
            # Concatenating the new attributes without adding bedrooms_per_room
            return np.c_[X, rooms_per_household, population_per_household]



