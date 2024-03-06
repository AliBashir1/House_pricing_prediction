from load_data import (
    fetch_house_data,
    load_housing_data
)
from preprocessing_data import (
    preprocessing_data,
    create_median_income_cat,
    startified_shuffle
)

def main() -> None:
    # fetch_house_data()
    housing_df = load_housing_data()
    housing_df = create_median_income_cat(housing_df)
    train_set, test_set = startified_shuffle(housing_df)

    # create a copy of trained set
    housing_tr = train_set.copy()
    housing_tr_prepared = preprocessing_data(housing_tr)
    print(housing_tr_prepared)


if __name__ == "__main__":
    main()
