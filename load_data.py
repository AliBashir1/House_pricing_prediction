import os
from six.moves import urllib
import pandas as pd
import tarfile

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_house_data(
        housing_url=HOUSING_URL,
        housing_path=HOUSING_PATH
) -> None:
    # Create the directory
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)

    # construct the path to the tgz file
    tgz_path = os.path.join(housing_path, "housing.tgz")
    # Download the file
    urllib.request.urlretrieve(housing_url, tgz_path)
    # open the tgz file
    housing_tgz = tarfile.open(tgz_path)
    # extract the file (unzip)
    housing_tgz.extractall(path=housing_path)
    # close the file
    housing_tgz.close()


def load_housing_data(
        housing_path: str = HOUSING_PATH
) -> pd.DataFrame:
    # construct the path to the csv file
    csv_path = os.path.join(housing_path, "housing.csv")
    # read the csv file dataframe.
    return pd.read_csv(csv_path)


if __name__ == "__main__":
    fetch_house_data()
    a = load_housing_data()
    print(a.head(1))