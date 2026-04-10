import os

import pytest

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "anonymyzed_csv.csv")


@pytest.fixture(scope="session")
def csv_path():
    if not os.path.exists(CSV_PATH):
        pytest.skip("anonymyzed_csv.csv not found")
    return CSV_PATH
