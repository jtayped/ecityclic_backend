import pandas as pd
from config import DATA_FILE_PATH

# Load data
tramits_data = pd.read_csv(DATA_FILE_PATH)


def get_tramit_by_id(tramit_id):
    result = tramits_data[tramits_data["id"] == tramit_id]
    if not result.empty:
        return result.to_dict(orient="records")[0]
    return None
