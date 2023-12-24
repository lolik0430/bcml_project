import pandas as pd
from owlracle_python import OwlracleAPI


class GasDataCollector:
    def __init__(self, api_key, user_agent="User-Agent"):
        self.api_key = api_key
        self.user_agent = user_agent
        self.owlracle = OwlracleAPI(self.api_key, useragent=self.user_agent)

    def collect_data(self):
        result = self.owlracle.get_gas_history(timeframe=60, candles=1000)
        flattened_data = []

        for entry in result["candles"]:
            try:
                flattened_entry = {
                    'open': entry['gasPrice']['open'],
                    'close': entry['gasPrice']['close'],
                    'low': entry['gasPrice']['low'],
                    'high': entry['gasPrice']['high'],
                    'avgGas': entry['avgGas'],
                    'timestamp': entry['timestamp'],
                }
                flattened_data.append(flattened_entry)
            except Exception as e:
                print(e)

        df = pd.DataFrame(flattened_data)

        self._save_data(df, './data/gas_data_30days.csv', 720)
        self._save_data(df, './data/gas_dataset.csv')

    @staticmethod
    def _save_data(df, file_path, slice_end=None):
        if slice_end:
            data_to_save = df.iloc[0:slice_end, :][::-1].reset_index(drop=True)
        else:
            data_to_save = df[::-1]

        data_to_save.to_csv(file_path, index=False)
