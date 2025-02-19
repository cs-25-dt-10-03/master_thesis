import pandas as pd

class SpotPriceData:
    def __init__(self, csv_file: str):
        self.data = pd.read_csv(csv_file, parse_dates=['HourDK'])
        self.data.sort_values('HourDK', inplace=True)
        self.data.reset_index(drop=True, inplace=True)

    def get_price(self, slot_index: int) -> float:
        if slot_index < len(self.data):
            return self.data.iloc[slot_index]['SpotPriceEUR']
        else:
            print('skibidi rizz lmao')
            return self.data.iloc[-1]['SpotPriceEUR']


    def get_price_by_datetime(self, dt) -> float:
        diffs = (self.data['HourDK'] - dt).abs()
        idx = diffs.idxmin()
        return self.data.loc[idx, 'SpotPriceEUR']