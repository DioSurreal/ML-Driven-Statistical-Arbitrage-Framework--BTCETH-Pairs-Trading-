import numpy as np
import pandas as pd


class PairsStrategy:
    def __init__(self, z_entry=2.0, z_exit=0.0):
        self.z_entry = z_entry
        self.z_exit = z_exit

    def calculate_spread(self, df1, df2):
        ratio = df1['Close'] / df2['Close']
        mean = ratio.rolling(window=30).mean()
        std = ratio.rolling(window=30).std()
        return (ratio - mean) / std

    def generate_signals(self, df1, df2, regime_series, target_regime=0):
        z_score = self.calculate_spread(df1, df2)
        signals = pd.DataFrame(index=z_score.index)
        signals['z_score'] = z_score
        signals['regime'] = regime_series
        signals['signal'] = 0

        signals.loc[(signals['z_score'] > self.z_entry) & (
            signals['regime'] == target_regime), 'signal'] = -1
        signals.loc[(signals['z_score'] < -self.z_entry) &
                    (signals['regime'] == target_regime), 'signal'] = 1
        signals.loc[abs(signals['z_score']) <= self.z_exit, 'signal'] = 0

        return signals.fillna(0)
