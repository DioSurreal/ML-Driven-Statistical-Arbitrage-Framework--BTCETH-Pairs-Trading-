import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

class RegimeDetector:
    def __init__(self, n_regimes=3):
        self.model = GaussianMixture(n_components=n_regimes, random_state=42, n_init=10)
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.regime_labels = {} 

    def extract_features(self, df):
        data = df.copy()
        data['returns'] = np.log(data['Close'] / data['Close'].shift(1))
        data['volatility'] = data['returns'].rolling(window=20).std()
        data['ma_ratio'] = data['Close'] / data['Close'].rolling(window=50).mean()
        return data[['returns', 'volatility', 'ma_ratio']].dropna()

    def fit(self, df):
        features = self.extract_features(df)
        scaled_features = self.scaler.fit_transform(features)
        self.model.fit(scaled_features)
        self.is_fitted = True

        # --- AUTO-LABELING LOGIC ---
        train_preds = self.model.predict(scaled_features)
        temp_df = features.copy()
        temp_df['regime'] = train_preds
        stats = temp_df.groupby('regime').mean()

        sideway_idx = stats['volatility'].idxmin()
        
        remaining_indices = [i for i in range(3) if i != sideway_idx]
        if stats.loc[remaining_indices[0], 'ma_ratio'] > stats.loc[remaining_indices[1], 'ma_ratio']:
            bull_idx, bear_idx = remaining_indices[0], remaining_indices[1]
        else:
            bull_idx, bear_idx = remaining_indices[1], remaining_indices[0]

        self.regime_labels = {
            bear_idx: "Bear Market",
            sideway_idx: "Sideways Market",
            bull_idx: "Bull Market"
        }
        
        print("\n📊 Market Regime Analysis Complete:")
        for idx, label in self.regime_labels.items():
            print(f"🔹 Class {idx}: {label} (Vol: {stats.loc[idx, 'volatility']:.4f}, MA-Ratio: {stats.loc[idx, 'ma_ratio']:.4f})")

    def predict(self, df):
        if not self.is_fitted: return None
        features = self.extract_features(df)
        scaled_features = self.scaler.transform(features)
        regimes = self.model.predict(scaled_features)
        return pd.Series(regimes, index=features.index)

    def get_class_by_label(self, label_name):
        """ส่งคืนเลข Class จากชื่อ (Bear, Sideways, Bull)"""
        for idx, label in self.regime_labels.items():
            if label_name.lower() in label.lower():
                return idx
        return None