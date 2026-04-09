import yfinance as yf
import pandas as pd
import os


class DataHandler:
    def __init__(self, storage_path="data"):
        self.storage_path = storage_path
        # สร้างโฟลเดอร์ data ถ้ายังไม่มี
        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path)

    def download_data(self, tickers, start_date, end_date, interval="1d"):
        """
        ดึงข้อมูลจาก Yahoo Finance
        tickers: list ของชื่อหุ้นหรือ crypto เช่น ['BTC-USD', 'ETH-USD']
        """
        print(f" Downloading data for: {tickers}")
        for ticker in tickers:
            try:
                df = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    interval=interval,
                    progress=False,
                    auto_adjust=False,
                    multi_level_index=False,
                )
                if df.empty:
                    print(f"⚠️ No data found for {ticker}")
                    continue

                # Normalize columns to a single level to keep CSV schema stable.
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                # บันทึกเป็น CSV (โชว์ทักษะการจัดการไฟล์)
                file_path = os.path.join(self.storage_path, f"{ticker.replace('-', '_')}.csv")
                df.to_csv(file_path)
                print(f" Saved {ticker} to {file_path}")
            except Exception as e:
                print(f" Error downloading {ticker}: {e}")

    def load_local_data(self, ticker):
        file_path = os.path.join(self.storage_path, f"{ticker.replace('-', '_')}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, index_col=0, parse_dates=[0])
            df.index = pd.to_datetime(df.index, errors="coerce")
            df = df[df.index.notna()]
            df.columns = [str(col).strip() for col in df.columns]

            numeric_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            if "Close" in df.columns:
                df = df.dropna(subset=["Close"])

            return df.sort_index()
        else:
            print(f"⚠️ File {file_path} not found.")
            return None
