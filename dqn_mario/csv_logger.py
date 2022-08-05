import pandas as pd
from config import Config

class CSVLogger:
    @staticmethod
    def df_path(config: Config):
        return config.model_dir / f'{config.project_id}_log.csv'

    @staticmethod
    def load_df(config: Config, columns: list) -> pd.DataFrame:
        df_path = CSVLogger.df_path(config)
        if df_path.exists():
            return pd.read_csv(df_path)
        else:
            return pd.DataFrame(columns=columns)

    def __init__(self, config: Config, columns: list) -> None:
        self.config = config
        self.columns = columns
        self.df = CSVLogger.load_df(config, columns)

    def append(self, row: dict):
        self.df = pd.concat([self.df, pd.DataFrame(row, index=['episode'])])

    def save(self):
        df_path = CSVLogger.df_path(self.config)
        self.df = self.df.drop_duplicates(subset=['episode'])
        self.df.to_csv(df_path, index=False)