import pandas as pd

class PlayerService:

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def get_median_strikes(self, fighter_id: int):
        """
        Finds all rows where the fighter appears as local or away.
        Computes median total strikes (head/body/leg).
        """
        df = self.df
        print(df.head())
        local_rows = df[df["local_id"] == fighter_id][[
            "local_strikes_total_head",
            "local_strikes_total_body",
            "local_strikes_total_leg",
            "localteam_name"
        ]]

        away_rows = df[df["away_id"] == fighter_id][[
            "away_strikes_total_head",
            "away_strikes_total_body",
            "away_strikes_total_leg",
            "awayteam_name"
        ]]

        if local_rows.empty and away_rows.empty:
            return None

        # Normalize column names
        local_rows = local_rows.rename(columns={
            "local_strikes_total_head": "head",
            "local_strikes_total_body": "body",
            "local_strikes_total_leg": "leg",
            "localteam_name": "name"
        })

        away_rows = away_rows.rename(columns={
            "away_strikes_total_head": "head",
            "away_strikes_total_body": "body",
            "away_strikes_total_leg": "leg",
            "awayteam_name": "name"
        })

        combined = pd.concat([local_rows, away_rows], ignore_index=True)
        
        return {
            "id": fighter_id,
            "name": combined["name"].dropna().iloc[0],
            "head": int(combined["head"].median()),
            "body": int(combined["body"].median()),
            "leg": int(combined["leg"].median())
        }
