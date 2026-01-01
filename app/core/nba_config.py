
from pydantic_settings import BaseSettings


class NBASettings(BaseSettings):
    # These will be loaded from .env file or environment variables
    goalserve_api_key: str = ""
    goalserve_base_url: str = "https://www.goalserve.com/getfeed/"
    use_mock_data: bool = False

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


nba_settings = NBASettings()
