
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    goalserve_api_key: str = ""  
    goalserve_base_url: str = "http://www.goalserve.com/getfeed"
    player_image_base_url: str = "http://127.0.0.1:8000/api/v1/static/"
    api_request_delay: float = 0.5
    
    
    log_level: str = "INFO"
    log_directory: str = "logs"
    log_file: str = "app.log"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"





class NBASettings(Settings):
    # These will be loaded from .env file or environment variables
    use_mock_data: bool = False


nba_settings = NBASettings()
settings = Settings()