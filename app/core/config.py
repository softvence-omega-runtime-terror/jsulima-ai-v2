
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    goalserve_api_key: str = ""  
    goalserve_base_url: str = "http://www.goalserve.com/getfeed"
    api_request_delay: float = 0.5
    log_level: str = "INFO"
    log_directory: str = "logs"
    log_file: str = "app.log"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
