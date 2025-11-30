from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    GOALSERVE_API_KEY: str

    log_level: str = "INFO"
    log_directory: str = "logs"
    log_file: str = "app.log"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
