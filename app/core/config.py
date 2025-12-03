try:
    # Pydantic v2
    from pydantic_settings import BaseSettings, SettingsConfigDict
    PYDANTIC_V2 = True
except ImportError:
    # Pydantic v1
    from pydantic import BaseSettings
    PYDANTIC_V2 = False


if PYDANTIC_V2:
    class Settings(BaseSettings):
        goalserve_api_key: str = ""  

        log_level: str = "INFO"
        log_directory: str = "logs"
        log_file: str = "app.log"

        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            extra="ignore"  # Allow extra fields in .env file
        )
else:
    class Settings(BaseSettings):
        goalserve_api_key: str = ""  

        log_level: str = "INFO"
        log_directory: str = "logs"
        log_file: str = "app.log"

        class Config:
            env_file = ".env"
            env_file_encoding = "utf-8"


settings = Settings()
