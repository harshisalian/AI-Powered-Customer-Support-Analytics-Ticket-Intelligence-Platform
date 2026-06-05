from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "Smart Customer Support Ticket Classification System"
    PROJECT_VERSION: str = "0.1.0"

    MYSQL_HOST: str = "localhost"
    MYSQL_PORT: int = 3306
    MYSQL_USER: str = "support_user"
    MYSQL_PASSWORD: str = "support_password"
    MYSQL_DATABASE: str = "support_tickets_db"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
