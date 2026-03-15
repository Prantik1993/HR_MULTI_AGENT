from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    openai_api_key: str
    openai_model: str = "gpt-4o"
    chroma_path: str = "./vectorstore"
    collection_name: str = "hr_docs"
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
