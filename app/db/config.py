import os
from dotenv import load_dotenv
from tortoise import Tortoise

load_dotenv()

DB_URL = os.getenv("DATABASE_URL")

TORTOISE_ORM = {
    "connections": {"default": DB_URL},
    "apps": {
        "models": {
            "models": [
                "app.db.models",
                # "aerich.models",  # aerich for migrations
            ],
            "default_connection": "default",
        }
    },
}
