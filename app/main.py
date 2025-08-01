from fastapi import FastAPI
from app.api import routes
from app.db.config import TORTOISE_ORM
from tortoise.contrib.fastapi import register_tortoise

app = FastAPI()

register_tortoise(
    app,
    config=TORTOISE_ORM,
    generate_schemas=True,
    add_exception_handlers=True,
)

app.include_router(routes.router)
