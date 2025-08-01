from tortoise.models import Model
from tortoise import fields


class Brief(Model):
    id = fields.IntField(pk=True)
    user_id = fields.CharField(max_length=255)
    topic = fields.TextField()
    brief_json = fields.JSONField()
    created_at = fields.DatetimeField(auto_now_add=True)

    class Meta:
        table = "briefs"
