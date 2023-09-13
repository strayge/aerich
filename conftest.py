import asyncio
import os

import pytest
from tortoise import Tortoise, expand_db_url, generate_schema_for_client
from tortoise.backends.asyncpg.schema_generator import AsyncpgSchemaGenerator
from tortoise.backends.mysql.schema_generator import MySQLSchemaGenerator
from tortoise.backends.sqlite.schema_generator import SqliteSchemaGenerator

from aerich.ddl.mysql import MysqlDDL
from aerich.ddl.postgres import PostgresDDL
from aerich.ddl.sqlite import SqliteDDL
from aerich.migrate import Migrate

db_url = os.getenv("TEST_DB", "sqlite://test.sqlite3")
db_url_second = os.getenv("TEST_DB_SECOND", "sqlite://:memory:")
tortoise_orm = {
    "connections": {
        "default": expand_db_url(db_url, True),
        # "second": expand_db_url(db_url_second, True),
    },
    "apps": {
        "models": {"models": ["tests.models", "aerich.models"], "default_connection": "default"},
        # "models_second": {"models": ["tests.models_second"], "default_connection": "second"},
    },
}


@pytest.fixture(scope="function", autouse=True)
def reset_migrate():
    Migrate.upgrade_operators = []
    Migrate.downgrade_operators = []
    Migrate._upgrade_fk_m2m_index_operators = []
    Migrate._downgrade_fk_m2m_index_operators = []
    Migrate._upgrade_m2m = []
    Migrate._downgrade_m2m = []


@pytest.fixture(scope="session")
def event_loop():
    policy = asyncio.get_event_loop_policy()
    res = policy.new_event_loop()
    asyncio.set_event_loop(res)
    res._close = res.close
    res.close = lambda: None

    yield res

    res._close()


@pytest.fixture(scope="session", autouse=True)
async def initialize_tests(event_loop, request):
    await Tortoise.init(config=tortoise_orm, _create_db=True)
    await generate_schema_for_client(Tortoise.get_connection("default"), safe=True)

    client = Tortoise.get_connection("default")
    if client.schema_generator is MySQLSchemaGenerator:
        Migrate.ddl = MysqlDDL(client)
    elif client.schema_generator is SqliteSchemaGenerator:
        Migrate.ddl = SqliteDDL(client)
    elif client.schema_generator is AsyncpgSchemaGenerator:
        Migrate.ddl = PostgresDDL(client)
    Migrate.dialect = Migrate.ddl.DIALECT
    request.addfinalizer(lambda: event_loop.run_until_complete(Tortoise._drop_databases()))
