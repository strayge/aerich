import pytest
from tortoise import Model

from aerich import Command
from aerich.ddl.sqlite import SqliteDDL
from aerich.exceptions import UpgradeError
from aerich.migrate import Migrate
from conftest import tortoise_orm


def mock_migrations_list(mocker, migrations=("1_new.py",)):
    mocker.patch("os.listdir", return_value=migrations)


def mock_migrations_sql(mocker):
    class EmptyMigration:
        @staticmethod
        async def upgrade(db) -> str:
            return "SELECT 1;"

        @staticmethod
        async def downgrade(db) -> str:
            return "SELECT 1;"

    mocker.patch("aerich.import_py_file", return_value=EmptyMigration)


def change_model_class(mocker):
    mocker.patch("tests.models.Category", Model)


async def test_upgrade_without_migrate(mocker):
    """Test `upgrade` command will fail with changes in models."""
    if isinstance(Migrate.ddl, SqliteDDL):
        # test requires saving state between connections
        return

    mock_migrations_list(mocker)
    mock_migrations_sql(mocker)

    command = Command(tortoise_orm)

    await command.upgrade()  # there is still missed case with changes before first upgrade

    change_model_class(mocker)
    await command.init()
    with pytest.raises(UpgradeError):
        await command.upgrade()
