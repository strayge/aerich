from enum import Enum
import importlib.util
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from click import BadOptionUsage, ClickException, Context
from pydantic import BaseModel, Field
from tortoise import BaseDBAsyncClient, Tortoise


def add_src_path(path: str) -> str:
    """
    add a folder to the paths, so we can import from there
    :param path: path to add
    :return: absolute path
    """
    if not os.path.isabs(path):
        # use the absolute path, otherwise some other things (e.g. __file__) won't work properly
        path = os.path.abspath(path)
    if not os.path.isdir(path):
        raise ClickException(f"Specified source folder does not exist: {path}")
    if path not in sys.path:
        sys.path.insert(0, path)
    return path


def get_app_connection_name(config, app_name: str) -> str:
    """
    get connection name
    :param config:
    :param app_name:
    :return:
    """
    app = config.get("apps").get(app_name)
    if app:
        return app.get("default_connection", "default")
    raise BadOptionUsage(
        option_name="--app",
        message=f'Can\'t get app named "{app_name}"',
    )


def get_app_connection(config, app) -> BaseDBAsyncClient:
    """
    get connection name
    :param config:
    :param app:
    :return:
    """
    return Tortoise.get_connection(get_app_connection_name(config, app))


def get_tortoise_config(ctx: Context, tortoise_orm: str) -> dict:
    """
    get tortoise config from module
    :param ctx:
    :param tortoise_orm:
    :return:
    """
    splits = tortoise_orm.split(".")
    config_path = ".".join(splits[:-1])
    tortoise_config = splits[-1]

    try:
        config_module = importlib.import_module(config_path)
    except ModuleNotFoundError as e:
        raise ClickException(f"Error while importing configuration module: {e}") from None

    config = getattr(config_module, tortoise_config, None)
    if not config:
        raise BadOptionUsage(
            option_name="--config",
            message=f'Can\'t get "{tortoise_config}" from module "{config_module}"',
            ctx=ctx,
        )
    return config


class OnDelete(str, Enum):
    CASCADE = 'CASCADE'
    RESTRICT = 'RESTRICT'
    SET_NULL = 'SET_NULL'
    DO_NOTHING = 'SET_DEFAULT'


class TortoiseFieldDescribe(BaseModel):
    name: str
    field_type: str
    db_column: str
    python_type: str
    generated: bool = False
    nullable: bool = False
    unique: bool = False
    indexed: bool = False
    default: Optional[Union[str, bool, int, float]]  # "<function tests.models.default_name>",
    description: Optional[str] = None
    docstring: Optional[str] = None
    constraints: Dict[str, Any] = Field(default_factory=dict)  # "ge": 1, "le": 2147483647, "max_length": 200, "readOnly": true
    db_field_types: Dict[str, str] = Field(default_factory=dict)
    auto_now_add: Optional[bool] = None
    auto_now: Optional[bool] = None


class TortoiseFkDescribe(TortoiseFieldDescribe):
    db_column: None = None
    db_field_types: None = None
    raw_field: str
    db_constraint: bool = True
    on_delete: OnDelete = OnDelete.CASCADE


class TortoiseM2MDescribe(TortoiseFieldDescribe):
    db_column: None = None
    db_field_types: None = None
    db_constraint: bool
    model_name: str
    related_name: str
    forward_key: str
    backward_key: str
    through: str
    on_delete: OnDelete
    _generated: bool


class TortoiseTableDescribe(BaseModel):
    name: str
    app: str
    table: str
    pk_field: TortoiseFieldDescribe
    abstract: bool = False
    description: Optional[str] = None
    docstring: Optional[str] = None
    unique_together: List[Tuple[str, ...]] = Field(default_factory=list)
    indexes: List[Tuple[str, ...]] = Field(default_factory=list)
    data_fields: List[TortoiseFieldDescribe] = Field(default_factory=list)
    fk_fields: List[TortoiseFkDescribe] = Field(default_factory=list)
    backward_fk_fields: list = Field(default_factory=list)
    o2o_fields: list = Field(default_factory=list)
    backward_o2o_fields: list = Field(default_factory=list)
    m2m_fields: List[TortoiseM2MDescribe] = Field(default_factory=list)

    def get_db_column_name(self, name: str) -> str:
        if self.pk_field.name == name:
            return self.pk_field.db_column
        for field in self.data_fields:
            if field.name == name:
                return field.db_column
        return name


def get_models_describe(app: str) -> Dict[str, TortoiseTableDescribe]:
    """
    get app models describe
    :param app:
    :return:
    """
    ret = {}
    for model in Tortoise.apps.get(app).values():
        describe = model.describe()
        model_describe = TortoiseTableDescribe(**describe)
        ret[model_describe.table] = model_describe
    return ret


def is_default_function(string: str):
    return re.match(r"^<function.+>$", str(string or ""))


def import_py_file(file: Path):
    module_name, file_ext = os.path.splitext(os.path.split(file)[-1])
    spec = importlib.util.spec_from_file_location(module_name, file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
