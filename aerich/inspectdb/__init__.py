from enum import Enum
from typing import Any, List, Optional, Union

from pydantic import BaseModel
from tortoise import BaseDBAsyncClient, fields


class Column(BaseModel):
    name: str
    data_type: str
    null: bool
    default: Any
    comment: Optional[str]
    pk: bool
    fk: bool
    unique: bool
    index: bool
    length: Optional[int]
    extra: Optional[str]
    decimal_places: Optional[int]
    max_digits: Optional[int]
    ref_table_name: Optional[str] = None
    ref_column_name: Optional[str] = None

    def default_value(self) -> Optional[Union[str, bool]]:
        if self.default is None:
            return None
        if self.data_type in ["tinyint", "INT"]:
            return self.default == "1"
        if self.data_type == "bool":
            return self.default == "true"
        if self.data_type in ["datetime", "timestamptz", "TIMESTAMP"]:
            return None

        if "::" in self.default:
            default = self.default.split("::")[0]
            if default.startswith("'") and default.endswith("'"):
                return default[1:-1]
            return default
        elif self.default.endswith("()"):
            return None
        else:
            return self.default

    @property
    def auto_now_add(self) -> bool:
        return (
            self.data_type in ["datetime", "timestamptz", "TIMESTAMP"]
            and "CURRENT_TIMESTAMP" == self.default
        )

    def translate(self) -> dict:
        comment = default = length = index = null = pk = ""
        if self.pk:
            pk = "pk=True, "
        elif self.unique:
            index = "unique=True, "
        elif self.index:
            index = "index=True, "
        if self.data_type in ["varchar", "VARCHAR"]:
            length = f"max_length={self.length}, "
        if self.data_type in ["decimal", "numeric"]:
            length_parts = []
            if self.max_digits:
                length_parts.append(f"max_digits={self.max_digits}")
            if self.decimal_places:
                length_parts.append(f"decimal_places={self.decimal_places}")
            length = ", ".join(length_parts)
        if self.null:
            null = "null=True, "

        default_value = self.default_value()
        if self.auto_now_add:
            default = "auto_now=True, "
        elif default_value is not None:
            default = f"default={default_value}, "

        if self.comment:
            comment = f"description='{self.comment}', "
        return {
            "name": self.name,
            "pk": pk,
            "index": index,
            "null": null,
            "default": default,
            "length": length,
            "comment": comment,
        }

    def describe(self, inspect) -> dict:
        """Analog of Tortoise describe result."""

        # get tortoise field class from database type
        field = inspect.field_map.get(self.data_type).__name__.split("_")[0] + "Field"
        replace_map = {
            "int": "Int",
            "uuid": "UUID",
            "json": "JSON",
            "bool": "Boolean",
        }
        for key, value in replace_map.items():
            field = field.replace(key, value)
        field = field[0].upper() + field[1:]

        field_class = getattr(fields, field)

        kwargs = {}
        if self.max_digits is not None:
            kwargs["max_digits"] = self.max_digits
        if self.decimal_places is not None:
            kwargs["decimal_places"] = self.decimal_places
        elif self.auto_now_add:
            kwargs["auto_now_add"] = True

        is_enum = self.comment and "\n" in self.comment
        if is_enum:
            options = {}
            for line in self.comment.split("\n"):
                name, value = line.split(": ", 1)
                options[name] = value
            kwargs["enum_type"] = Enum("CharEnum", options)
            if field.startswith("Char"):
                field_class = fields.CharEnumField
            else:
                field_class = fields.IntEnumField

        field_instance = field_class(
            pk=self.pk,
            index=self.index,
            null=self.null,
            default=self.default_value(),
            max_length=self.length,
            unique=self.unique,
            description=self.comment,
            **kwargs,
        )
        field_instance.model_field_name = self.name
        described = field_instance.describe(serializable=True)
        return described


class Inspect:
    _table_template = "class {table}(Model):\n"

    def __init__(self, conn: BaseDBAsyncClient, tables: Optional[List[str]] = None):
        self.conn = conn
        try:
            self.database = conn.database
        except AttributeError:
            pass
        self.tables = tables

    @property
    def field_map(self) -> dict:
        raise NotImplementedError

    async def inspect(self) -> str:
        if not self.tables:
            self.tables = await self.get_all_tables()
        result = "from tortoise import Model, fields\n\n\n"
        tables = []
        for table in self.tables:
            columns = await self.get_columns(table)
            fields = []
            model = self._table_template.format(table=table.title().replace("_", ""))
            for column in columns:
                field = self.field_map[column.data_type](**column.translate())
                fields.append("    " + field)
            tables.append(model + "\n".join(fields))
        return result + "\n\n\n".join(tables)

    async def get_columns(self, table: str) -> List[Column]:
        raise NotImplementedError

    async def get_all_tables(self) -> List[str]:
        raise NotImplementedError

    async def describe(self, app: str, class_names: dict[str, str]) -> dict:
        def table_name_to_class_name(table_name: str) -> str:
            if table_name in class_names:
                return class_names[table_name]
            else:
                # convert table name to camel case
                return f'{app}.{table_name.title().replace("_", "")}'

        models = {}
        for table_name in await self.get_all_tables():
            columns = await self.get_columns(table_name)
            columns_described: dict[str, dict] = {}
            columns_dataclass: dict[str, Column] = {}
            for column in columns:
                columns_described[column.name] = column.describe(self)
                columns_dataclass[column.name] = column

            foreign_keys = []
            for column_data in columns_dataclass.values():
                if column_data.fk:
                    foreign_keys.append(
                        {
                            "constraints": {},
                            "db_constraint": True,
                            "default": None,
                            "description": None,
                            "docstring": None,
                            "field_type": "ForeignKeyFieldInstance",
                            "generated": False,
                            "indexed": False,
                            "name": column_data.name[:-3],
                            "nullable": False,
                            "on_delete": "CASCADE",
                            "python_type": table_name_to_class_name(column_data.ref_table_name),
                            "raw_field": column_data.name,
                            "unique": False,
                        }
                    )

            pk_names = [column.name for column in columns if column.pk]
            data_names = [column.name for column in columns if column.name not in pk_names]

            class_name = table_name_to_class_name(table_name)
            models[class_name] = {
                "name": class_name,
                "app": app,
                "table": table_name,
                "abstract": False,
                "description": None,
                "docstring": None,
                "unique_together": [],
                "indexes": [],
                "pk_field": columns_described[pk_names[0]],
                "data_fields": [
                    col for name, col in columns_described.items() if name in data_names
                ],
                "fk_fields": foreign_keys,
                "backward_fk_fields": [],
                "o2o_fields": [],
                "backward_o2o_fields": [],
                "m2m_fields": [],
            }
        return models

    @classmethod
    def decimal_field(cls, **kwargs) -> str:
        return "{name} = fields.DecimalField({pk}{index}{length}{null}{default}{comment})".format(
            **kwargs
        )

    @classmethod
    def time_field(cls, **kwargs) -> str:
        return "{name} = fields.TimeField({null}{default}{comment})".format(**kwargs)

    @classmethod
    def date_field(cls, **kwargs) -> str:
        return "{name} = fields.DateField({null}{default}{comment})".format(**kwargs)

    @classmethod
    def float_field(cls, **kwargs) -> str:
        return "{name} = fields.FloatField({null}{default}{comment})".format(**kwargs)

    @classmethod
    def datetime_field(cls, **kwargs) -> str:
        return "{name} = fields.DatetimeField({null}{default}{comment})".format(**kwargs)

    @classmethod
    def text_field(cls, **kwargs) -> str:
        return "{name} = fields.TextField({null}{default}{comment})".format(**kwargs)

    @classmethod
    def char_field(cls, **kwargs) -> str:
        return "{name} = fields.CharField({pk}{index}{length}{null}{default}{comment})".format(
            **kwargs
        )

    @classmethod
    def int_field(cls, **kwargs) -> str:
        return "{name} = fields.IntField({pk}{index}{comment})".format(**kwargs)

    @classmethod
    def smallint_field(cls, **kwargs) -> str:
        return "{name} = fields.SmallIntField({pk}{index}{comment})".format(**kwargs)

    @classmethod
    def bigint_field(cls, **kwargs) -> str:
        return "{name} = fields.BigIntField({pk}{index}{default}{comment})".format(**kwargs)

    @classmethod
    def bool_field(cls, **kwargs) -> str:
        return "{name} = fields.BooleanField({null}{default}{comment})".format(**kwargs)

    @classmethod
    def uuid_field(cls, **kwargs) -> str:
        return "{name} = fields.UUIDField({pk}{index}{default}{comment})".format(**kwargs)

    @classmethod
    def json_field(cls, **kwargs) -> str:
        return "{name} = fields.JSONField({null}{default}{comment})".format(**kwargs)

    @classmethod
    def binary_field(cls, **kwargs) -> str:
        return "{name} = fields.BinaryField({null}{default}{comment})".format(**kwargs)
