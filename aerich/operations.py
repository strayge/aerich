import json
from hashlib import sha256
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel

from aerich.utils import OnDelete, TortoiseFieldDescribe, TortoiseFkDescribe, TortoiseTableDescribe


def _generate_index_name(unique: bool, table_name: str, field_names: List[str]) -> str:
    # NOTE: for compatibility, index name should not be longer than 30
    # characters (Oracle limit).
    # That's why we slice some of the strings here.

    def _make_hash(*args: str, length: int) -> str:
        # Hash a set of string values and get a digest of the given length.
        return sha256(";".join(args).encode("utf-8")).hexdigest()[:length]

    prefix = "uid" if unique else "idx"
    index_name = "{}_{}_{}_{}".format(
        prefix,
        table_name[:11],
        field_names[0][:7],
        _make_hash(table_name, *field_names, length=6),
    )
    return index_name


def _field_into_db_type(field_type: str) -> str:
    map = {
        'DecimalField': 'REAL',
        'TimeField': 'TIME',
        'DateField': 'DATE',
        'FloatField': 'REAL',
        'DatetimeField': 'TIMESTAMP',
        'TextField': 'TEXT',
        'CharField': 'VARCHAR',
        'IntField': 'INTEGER',
        'SmallIntField': 'SMALLINT',
        'BigIntField': 'BIGINT',
        'BooleanField': 'INT',
        'UUIDField': 'VARCHAR',
        'JSONField': 'JSON',
        'BinaryField': 'BLOB',
        'IntEnumFieldInstance': 'INTEGER',
    }
    return map[field_type]


class Operation(BaseModel):
    async def to_sql(self) -> str:
        """Convert operation to sql."""
        raise NotImplementedError


class RenameTable(Operation):
    old_name: str
    new_name: str

    async def to_sql(self) -> str:
        return f"ALTER TABLE {self.old_name} RENAME TO {self.new_name}"


class AddTable(Operation):
    table_name: str
    fields: list['AddField']
    fk_fields: list['AddFK']

    async def to_sql(self) -> str:
        fields = self.fields + self.fk_fields
        fields_sql = ", ".join(field.create_table_column_sql() for field in fields)
        return f"CREATE TABLE {self.table_name} ({fields_sql})"


class DropTable(Operation):
    table_name: str

    async def to_sql(self) -> str:
        return f"DROP TABLE {self.table_name}"


class CreateM2M(Operation):
    ...


class DropM2M(Operation):
    table_name: str


class AddIndex(Operation):
    table_name: str
    field_names: Tuple[str, ...]
    unique: bool = False

    async def to_sql(self) -> str:
        index_name = _generate_index_name(self.unique, self.table_name, self.field_names)
        fields_sql = ", ".join(self.field_names)
        unique_sql = "UNIQUE" if self.unique else ""
        return f"CREATE {unique_sql} INDEX {index_name} ON {self.table_name} ({fields_sql})"


class DropIndex(Operation):
    table_name: str
    field_names: Tuple[str, ...]
    unique: bool = False

    async def to_sql(self) -> str:
        index_name = _generate_index_name(self.unique, self.table_name, self.field_names)
        return f"DROP INDEX {index_name}"


class AddField(Operation):
    table_name: str
    field_name: str
    field_type: str
    primary: bool = False
    nullable: bool = False
    unique: bool = False
    default: Optional[Union[str, bool, int, float]] = None
    comment: Optional[str] = None

    def create_table_column_sql(self) -> str:
        db_type = _field_into_db_type(self.field_type)
        sql = f"{self.field_name} {db_type}"
        if self.primary:
            sql += " PRIMARY KEY"
        else:
            if not self.nullable:
                sql += " NOT NULL"
            if self.unique:
                sql += " UNIQUE"
        if self.default is not None:
            sql += f" DEFAULT {self.default!r}"
        # if self.comment is not None:
        #     sql += f" COMMENT {self.comment}"
        return sql

    async def to_sql(self) -> str:
        return f"ALTER TABLE {self.table_name} ADD COLUMN {self.create_table_column_sql()}"


class RemoveField(Operation):
    table_name: str
    field_name: str

    async def to_sql(self) -> str:
        return f"ALTER TABLE {self.table_name} DROP COLUMN {self.field_name}"


class RenameField(Operation):
    table_name: str
    old_field_name: str
    new_field_name: str

    async def to_sql(self) -> str:
        return f"ALTER TABLE {self.table_name} RENAME COLUMN {self.old_field_name} TO {self.new_field_name}"


class ChangeField(Operation):
    ...


class ModifyField(Operation):
    ...


class AlterDefault(Operation):
    ...


class AlterNull(Operation):
    ...


class SetComment(Operation):
    ...


class AddFK(Operation):
    table_name: str
    field_name: str
    on_delete: OnDelete
    related_model_name: str
    related_field_name: str
    nullable: bool = False
    unique: bool = False
    default: Optional[Union[str, bool, int, float]] = None
    comment: Optional[str] = None

    def create_table_column_sql(self) -> str:
        sql = f"FOREIGN KEY ({self.field_name}) REFERENCES {self.related_model_name} ({self.related_field_name})"
        sql += f" ON DELETE {self.on_delete.value}"
        return sql

    async def to_sql(self) -> str:
        return f"ALTER TABLE {self.table_name} ADD {self.create_table_column_sql()}"


class DropFK(Operation):
    table_name: str
    field_name: str

    async def to_sql(self) -> str:
        return f"ALTER TABLE {self.table_name} DROP FOREIGN KEY {self.field_name}"


def _remove_field_by_db_column(
    fields: Union[List[TortoiseFieldDescribe], List[TortoiseFkDescribe]],
    db_column: str,
) -> None:
    for i, field in enumerate(fields):
        if field.db_column == db_column:
            del fields[i]
            break


def _convert_into_tortoise_field(field: AddField) -> TortoiseFieldDescribe:
    return TortoiseFieldDescribe(
        name=field.field_name,
        field_type=field.field_type,
        db_column=field.field_name,
        python_type="",
        db_field_types={},
        nullable=field.nullable,
        unique=field.unique,
        default=field.default,
        description=field.comment,
    )


def _convert_into_tortoise_fk_field(field: AddFK) -> TortoiseFkDescribe:
    return TortoiseFkDescribe(
        name=field.field_name,
        field_type="",
        python_type="",
        nullable=field.nullable,
        unique=field.unique,
        default=field.default,
        description=field.related_model_name,
        db_constraint=True,
        raw_field=field.related_field_name,
        on_delete=field.on_delete,
    )


def make_model(operations: list[Operation]) -> Dict[str, TortoiseTableDescribe]:
    """Combine all upgrade operations to model dict."""
    tables: Dict[str, TortoiseTableDescribe] = {}
    for operation in operations:
        if isinstance(operation, AddTable):
            pk_field = None
            data_fields = []
            for field in operation.fields:
                if field.primary:
                    pk_field = _convert_into_tortoise_field(field)
                else:
                    data_fields.append(_convert_into_tortoise_field(field))
            for field in operation.fk_fields:
                data_fields.append(_convert_into_tortoise_fk_field(field))
            tables[operation.table_name] = TortoiseTableDescribe(
                name=operation.table_name,
                app="",
                table=operation.table_name,
                pk_field=pk_field,
                data_fields=data_fields,
            )
        elif isinstance(operation, DropTable):
            del tables[operation.table_name]
        elif isinstance(operation, RenameTable):
            tables[operation.new_name] = tables[operation.old_name]
            del tables[operation.old_name]
        elif isinstance(operation, AddField):
            field = _convert_into_tortoise_field(operation)
            if operation.primary:
                tables[operation.table_name].pk_field = field
            else:
                tables[operation.table_name].data_fields.append(field)
        elif isinstance(operation, RemoveField):
            if operation.field_name == tables[operation.table_name].pk_field.db_column:
                tables[operation.table_name].pk_field = None
            else:
                _remove_field_by_db_column(tables[operation.table_name].data_fields, operation.field_name)
        elif isinstance(operation, RenameField):
            if operation.old_field_name == tables[operation.table_name].pk_field.db_column:
                tables[operation.table_name].pk_field.db_column = operation.new_field_name
            else:
                for field in tables[operation.table_name].data_fields:
                    if field.db_column == operation.old_field_name:
                        field.db_column = operation.new_field_name
                        break
        elif isinstance(operation, AddIndex):
            tables[operation.table_name].unique_together.append(operation.field_names)
        elif isinstance(operation, DropIndex):
            tables[operation.table_name].unique_together = [
                u for u in tables[operation.table_name].unique_together
                if u != operation.field_names
            ]
        elif isinstance(operation, AddFK):
            tables[operation.table_name].fk_fields.append(_convert_into_tortoise_fk_field(operation))
        elif isinstance(operation, DropFK):
            _remove_field_by_db_column(tables[operation.table_name].fk_fields, operation.field_name)
    return tables


def compare_models(
    old_model: Dict[str, TortoiseTableDescribe],
    new_model: Dict[str, TortoiseTableDescribe],
) -> tuple[list[Operation], list[Operation]]:
    """Compare old model and new model, return upgrade & downgrade operations."""
    print('new_model', new_model)

    upgrade_operations = []
    downgrade_operations = []

    table_pairs: List[Tuple[Optional[TortoiseTableDescribe], Optional[TortoiseTableDescribe]]] = []
    for name, model in old_model.items():
        table_pairs.append((model, new_model.get(name)))
    for name, model in new_model.items():
        if name not in old_model:
            table_pairs.append((None, model))

    for old_table, new_table in table_pairs:
        up_ops, down_ops = compare_tables(old_table, new_table)
        upgrade_operations.extend(up_ops)
        downgrade_operations.extend(down_ops)

    return upgrade_operations, downgrade_operations


def convert_tortoise_field(table_name: str, field: TortoiseFieldDescribe, primary: bool = False) -> AddField:
    default = field.default
    if isinstance(default, str) and default.startswith("<function "):
        default = None
    return AddField(
        table_name=table_name,
        field_name=field.db_column,
        field_type=field.field_type,
        primary=primary,
        nullable=field.nullable,
        unique=field.unique,
        default=default,
        comment=field.description,
    )


def convert_tortoise_fk(table_name: str, field: TortoiseFkDescribe) -> AddFK:
    default = field.default
    if isinstance(default, str) and default.startswith("<function "):
        default = None
    return AddFK(
        table_name=table_name,
        field_name=field.raw_field,
        on_delete=field.on_delete,
        related_model_name=field.python_type.split('.')[-1],
        related_field_name=field.raw_field,
        nullable=field.nullable,
        unique=field.unique,
        default=default,
        comment=field.description,
    )


def compare_tables(
    old_table: Optional[TortoiseTableDescribe],
    new_table: Optional[TortoiseTableDescribe],
) -> tuple[list[Operation], list[Operation]]:
    upgrade_operations = []
    downgrade_operations = []

    if not old_table and not new_table:
        return upgrade_operations, downgrade_operations

    # fields
    if not old_table and new_table:
        print(f'new table: {new_table.table}')
        upgrade_operations.append(AddTable(
            table_name=new_table.table,
            fields=(
                [convert_tortoise_field(new_table.table, new_table.pk_field, primary=True)]
                + [convert_tortoise_field(new_table.table, f) for f in new_table.data_fields]
            ),
            fk_fields=[convert_tortoise_fk(new_table.table, f) for f in new_table.fk_fields],
        ))
        downgrade_operations.append(DropTable(table_name=new_table.table))

    if old_table and not new_table:
        print(f'removed table: {old_table.table}')
        upgrade_operations.append(DropTable(table_name=old_table.table))
        downgrade_operations.append(AddTable(
            table_name=old_table.table,
            fields=(
                [convert_tortoise_field(old_table.table, old_table.pk_field, primary=True)]
                + [convert_tortoise_field(old_table.table, f) for f in old_table.data_fields]
            ),
            fk_fields=[convert_tortoise_fk(old_table.table, f) for f in old_table.fk_fields],
        ))

    if old_table and new_table:
        print(f'same table: {new_table.table}')
        fields_old = {field.db_column: field for field in [old_table.pk_field] + old_table.data_fields}
        fields_new = {field.db_column: field for field in [new_table.pk_field] + new_table.data_fields}
        for db_column in fields_old.keys() | fields_new.keys():
            up_ops, down_ops = compare_fields(new_table.table, fields_old.get(db_column), fields_new.get(db_column))
            upgrade_operations.extend(up_ops)
            downgrade_operations.extend(down_ops)
        fk_fields_old = {field.raw_field: field for field in old_table.fk_fields}
        fk_fields_new = {field.raw_field: field for field in new_table.fk_fields}
        for raw_field in fk_fields_old.keys() | fk_fields_new.keys():
            up_ops, down_ops = compare_fk_fields(new_table.table, fk_fields_old.get(raw_field), fk_fields_new.get(raw_field))
            upgrade_operations.extend(up_ops)
            downgrade_operations.extend(down_ops)

    # unique_together
    new_table_unique = set()
    if new_table:
        for unique in new_table.unique_together:
            new_table_unique.add(tuple([new_table.get_db_column_name(name) for name in unique]))
    old_table_unique = set()
    if old_table:
        for unique in old_table.unique_together:
            old_table_unique.add(tuple([old_table.get_db_column_name(name) for name in unique]))

    for new_unique in new_table_unique - old_table_unique:
        print(f'new unique: {new_unique}')
        upgrade_operations.append(AddIndex(
            table_name=new_table.table,
            field_names=new_unique,
            unique=True,
        ))
        downgrade_operations.append(DropIndex(
            table_name=new_table.table,
            field_names=new_unique,
            unique=True,
        ))
    for old_unique in old_table_unique - new_table_unique:
        print(f'removed unique: {old_unique}')
        upgrade_operations.append(DropIndex(
            table_name=new_table.table,
            field_names=old_unique,
            unique=True,
        ))
        downgrade_operations.append(AddIndex(
            table_name=new_table.table,
            field_names=old_unique,
            unique=True,
        ))

    # fk
    if old_table and new_table:
        new_fk_fields = set()
        for fk in new_table.fk_fields:
            new_fk_fields.add(convert_tortoise_fk(new_table.table, fk))
        old_fk_fields = set()
        for fk in old_table.fk_fields:
            old_fk_fields.add(convert_tortoise_fk(old_table.table, fk))

        for new_fk in new_fk_fields - old_fk_fields:
            print(f'new fk: {new_fk.raw_field}')
            upgrade_operations.append(AddFK(
                table_name=new_table.table,
                field_name=new_fk.raw_field,
                on_delete=new_fk.on_delete,
                related_model_name=new_fk.description,
                related_field_name=new_fk.raw_field,
                nullable=new_fk.nullable,
                unique=new_fk.unique,
                default=new_fk.default,
                comment=new_fk.description,
            ))
            downgrade_operations.append(DropFK(
                table_name=new_table.table,
                field_name=new_fk.raw_field,
            ))
        for old_fk in old_fk_fields - new_fk_fields:
            print(f'removed fk: {old_fk.raw_field}')
            upgrade_operations.append(DropFK(
                table_name=old_table.table,
                field_name=old_fk.raw_field,
            ))
            downgrade_operations.append(AddFK(
                table_name=old_table.table,
                field_name=old_fk.raw_field,
                on_delete=old_fk.on_delete,
                related_model_name=old_fk.description,
                related_field_name=old_fk.raw_field,
                nullable=old_fk.nullable,
                unique=old_fk.unique,
                default=old_fk.default,
                comment=old_fk.description,
            ))

    # TODO: indexes, m2m, o2o

    return upgrade_operations, downgrade_operations


def compare_fields(
    table_name: str,
    old_field: Optional[TortoiseFieldDescribe],
    new_field: Optional[TortoiseFieldDescribe],
) -> tuple[list[Operation], list[Operation]]:
    upgrade_operations = []
    downgrade_operations = []

    if not old_field and not new_field:
        pass
    elif not old_field and new_field:
        print(f'  new field: {new_field.db_column}')
        upgrade_operations.append(convert_tortoise_field(table_name, new_field))
        downgrade_operations.append(RemoveField(table_name=table_name, field_name=new_field.name))
    elif old_field and not new_field:
        print(f'  removed field: {old_field.db_column}')
        upgrade_operations.append(RemoveField(table_name=table_name, field_name=old_field.name))
        downgrade_operations.append(convert_tortoise_field(table_name, old_field))
    else:
        # TODO: alter column
        print(f'  same field: {new_field.db_column}')
    return upgrade_operations, downgrade_operations


def compare_fk_fields(
    table_name: str,
    old_field: Optional[TortoiseFkDescribe],
    new_field: Optional[TortoiseFkDescribe],
) -> tuple[list[Operation], list[Operation]]:
    upgrade_operations = []
    downgrade_operations = []

    if not old_field and not new_field:
        pass
    elif not old_field and new_field:
        print(f'  new fk field: {new_field.raw_field}')
        upgrade_operations.append(convert_tortoise_fk(table_name, new_field))
        downgrade_operations.append(DropFK(table_name=table_name, field_name=new_field.name))
    elif old_field and not new_field:
        print(f'  removed fk field: {old_field.raw_field}')
        upgrade_operations.append(DropFK(table_name=table_name, field_name=old_field.name))
        downgrade_operations.append(convert_tortoise_fk(table_name, old_field))
    else:
        # TODO: alter fk
        print(f'  same field: {new_field.raw_field}')
    return upgrade_operations, downgrade_operations


def read_operations(path: Path) -> tuple[list[Operation], list[Operation]]:
    """Read upgrade & downgrade operations from file."""
    with open(path, encoding="utf-8") as f:
        content = json.load(f)

    classes = {
        name: cls
        for name, cls in globals().items()
        if isinstance(cls, type) and issubclass(cls, Operation)
    }

    upgrade = [classes[name](**kwargs) for name, kwargs in content["upgrade"]]
    downgrade = [classes[name](**kwargs) for name, kwargs in content["downgrade"]]

    return upgrade, downgrade


def write_operations(
    path: Path,
    upgrade: list[Operation],
    downgrade: list[Operation],
) -> None:
    """Write upgrade & downgrade operations to file."""

    content = {
        "upgrade": [(op.__class__.__name__, op.model_dump()) for op in upgrade],
        "downgrade": [(op.__class__.__name__, op.model_dump()) for op in downgrade],
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(content, f, ensure_ascii=False, indent=4)
