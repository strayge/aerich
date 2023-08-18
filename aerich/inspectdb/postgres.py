from typing import List, Optional

from tortoise import BaseDBAsyncClient

from aerich.inspectdb import Column, Inspect


class InspectPostgres(Inspect):
    def __init__(self, conn: BaseDBAsyncClient, tables: Optional[List[str]] = None):
        super().__init__(conn, tables)
        self.schema = self.conn.server_settings.get("schema") or "public"

    @property
    def field_map(self) -> dict:
        return {
            "int4": self.int_field,
            "int8": self.int_field,
            "smallint": self.smallint_field,
            "varchar": self.char_field,
            "text": self.text_field,
            "bigint": self.bigint_field,
            "timestamptz": self.datetime_field,
            "float4": self.float_field,
            "float8": self.float_field,
            "date": self.date_field,
            "time": self.time_field,
            "decimal": self.decimal_field,
            "numeric": self.decimal_field,
            "uuid": self.uuid_field,
            "jsonb": self.json_field,
            "bytea": self.binary_field,
            "bool": self.bool_field,
            "timestamp": self.datetime_field,
        }

    async def get_all_tables(self) -> List[str]:
        sql = "select TABLE_NAME from information_schema.TABLES where table_catalog=$1 and table_schema=$2"
        ret = await self.conn.execute_query_dict(sql, [self.database, self.schema])
        return list(map(lambda x: x["table_name"], ret))

    async def get_columns(self, table: str) -> List[Column]:
        columns = []
        sql = """SELECT
                    c.column_name,
                    col_description(($3 || '.' || $2)::regclass, c.ordinal_position) AS column_comment,
                    tc.constraint_type AS column_key,
                    tc.constraint_name,
                    ccu.table_name AS ref_table_name,
                    ccu.column_name AS ref_column_name,
                    udt_name AS data_type,
                    is_nullable,
                    column_default,
                    character_maximum_length,
                    numeric_precision,
                    numeric_scale,
                    idx.index_name AS index_name
                FROM information_schema.key_column_usage const
                    JOIN information_schema.table_constraints tc
                        USING (table_catalog, table_schema, table_name, constraint_catalog, constraint_schema, constraint_name)
                    RIGHT JOIN information_schema.columns c USING (column_name, table_catalog, table_schema, table_name)
                    LEFT JOIN information_schema.constraint_column_usage ccu ON ccu.constraint_name = tc.constraint_name
                    LEFT JOIN (
                        SELECT
                            pi.indrelid::regclass AS table_name,
                            pcl.relname AS index_name,
                            pa.attname AS column_name,
                            pa.attnum AS column_number,
                            co.conname AS constraint_name
                        FROM pg_index pi
                        JOIN pg_class pcl ON pcl.oid = pi.indexrelid
                        JOIN pg_attribute pa ON pa.attnum = ANY(pi.indkey) AND pa.attrelid = pi.indrelid
                        LEFT JOIN pg_constraint co ON co.conindid = pi.indexrelid
                        WHERE co.conname IS NULL
                    ) idx ON (idx.table_name::varchar = c.table_name AND idx.column_name = c.column_name)
                WHERE c.table_catalog = $1
                AND c.table_name = $2
                AND c.table_schema = $3"""
        ret = await self.conn.execute_query_dict(sql, [self.database, table, self.schema])
        for row in ret:
            columns.append(
                Column(
                    name=row["column_name"],
                    data_type=row["data_type"],
                    null=row["is_nullable"] == "YES",
                    default=row["column_default"],
                    length=row["character_maximum_length"],
                    max_digits=row["numeric_precision"],
                    decimal_places=row["numeric_scale"],
                    comment=row["column_comment"],
                    pk=row["column_key"] == "PRIMARY KEY",
                    unique=row["column_key"] == "UNIQUE",
                    index=row["index_name"] is not None,
                    extra=None,
                )
            )
        return columns
