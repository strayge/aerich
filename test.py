from pathlib import Path
from aerich.operations import RenameTable, read_operations, write_operations

operation = RenameTable(old_name="old", new_name="new")
path = Path('test.json')
write_operations(path, [operation], [])

ops_up, ops_down = read_operations(path)
assert ops_up == [operation], ops_up
assert ops_down == [], ops_down
