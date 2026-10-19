# Sql Indexes

A database index, usually a B-tree, lets the engine find rows without scanning the whole table, so lookups are much faster. The trade-off is that every insert, update and delete must also maintain the index, which slows writes. In a composite index the column order matters. Use EXPLAIN to check whether a query uses an index.
