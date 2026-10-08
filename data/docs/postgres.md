# PostgreSQL Documentation

Source: https://www.postgresql.org/docs/current/

## Connecting and Basic Queries
PostgreSQL uses standard SQL with advanced extensions:
- Connect via psql CLI: `psql -h localhost -U myuser -d mydb`
- List databases: `\l`
- List tables: `\dt`
- Describe table schema: `\d table_name`

## Indexing Strategies: B-Tree and GIN
PostgreSQL provides several index types:
- **B-Tree Index** (default): Ideal for equality and range queries (`<`, `<=`, `=`, `>=`, `>`).
  `CREATE INDEX idx_users_email ON users(email);`
- **GIN Index** (Generalized Inverted Index): Best for composite values, arrays, and JSONB document searches.
  `CREATE INDEX idx_products_tags ON products USING GIN(tags);`
- Multi-column and partial indexes reduce index footprint:
  `CREATE INDEX idx_active_users ON users(id) WHERE status = 'active';`

## Working with JSONB
PostgreSQL supports storing structured documents using the binary JSON type (`jsonb`):
- Insert JSON:
  `INSERT INTO orders (data) VALUES ('{"customer": "Alice", "items": [{"id": 1, "qty": 2}]}');`
- Query top-level key: `SELECT data->>'customer' FROM orders;`
- Check key existence: `SELECT * FROM orders WHERE data ? 'customer';`
- JSON path containment: `SELECT * FROM orders WHERE data @> '{"customer": "Alice"}';`

## Transactions and ACID Guarantees
PostgreSQL guarantees full ACID compliance through Multi-Version Concurrency Control (MVCC).
```sql
BEGIN;
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
UPDATE accounts SET balance = balance + 100 WHERE id = 2;
COMMIT;
```
If any error occurs before commit, execute `ROLLBACK;` to revert all modifications.

## Connection Pooling with PgBouncer
Because PostgreSQL forks a separate operating system process for each client connection, opening hundreds of direct database connections causes memory thrashing.
- Use **PgBouncer** as a lightweight connection pooler.
- Transaction pooling mode allows reusing server connections across clients immediately after each transaction completes.
