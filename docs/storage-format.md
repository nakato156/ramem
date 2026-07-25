# Formato de almacenamiento y recuperación

## Ubicación

La raíz predeterminada procede de `platformdirs.user_data_path("ramem", "ramem")`. Puede
sobrescribirse con `RAMEM_DATA_DIR`. Contiene:

```text
ramem.sqlite3
ramem.sqlite3-wal
ramem.sqlite3-shm
memory.lance/
models/ramem-gemma-1b/
traces/
prompt-history
```

No mover ni copiar la base mientras RAMEM está abierto. Para un respaldo coherente, cerrar el CLI y
copiar `ramem.sqlite3`; `memory.lance` puede omitirse porque se reconstruye.

## Esquema canónico

- `sessions`: UUID, título, timestamps y estado.
- `messages`: UUID, sesión, secuencia única, rol, contenido y SHA-256.
- `memory_chunks`: UUID, mensajes de origen JSON, texto, offsets, tokens, timestamp y SHA-256.
- `index_jobs`: operación `upsert|delete`, estado, intentos y error acotado.
- `schema_migrations`: versiones aplicadas.
- `memory_chunks_fts`: espejo lexical FTS5 del texto de chunks.

Mensajes y chunks son append-only; los triggers impiden `UPDATE`. El borrado de una sesión usa
`ON DELETE CASCADE`, limpia FTS y crea trabajos de borrado para LanceDB en una sola transacción.

## Restaurar

1. Cerrar RAMEM.
2. Restaurar `ramem.sqlite3` en una raíz vacía.
3. Ejecutar `ramem setup`.
4. Ejecutar `ramem memory rebuild`.
5. Confirmar que `chunks == lancedb_chunks` con `ramem memory stats`.

Una base con una versión de esquema superior se rechaza sin modificarla.

## Borrar

`ramem session delete UUID --yes` elimina una sesión. En el REPL, `/forget all` exige escribir
`BORRAR TODO`. RAMEM comprueba que SQLite y LanceDB no conserven los chunks objetivo. Las copias de
seguridad creadas por el usuario no se eliminan automáticamente.

Las trazas contienen hashes, IDs, conteos, latencias y resultados; no contienen consultas,
mensajes ni texto recuperado.
