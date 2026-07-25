# Data governance

Dataset use is deny-by-default. `data/datasets_manifest.yaml` records provenance, license status,
authorized splits, expected hashes, and intended function before any download. Entries with an
unverified license or hash remain blocked. Test-only data must never be consumed by training code.

Raw data is immutable. Derived artifacts record source hashes. User secrets, sensitive inferred
attributes, and assistant-generated claims are never persisted as facts. Persistent personal data
requires explicit consent and keeps source, time, confidence, and supersession history.

## Conversaciones locales

RAMEM V1 almacena únicamente conversaciones creadas dentro de RAMEM. Los mensajes son la evidencia
original, no hechos derivados ni perfiles. El usuario puede borrar una sesión o toda la memoria;
SQLite y LanceDB deben quedar sin sus chunks. Las copias que el usuario haya creado quedan fuera del
alcance del borrado y deben documentarse al restaurar o exportar.

Las trazas admitidas contienen IDs, hashes, conteos, latencias y estados, nunca consultas, mensajes
ni texto recuperado. Los artefactos de benchmark no deben incluir conversaciones reales sin
consentimiento, anonimización y una finalidad declarada.

## Servicios de evaluación

Un evaluador LLM externo puede recibir consulta, respuesta y contexto completo. Por defecto solo se
usan datos públicos o sintéticos revisados. Para datos personales se requiere consentimiento y una
política aprobada de proveedor, región, retención y eliminación. Ragas debe ejecutarse con
`RAGAS_DO_NOT_TRACK=true`; desactivar analítica no evita que un proveedor LLM procese el contenido.
