# ADR 0001: máquina de estados explícita

- Estado: aceptado
- Alcance: orquestación del runtime

Usar una máquina de estados tipada y explícita en el núcleo en vez de un framework de agentes. Esto
mantiene medibles los límites entre módulos, hace reproducibles las trazas y permite ablaciones de
una sola variable. ADR-0004 concreta los componentes especializados usados por la implementación.
