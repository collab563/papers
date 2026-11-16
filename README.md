# Papers de matemáticas y cómputo exacto

Colección de manuscritos, código y datos de investigación de **Wendel León Alvarado**. El proyecto reúne resultados de combinatoria, geometría discreta, teoría de polinomios, física estadística e investigación operativa. La página estática de la colección está en [`index.html`](index.html).

## Artículos y materiales

- **Politopos de orden de coronas multicapa** — numeradores de Ehrhart, redes tensoriales, reconstrucción exacta y certificados de raíces. El manuscrito presenta resultados generales para $n=3$ y $n=4$, junto con cálculos para casos finitos mayores; la conjetura de que las raíces sean reales y negativas para todo $n \ge 5$ y las recurrencias universales correspondientes permanecen abiertas. [PDF en español](FORMATO%201/articulo_es.pdf) · [PDF en inglés](FORMATO%201/articulo.pdf) · [código y certificados](TENSORES%202D/).
- **Benchmark de Ehrhart hasta dimensión 40** — contracción tensorial cilíndrica y teorema chino del resto (CRT) para reconstruir datos de Ehrhart y volúmenes exactos. Incluye código, resultados y figuras. [PDF en español](BENCHMARK_D40/paper_tensor_crt_es.pdf) · [PDF en inglés](BENCHMARK_D40/paper_tensor_crt.pdf) · [datos JSON](BENCHMARK_D40/resultados_benchmark_d40.json).
- **Transferencia y raíces para $n=4$** — desarrollo detallado de la recurrencia de transferencia y del conteo de raíces, con certificado legible por máquina y scripts de auditoría. [PDF](PROOF_N4_TRANSFER/theorem_n4_proof.pdf) · [certificado](PROOF_N4_TRANSFER/certificado_teorema_n4.json).
- **Ceros de Lee–Yang y modelos SOS** — interpretación estadístico-mecánica de los polinomios, susceptibilidad autodual y brechas de ceros. El análisis distingue los resultados exactos de las comparaciones finitas y de la extensión condicional de la recurrencia para $n=4$. [PDF en español](LEE_YANG_PHYSICS/paper_lee_yang_physics_es.pdf) · [PDF en inglés](LEE_YANG_PHYSICS/paper_lee_yang_physics.pdf) · [código y resultados](LEE_YANG_PHYSICS/).
- **Planificación de pipelines cíclicos y garantías SLA** — conteo exacto y muestreo uniforme de planificaciones, con cotas de cola de Pitman–Chernoff y Hoeffding. Los benchmarks reportados llegan a 500&nbsp;000 tareas acopladas; las pruebas estadísticas son diagnósticos empíricos, mientras que la uniformidad se justifica mediante los pesos exactos de completación. [PDF](CYCLIC_PIPELINE_SLA/paper_cyclic_pipeline_sla.pdf) · [código y datos](CYCLIC_PIPELINE_SLA/).
- **Politopos de orden de treewidth acotado** — programación dinámica sobre descomposiciones de árbol y reconstrucción CRT. El manuscrito informa 12 instancias de tres familias, hasta dimensión 40. [PDF](TREEWIDTH_TENSOR_CRT/paper_treewidth_tensor_crt.pdf) · [código y resultados](TREEWIDTH_TENSOR_CRT/).
- **Exploración de Borcea–Brändén** — cálculos exactos para instancias finitas y un lema local relacionado con intercalación de raíces. Son exploraciones computacionales acotadas y no una demostración general. [Código y datos](BORCEA_BRANDEN_PROOF/).

## Cómo leer los resultados

Los manuscritos incluyen demostraciones propuestas, certificados algebraicos y resultados computacionales. Un certificado o una ejecución reproducible ayuda a revisar un argumento, pero por sí solo no garantiza que todas las afirmaciones matemáticas sean correctas. En cada artículo se señalan, cuando corresponde, las conjeturas abiertas, los resultados limitados a casos finitos y los argumentos que dependen de hipótesis.

Los cálculos usan, según el módulo, aritmética entera y racional exacta, reconstrucción CRT, polinomios simbólicos y aislamiento de raíces mediante sucesiones de Sturm; también hay análisis numérico y pruebas estadísticas donde se indica en los manuscritos. Los archivos JSON conservan datos y certificados para facilitar su inspección.

## Cómputo e inteligencia artificial

Estos trabajos y resultados se desarrollaron con cálculos por computadora y con asistencia de herramientas de inteligencia artificial. La IA se utilizó como apoyo durante el trabajo; su uso no constituye una verificación matemática independiente ni elimina la posibilidad de errores en los argumentos, el código o los datos. Por eso publico los manuscritos, programas y certificados e invito a que otras personas revisen las afirmaciones.

## Reproducibilidad

- Abre [`index.html`](index.html) para consultar la colección y los PDF desde la página local.
- El código de cada estudio está en su carpeta correspondiente. Las dependencias disponibles se indican en los archivos `requirements.txt` de los subproyectos; no hay un único entorno de Python común a todo el repositorio.
- Para los resultados de coronas, el punto de entrada es [`TENSORES 2D/main.py`](TENSORES%202D/main.py); las verificaciones y auditorías adicionales están en `PROOF_N4_TRANSFER/` y en los scripts `verify_*.py`.
- Los datos generados y certificados se distribuyen como JSON junto a los códigos y artículos. Las versiones exactas de Python, las bibliotecas, parámetros y límites de los experimentos se especifican en esos archivos o en los manuscritos cuando están disponibles.

## Invitación a revisión matemática

Si eres matemático/a y te interesa verificar rigurosamente las afirmaciones, demostraciones o certificados de estos artículos, agradeceré mucho que te comuniques conmigo. Puedes escribirme desde mi [perfil de GitHub](https://github.com/collab563) o abrir una [conversación en el repositorio](https://github.com/collab563/papers/issues). Cualquier observación que identifique el artículo y la afirmación correspondiente será bienvenida.

Repositorio: [github.com/collab563/papers](https://github.com/collab563/papers).
