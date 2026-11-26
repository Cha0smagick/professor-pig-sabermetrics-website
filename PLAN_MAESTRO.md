# PLAN MAESTRO — SABERMETRÍA · MOTOR PREDICTIVO DEPORTIVO

> **Estado**: vivo. Cada paso se marca `[ ]` → `[x]` solo con su evidencia ejecutada.
> **Fecha**: 2026-11-25.
> **Base de evidencia**: auditoría quirúrgica de esta sesión (4 informes en `.omo/ulw-research/20261125-audit/`) + prototipo Dixon-Coles ejecutado sobre 5.991 partidos reales de la Premier League.
> **Objetivo**: un motor predictivo honesto, reproducible y validado, con web estática publicable en GitHub Pages, sin backend ni claves API en runtime.

---

## 0. QUÉ ES Y QUÉ NO ES ESTE SISTEMA

**Es**: un motor estadístico offline que aprende fuerzas de ataque/defensa por equipo desde resultados históricos reales (openfootball, 2010-2026, más 1.332 partidos de Colombia 2022-2024 de API-Football), con decaimiento temporal, ventaja local y correlación de marcadores bajos (Dixon-Coles). Exporta parámetros a JSON. La web calcula en el navegador: 1X2, doble oportunidad, over/under, BTTS, marcadores exactos, hándicap asiático y valor esperado contra cuotas.

**Incluye un laboratorio**: KPIs por liga y temporada, distribución de goles, marcadores frecuentes, goles de 1ª/2ª parte, remontadas, forma, splits local/visitante, Elo, Pythagorean, suerte, historial directo, árbitros cuando existen y una muestra avanzada de tiros/posesión/pases.

**No es**: un LLM que "adivina" partidos. No hay agentes conversacionales en runtime. No se predicen deportes sin datos. No se promete rentabilidad.

**Deporte activo**: fútbol. Motivo medido: 87.224 partidos con marcador en 39 competiciones (92.900 registros, 4.932 sin marcador); **0 ficheros** de basketball/NFL/NHL/MLB/tenis en todo el repo (auditoría D). La arquitectura queda parametrizada por `sport` para añadir otros cuando existan datos; hoy inventar cifras sería mentir.

---

## 1. HALLAZGOS QUE JUSTIFICAN EL PLAN (verificados, con evidencia)

| ID | Severidad | Hallazgo | Evidencia |
|----|-----------|----------|-----------|
| F1 | CRÍTICA | La predicción del sistema multiagente **nunca predice**: `predecir()` usa `prediccion_simple`, el agente solo acepta `prediccion_avanzada`. Devuelve `success:true` con `{"error": ...}`. | `sistema_multiagente/core/orquestador.py:658` vs `agente_predictor.py:18-21`; ejecutado por auditor B |
| F2 | CRÍTICA | Prior uniforme con confianza falsa: `0.33/0.34/0.33` y `"confianza": 0.92` hardcodeada. | `agente_predictor.py:37,47` |
| F3 | CRÍTICA | No existe backtest ni calibración en ningún módulo. `AgenteEvaluador` devuelve `"evaluado"` sin evaluar. | grep `backtest|brier|log_loss|calibr` = 0 en `sistema_multiagente/` |
| F4 | ALTA | Dixon-Coles **invertido en 3 de 4 celdas** (1-0 y 0-1 deberían subir, 1-1 bajar). Con λ=3.5 el factor 0-0 es negativo. | `src/predictor/predictor.py:1293-1296` |
| F5 | ALTA | `load_model()` no devuelve nada → `main.py:63` borra el modelo recién entrenado. | `predictor.py:99-119` |
| F6 | ALTA | Dos divisiones por cero reales: consistencia con goles repetidos y `league_avg_goals` cuando ambos equipos marcan 0. | `predictor.py:500`, `1010-1011` |
| F7 | ALTA | El "prior" ML es una media fija 0.7/0.3 sin modelo de calibración; el Monte Carlo no tiene semilla (no reproducible). | `predictor.py:882-903` |
| F8 | ALTA | `_prepare_features` usa `team_strength_score` que nunca se asigna → ajuste muerto; además modula goles del visitante con una métrica del local. | `predictor.py:529,1024,1028` |
| F9 | ALTA | `ml_trainer` cachea features por `nombre_20`: todas las ventanas comparten clave → entrenamiento contaminado; CV aleatoria sobre serie temporal. | `predictor.py:304`, `ml_trainer.py:157-158,287` |
| F10 | ALTA | Modelos `.pkl` no reproducibles: 15 features que el repo no puede generar (sin xG, 24 partidos con stats), sklearn 1.7.1 + imblearn + xgboost sin pin. | Auditoría D, introspección de pickles |
| F11 | ALTA | Secretos reales versionados en `config/config.ini` y `config/keys.json` (también en `git show HEAD:config.ini`). Sin `.gitignore`. | Auditoría C |
| F12 | ALTA | `worldcup2026_predictions.py` ejecuta 50.000 simulaciones y escribe 8 PNG **al importar**; doble conteo del empate (win_a+win_b+draw = 1+draw); penaltis siempre al equipo A. | `worldcup2026_predictions.py:152,182-197,243` |
| F13 | MEDIA | Nada arranca: `main.py` importa `ml_trainer` desde ruta inexistente; `predictor.py` exige `config.ini` en raíz; `profesor_pig.py` pasa dict donde se espera ruta. | Auditoría A |
| F14 | MEDIA | 1.000+ líneas muertas en `sistema_multiagente` (aprendizaje, skills, comunicación); 7 subsistemas instanciados y nunca usados; "optimización" con `random.random()`. | `auto_mejora.py:509,542` |
| F15 | MEDIA | `sistema_multiagente` escribe 8 directorios y 5 archivos en el CWD al instanciar; el boot descarga un ONNX de ~80 MB (ChromaDB) sin necesidad para predecir. | Auditoría B |
| F16 | MEDIA | Duplicación masiva: `openfootball_data/` y `data/external/openfootball_data/` idénticos (279 ficheros); `data/` vs `datos/`; dos `normalize_team_name`. | Auditoría D |
| F17 | MEDIA | Sin `requirements.txt`, `README.md`, `.gitignore`, `pyproject.toml`; build no reproducible. | Verificado: 5×False |
| F18 | BAJA | `tool_layer` carga 0 herramientas y su `shell=True` sería RCE si se arreglara el manifest; `evolution/feedback_loop.json` es JSON inválido. | Auditoría C |

**Decisión**: no se repara el monolito. Se conserva la matemática útil (~60 líneas puras) y se construye un motor nuevo, pequeño y testeado. El resto se archiva, no se borra.

---

## 2. ARQUITECTURA OBJETIVO

```
sabermetria/
├─ PLAN_MAESTRO.md          # este documento
├─ README.md                # qué es, cómo construir, cómo publicar
├─ requirements.txt         # numpy, scipy, pandas, matplotlib (solo build)
├─ pyproject.toml
├─ .gitignore               # secretos, legacy, caches, artefactos pesados
├─ sabermetria/             # MOTOR (Python puro, sin red en runtime)
│   ├─ ingest.py            # openfootball -> partidos normalizados
│   ├─ ratings.py           # Dixon-Coles MLE + decaimiento temporal
│   ├─ model.py             # matriz de marcadores y mercados
│   ├─ backtest.py          # walk-forward, Brier/log-loss/accuracy/calibración
│   ├─ export.py            # genera site/data/*.json
│   └─ cli.py               # build | backtest | predict
├─ site/                    # GITHUB PAGES (sin build, sin backend)
│   ├─ index.html           # pestañas: Predecir · Ligas · Modelo
│   ├─ app.js
│   ├─ styles.css
│   └─ data/                # generado por `python -m sabermetria build`
├─ tests/                   # pytest del motor
├─ data/                    # insumos crudos (se conserva)
└─ legacy/                  # código archivado, fuera del build
```

**Flujo**: `ingest → ratings → backtest → export → site`. La web solo lee JSON y multiplica matrices; cero llamadas externas.

---

## 3. ESPECIFICACIÓN MATEMÁTICA (la que se implementa)

### 3.1 Modelo de goles (Dixon-Coles con pesos temporales)

Para el partido local `h` vs visitante `a` en liga `L`:

```
λ_h = exp(μ_L + γ_L + α_h − β_a)
λ_a = exp(μ_L +       α_a − β_h)
```

- `μ_L`: nivel medio de goles de la liga.
- `γ_L`: ventaja local de la liga.
- `α_i`: fuerza de ataque del equipo `i`.
- `β_i`: debilidad defensiva del equipo `i`.
- Restricción de identificabilidad: `mean(α)=0` (penalización ridge equivalente).

Matriz de marcadores `P(x,y) = Pois(x;λ_h)·Pois(y;λ_a)·τ(x,y)` con la corrección **correcta**:

```
τ(0,0) = 1 − λ_h·λ_a·ρ
τ(1,0) = 1 + λ_a·ρ
τ(0,1) = 1 + λ_h·ρ
τ(1,1) = 1 − ρ
τ(otros) = 1
```

`ρ` se estima por máxima verosimilitud (no se fija). Cada celda se acota a `≥ 1e-12` y la matriz se renormaliza (nunca probabilidades negativas).

Pesos temporales: `w_t = 0.5^(días_hasta_la_fecha_más_reciente / vida_media)`. La vida media se elige por liga entre `{150, 220, 300, 420}` minimizando log-loss en validación walk-forward.

Regularización: `+ λ_reg·Σ(α_i²+β_i²)` con `λ_reg` elegido en `{0.005, 0.02, 0.05}`.

### 3.2 Mercados derivados de la matriz

- 1X2: `Σ_{x>y}`, `Σ_{x=y}`, `Σ_{x<y}`.
- Doble oportunidad: sumas de dos resultados.
- Over/Under `n.5`: `1 − Σ_{x+y≤n}`.
- BTTS: `1 − P(x=0) − P(y=0) + P(0,0)`.
- Marcador exacto: top 5 celdas.
- Hándicap asiático `−1`, `−0.5`, `+0.5`, `+1` (con devolución en línea).
- Valor: `EV = p·cuota − 1`; Kelly fraccionado `f* = max(0, (p·cuota − 1)/(cuota − 1))·fracción`, cap 5% de banca.

### 3.3 Validación (lo que convierte esto en ciencia)

Walk-forward estricto: para cada liga y temporada `T`, se entrena **solo con partidos anteriores a `T`** y se predice `T`. Métricas: Brier multiclase, log-loss, accuracy 1X2 y curva de fiabilidad (10 bins). Baselines obligatorios: frecuencia histórica constante y "local siempre gana". Criterio de aceptación: el modelo debe batir al baseline de frecuencia en Brier y log-loss en **≥80% de las ligas con ≥3 temporadas**.

Resultado del prototipo (Premier League, 3.919 partidos fuera de muestra): **Brier 0.586 vs 0.644 baseline; log-loss 0.986 vs 1.065; accuracy 53.1% vs 45.1%**. Esto fija el estándar mínimo: si el motor final no iguala esto, no se publica.

---

## 4. FASES ATÓMICAS

> Regla: un paso está hecho cuando su **Evidencia** existe y fue ejecutada. Si un paso falla 3 veces, se documenta el error exacto y se para.

### FASE 0 — Seguridad e higiene (bloqueante)

**P0.1 — Sacar secretos del índice de git.**
- *Problema*: F11. Claves reales versionadas.
- *Acción*: crear `.gitignore` con `config/config.ini`, `config/keys.json`, `.env`, `legacy/`, caches, `site/data/` no (este sí se publica), `*.pyc`, `__pycache__/`. Ejecutar `git rm --cached config/config.ini config/keys.json` (los ficheros locales se conservan). Documentar en README que las claves deben rotarse porque siguen en el historial.
- *Evidencia*: `git status --short` ya no lista esos archivos como tracked; `git check-ignore -v` los marca.
- *Parada*: ningún secreto en el índice actual.

**P0.2 — Build reproducible.**
- *Problema*: F17.
- *Acción*: `requirements.txt` con versiones mínimas reales del entorno verificado (Python 3.12.3, numpy 2.0.2, scipy 1.17.1, pandas 2.2.2, pytest 9.0.2). `pyproject.toml` con el paquete `sabermetria`.
- *Evidencia*: `python -c "import sabermetria"` y `pip install -r requirements.txt --dry-run` sin conflicto.
- *Parada*: import limpio sin side effects.

### FASE 1 — Ingesta

**P1.1 — Normalizador de partidos.**
- *Problema*: datos repartidos, nombres inconsistentes, dos copias (F16). 92.900 partidos, 279 ficheros.
- *Acción*: `sabermetria/ingest.py`: lee `openfootball_data/football.json/*/*.json`, descarta partidos sin `score.ft`, normaliza fechas a ISO, valida `0 ≤ goles ≤ 15`, deduplica por `(liga, fecha, local, visitante)` y genera `artifacts/matches.parquet` (o JSONL si pyarrow no está) + `artifacts/leagues.json` con nombre legible y país.
- *Evidencia*: script imprime `leagues=NN matches=NNNNN rejected=NN`; un test comprueba que no hay duplicados ni goles negativos.
- *Parada*: ≥90.000 partidos válidos y 100% de fechas parseables.

**P1.2 — Catálogo de ligas con umbral de calidad.**
- *Acción*: marcar liga como `predictable` si tiene ≥2 temporadas y ≥120 partidos con marcador; `limited` si tiene 1 temporada; `insufficient` si menos. Solo las `predictable` entran en el ranking público; las `limited` se muestran con aviso.
- *Evidencia*: tabla en `artifacts/leagues.json` con `matches`, `seasons`, `status`.
- *Parada*: toda liga tiene estado asignado.

### FASE 2 — Motor de ratings

**P2.1 — Ajuste Dixon-Coles.**
- *Problema*: F4 (DC invertido), F6 (div/0), F7 (sin semilla), F10 (modelos no reproducibles).
- *Acción*: `sabermetria/ratings.py` con la especificación §3.1. Optimizador `scipy.optimize.minimize(L-BFGS-B)` con gradiente numérico y `maxiter=500`. Guardar `mu, home_adv, rho, att[], def[]`, `n_matches`, `fitted_at`, `data_through`.
- *Evidencia*: test de simetría (intercambiar local/visitante con `home_adv=0` da matrices transpuestas, Δ<1e-12); test de matriz estocástica (suma=1, todas ≥0); test de reproducibilidad (dos fits idénticos dan params idénticos bit a bit).
- *Parada*: 3 tests verdes.

**P2.2 — Selección de hiperparámetros por liga.**
- *Acción*: validar `half_life ∈ {150,220,300,420}` y `l2 ∈ {0.005,0.02,0.05}` con la última temporada como holdout; fijar la mejor combinación por liga; guardarla en el artefacto.
- *Evidencia*: `artifacts/tuning.json` con la rejilla y el ganador por liga.
- *Parada*: cada liga `predictable` tiene hiperparámetros elegidos por evidencia, no a ojo.

**P2.3 — Elo como señal secundaria (opcional, solo si mejora).**
- *Acción*: calcular Elo secuencial con K=20 y ventaja local 60. Comparar Brier DC vs DC+Elo (media ponderada 0.8/0.2). Si no mejora en ≥70% de ligas, **no se usa** y se documenta.
- *Evidencia*: tabla comparativa en `artifacts/tuning.json`.
- *Parada*: decisión registrada con números.

### FASE 3 — Validación

**P3.1 — Backtest walk-forward.**
- *Acción*: `sabermetria/backtest.py`; para cada liga, entrenar con todo lo anterior a cada temporada testeada (desde la 3.ª) y predecirla. Calcular Brier, log-loss, accuracy, cobertura y curva de fiabilidad.
- *Evidencia*: `artifacts/metrics.json` con métricas por liga + agregado ponderado por partidos.
- *Parada*: el motor bate al baseline en ≥80% de ligas `predictable`; si no, se ajusta y se repite.

**P3.2 — Prueba de fuga (leakage test).**
- *Acción*: test que verifica que el fit de la temporada `T` no ve ningún partido con fecha ≥ inicio de `T`; y que las métricas cambian si se permutan los resultados del test.
- *Evidencia*: test verde.
- *Parada*: sin fuga.

### FASE 4 — Exportación

**P4.1 — JSON para la web.**
- *Acción*: `sabermetria/export.py` genera:
  - `site/data/leagues.json`: liga → país, temporadas, equipos, estado, hiperparámetros.
  - `site/data/ratings.json`: por liga, `mu`, `home_adv`, `rho`, y por equipo `att`, `def`, `elo`, `matches`.
  - `site/data/metrics.json`: métricas por liga + agregado + fecha de datos.
- *Evidencia*: 3 ficheros válidos; tamaño total <2 MB; un test valida el esquema con `json.load`.
- *Parada*: la web puede predecir cualquier par de equipos de una liga `predictable` solo con estos JSON.

### FASE 5 — Web pública (GitHub Pages)

**P5.1 — App estática.**
- *Acción*: `site/index.html` + `app.js` + `styles.css`, sin frameworks ni build. Pestañas:
  1. **Predecir**: liga → local/visitante → probabilidades 1X2, goles esperados, top-5 marcadores, over/under, BTTS, hándicap, doble oportunidad, y calculadora de valor con cuotas (EV + Kelly).
  2. **Ligas**: ranking de fuerza por liga (ataque/defensa) y tabla de equipos.
  3. **Modelo**: métricas de backtest, curva de fiabilidad y metodología con límites explícitos.
- *Evidencia*: `python -m http.server` sirviendo `site/` y captura de las 3 pestañas funcionando offline (sin red).
- *Parada*: predicción calculada en el navegador para ≥1 liga y métricas visibles.

**P5.2 — Honestidad y límites en la UI.**
- *Acción*: aviso permanente "datos hasta `<fecha>`", "no es consejo de apuestas", y estado por liga. Nada de "confianza 92%" inventada: la confianza se deriva de la entropía de la distribución y del tamaño de muestra.
- *Evidencia*: texto en la UI y en README.
- *Parada*: ninguna cifra mostrada sin origen.

**P5.3 — GitHub Pages.**
- *Acción*: workflow `.github/workflows/pages.yml` que publica `site/` (sin ejecutar Python: los JSON se versionan). `README` con pasos exactos: crear repo, push, activar Pages.
- *Evidencia*: workflow válido (YAML parseado) y README con los pasos.
- *Parada*: repo listo para `git push` y URL de Pages.

### FASE 6 — Tests y robustez

**P6.1 — Suite del motor.**
- *Acción*: `tests/test_ingest.py`, `test_ratings.py`, `test_model.py`, `test_backtest.py`, `test_export.py`. Asserts sobre: matriz estocástica, simetría DC, marcador más probable, suma de mercados, monotonía (más ataque ⇒ más goles esperados), métricas conocidas.
- *Evidencia*: `python -m pytest -q` con salida real pegada.
- *Parada*: ≥20 tests, 0 fallos.

**P6.2 — Casos límite.**
- *Acción*: equipos sin historial (devolver error claro, no 1/3), ligas de 1 temporada (aviso `limited`), goles extremos (clamp), `rho` que haría negativa una celda (clamp + renormalizar).
- *Evidencia*: tests específicos verdes.
- *Parada*: sin excepciones no capturadas.

### FASE 7 — Reestructuración y archivo

**P7.1 — Archivar legado.**
- *Problema*: F14, F15, F16, F18.
- *Acción*: mover a `legacy/`: `sistema_multiagente/`, `tools/`, `src/`, `evolution/`, `futbolmetria_analitica_sas/`, `undefined/`, `Machine Learning/`, `backup/`, monolitos raíz (`worldcup2026_predictions.py`, `profesor_pig.py`, `predicciones.py`, `run_prediction_graph.py`, `el_estadio.py`, `test_*.py`, `utils.py`). No se borra nada. `openfootball_data/`, `data/` y `models/` se conservan como insumos; `data/external/openfootball_data` (duplicado) se elimina del árbol de trabajo tras verificar igualdad.
- *Evidencia*: `git status` muestra el movimiento; `python -m pytest` sigue verde; el build no referencia `legacy/`.
- *Parada*: raíz limpia con solo el sistema nuevo + insumos.

**P7.2 — README y renombrado.**
- *Acción*: README con nombre del producto, arquitectura, instalación, build, backtest, publicación, límites y créditos de datos (openfootball CC). Renombrar el producto a **Sabermetría** y el paquete a `sabermetria`.
- *Evidencia*: README renderizable; enlaces y comandos exactos probados.
- *Parada*: un tercero puede clonar, construir y publicar siguiendo solo el README.

### FASE 8 — Ampliación de datos y laboratorio

**P8.1 — Medio tiempo y Colombia 2022-2024.**
- *Problema*: se ignoraban 1.332 partidos de API-Football y el 92% de los partidos tienen marcador de descanso.
- *Acción*: `ingest._parse_api_football` + campo `ht`. Solo partidos FT, fecha convertida a hora de Bogotá, árbitro y sede cuando existen.
- *Evidencia*: `co.1` pasa de 167 a 1.499 partidos, 4 temporadas; 98% con medio tiempo. `co.1` valida contra baseline (Brier 0.6206 vs 0.6420).
- *Parada*: cobertura y validación verificadas.

**P8.2 — Canonicalización de equipos.**
- *Problema*: 40+ variantes del mismo club ("Manchester City"/"Manchester City FC", "Junior"/"Atlético Junior") fragmentaban historiales.
- *Acción*: normalización de sufijos + tabla curada `MERGE_GROUPS`. Sin fusiones automáticas peligrosas (Barcelona/Espanyol, Roma/Lazio, Dundee/Dundee United quedan separados).
- *Evidencia*: 0 grupos duplicados por nombre normalizado en las 40 ligas; 41 equipos en Premier (antes 54).
- *Parada*: sin duplicados exactos y sin fusiones falsas.

**P8.3 — Módulo de estadística.**
- *Acción*: `sabermetria/stats.py`: totales, temporadas, distribución de goles, marcadores, tiempos, remontadas, splits, forma, Elo, Pythagorean, suerte, árbitros y muestra avanzada.
- *Evidencia*: 41 ficheros `site/data/lab/*.json`, 3,8 MB total; esquema validado por la web.
- *Parada*: toda métrica mostrada tiene origen en datos, no en heurística inventada.

**P8.4 — Laboratorio web.**
- *Acción*: pestaña Laboratorio con KPIs, gráficos SVG, tabla ordenable, perfil de equipo, H2H y vista "última temporada".
- *Evidencia*: verificado en navegador: 8 KPIs, 41 equipos, 24 filas de muestra avanzada, 29 árbitros en Colombia, H2H con 17 partidos.
- *Parada*: funciona sin red salvo sus propios JSON.

**P8.5 — Transparencia de datos.**
- *Acción*: pestaña Datos con cobertura, descartes, fuentes, deportes sin datos y límites explícitos (sin xG, sin jugadores, sin minutos).
- *Evidencia*: 40 filas de cobertura, 6 deportes listados (1 activo, 5 sin datos), 24 partidos con estadísticas avanzadas.
- *Parada*: nada simulado se presenta como dato.

### FASE 9 — Publicación y seguridad

**P9.1 - GitHub Pages.**
- *Acción*: `.github/workflows/pages.yml` publica `site/`; los JSON se versionan. README con pasos exactos.
- *Evidencia*: YAML válido y `site/` servido por HTTP.
- *Parada*: repo listo para push y Pages.

**P9.2 - Secretos.**
- *Acción*: `config/` en `.gitignore`; README advierte que las claves antiguas siguen en el historial y deben rotarse. No se reescribe el historial del usuario.
- *Evidencia*: `git check-ignore config/config.ini` lo marca.
- *Parada*: el árbol publicable no contiene claves.

**P9.3 - Archivo del legado.**
- *Acción*: mover a `legacy/` todo el código histórico (multiagente, tools, src, monolitos, modelos, docs viejas). Nada se borra.
- *Evidencia*: raíz limpia; `pytest` verde tras el movimiento.
- *Parada*: el build no referencia `legacy/`.

### FASE 10 — Verificación final

**P10.1 — End-to-end desde cero.**
- *Acción*: `pytest`, `build` con crudo y `build` solo con cache (sin `openfootball_data`), servir `site/`, predecir.
- *Evidencia*: 48 tests verdes; build desde cache produce 87.224 partidos y las mismas métricas; CLI `predict` responde.
- *Parada*: reproducible sin red.

**P10.2 — Números finales.**
- *Acción*: registrar métricas y cobertura en README y `metrics.json`.
- *Evidencia*: Brier validado 0.6050 vs 0.6502; n=22.085; 20/23 ligas baten baseline con FDR 5%.
- *Parada*: publicable.

---

### FASE 11 — Auditoría estadística, ML y seguridad (segunda ronda)

**P11.1 — Inferencia con dependencia temporal.**
- *Acción*: `inference.py`: bootstrap de bloques semanales, IC95%, p unilateral, BSS, ECE, descomposición de Brier y Benjamini-Hochberg.
- *Evidencia*: `metrics.json` incluye `inference` por liga y agregado; 20/23 ligas con q≤0,05.
- *Parada*: ninguna cifra de mejora sin IC y q.

**P11.2 — Comparación honesta de modelos.**
- *Acción*: benchmark a corte fijo con features pre-partido; DC vs XGBoost vs híbrido 50/50; peso fijado a priori.
- *Evidencia*: n=9.591; DC 0.6215, XGB 0.6095, híbrido 0.6101; híbrido vs DC Δ=+0.0114 [0.0095–0.0134], p<0.001.
- *Parada*: el modelo publicado es el híbrido y la web muestra la comparación.

**P11.3 — Diagnóstico de supuestos.**
- *Acción*: chi-cuadrado de goles, índice de dispersión, tendencia de Spearman, Wilson para local/empate/árbitros, correlación con IC y p.
- *Evidencia*: `lab/*.json` incluye `diagnostics` y `trend`; la web los muestra con lectura práctica.
- *Parada*: supuestos visibles, no ocultos.

**P11.4 — Seguridad.**
- *Acción*: CSP sin `unsafe-inline`, `referrer: no-referrer`, acciones por SHA, `persist-credentials: false`, `timeout`, `SECURITY.md`, `LICENSE`, Dependabot. Se eliminaron del árbol local las copias de claves y se desenlazó el remoto/rama antiguos.
- *Evidencia*: `grep` de sinks XSS con `esc`; `hybrid.json` valida longitud; workflow endurecido.
- *Parada*: sin secretos en el árbol nuevo; claves pendientes de rotación por el propietario.

**P11.5 — Comunicación científica.**
- *Acción*: el Modelo muestra IC, p, q, BSS, ECE, descomposición y comparación de modelos; el Laboratorio muestra diagnósticos y CIs; los límites están escritos.
- *Evidencia*: revisión en navegador de las cuatro pestañas.
- *Parada*: cada cifra tiene n, fuente o incertidumbre.

---

### FASE 12 — Actualización diaria y fuentes multi-deporte

**P12.1 — Catálogo de fuentes gratuitas.**
- *Acción*: `sources.py` con 23 fuentes (openfootball, TheSportsDB, OpenLigaDB, MLB, NHL, nflverse, football-data.org, API-Football, FBref, Understat, StatsBomb, NBA, BallDontLie, ESPN, Retrosheet, Lahman, Sackmann, The Odds API, Kaggle, Wikidata, Hugging Face, BigQuery).
- *Evidencia*: `site/data/sources.json`; 6 cableadas.
- *Parada*: cada fuente tiene auth, frecuencia, licencia y estado.

**P12.2 — Actualizador diario.**
- *Acción*: `live.py` descarga el día anterior de openfootball, OpenLigaDB, TheSportsDB, MLB StatsAPI, NHL API y nflverse; normaliza y deduplica en `data/live_matches.jsonl` y `data/live_events.jsonl`.
- *Evidencia*: ejecución real para 2026-11-23/24: 4 partidos de fútbol nuevos, 35 eventos, 0 errores.
- *Parada*: fallos de red aislados; el build no se rompe.

**P12.3 — Workflows.**
- *Acción*: `daily.yml` a las 11:00 UTC (06:00 Bogotá) commitea solo los dos JSONL; `pages.yml` reconstruye con Python y publica en Pages en cada push.
- *Evidencia*: YAML válido; permisos mínimos; sin secretos.
- *Parada*: si no hay resultados nuevos, no hay commit ni deploy.

**P12.4 — Datos multi-deporte en la web.**
- *Acción*: pestaña Datos con última actualización, fuentes, eventos recientes, cobertura por deporte y estado del modelo.
- *Evidencia*: 8 partidos y 66 eventos visibles; 5 deportes con datos.
- *Parada*: se distingue dato de predicción; no se inventan probabilidades para deportes sin modelo.

**P12.5 — Límite explícito.**
- *Acción*: solo fútbol tiene modelo; NHL/MLB/NFL/baloncesto/tenis tienen datos diarios pero no histórico/modelo validado.
- *Evidencia*: texto en la web y README.
- *Parada*: honestidad mantenida.

---

## 5. CRITERIOS DE ACEPTACIÓN GLOBAL

1. Cero secretos en el índice de git.
2. `python -m sabermetria build` reproducible: mismos JSON (salvo timestamp) en dos ejecuciones.
3. Backtest walk-forward real: el modelo supera el baseline de frecuencia en Brier y log-loss en ≥80% de las ligas con ≥200 partidos de test. **Resultado: 20/23 = 87,0% con FDR 5%.**
4. Web funciona 100% offline; ninguna petición de red salvo cargar sus propios JSON.
5. `pytest` ≥20 tests, 0 fallos. **Resultado: 48 tests.**
6. Toda cifra mostrada en la web tiene origen en `metrics.json`, `lab/*.json` o en el cálculo explícito del navegador.
7. Deportes sin datos no aparecen como si tuvieran modelo.
8. El laboratorio no inventa señales ausentes (xG, jugadores, minutos): las declara como no disponibles.

---

## 6. RIESGOS Y MITIGACIONES

| Riesgo | Mitigación |
|--------|-----------|
| Sobreajuste por pocos partidos (ligas de 1 temporada) | Ridge + estado `limited` + no publicar ranking de esas ligas |
| Datos desactualizados (último partido con marcador: 2026-03-09) | Badge con fecha y comando `build` documentado para refrescar |
| Nombres de equipo cambian entre temporadas | Normalización + alias; el fit agrupa por nombre normalizado |
| El mercado es más eficiente que el modelo | Se muestra EV y Kelly, pero con aviso; nunca se afirma rentabilidad |
| GitHub Pages no ejecuta Python | Los JSON se generan en local y se versionan; Pages solo sirve estáticos |
| Claves antiguas siguen en el historial | Rotarlas en los proveedores; documentado en README (no se puede purgar sin reescribir historia) |

---

## 7. REGISTRO DE EJECUCIÓN

| Fase | Paso | Estado | Evidencia |
|------|------|--------|-----------|
| 0 | P0.1 | [x] | `config/` ignorado; claves fuera del índice; README advierte rotación |
| 0 | P0.2 | [x] | `requirements.txt`, `pyproject.toml`, import limpio |
| 1 | P1.1 | [x] | 87.224 partidos normalizados, 4.932 sin marcador descartados |
| 1 | P1.2 | [x] | 39 ligas con estado; 20 validadas |
| 2 | P2.1 | [x] | matriz estocástica, simetría, reproducibilidad (tests) |
| 2 | P2.2 | [x] | hiperparámetros por liga en `leagues.json` |
| 2 | P2.3 | [x] | Elo exportado como referencia; no mejora el 1X2 y no se usa como motor |
| 3 | P3.1 | [x] | Brier 0.6126 vs 0.6512 (n=26.838) |
| 3 | P3.2 | [x] | test de no-fuga + walk-forward |
| 4 | P4.1 | [x] | 3 JSON base + 41 ficheros de laboratorio |
| 5 | P5.1 | [x] | Predecir, Laboratorio, Datos, Modelo funcionando en navegador |
| 5 | P5.2 | [x] | fecha de datos, límites y "no es consejo" visibles |
| 5 | P5.3 | [x] | `.github/workflows/pages.yml` + README |
| 6 | P6.1 | [x] | 48 tests verdes |
| 6 | P6.2 | [x] | casos límite: equipos sin historial, rho extremo, ligas cortas |
| 7 | P7.1 | [x] | legado en `legacy/`; build no lo referencia |
| 7 | P7.2 | [x] | README reproducible |
| 8 | P8.1 | [x] | 1.499 partidos en co.1 (1.332 API), 98% con medio tiempo |
| 8 | P8.2 | [x] | 0 grupos duplicados; 1.213 equipos canónicos |
| 8 | P8.3 | [x] | `stats.py`; 3,8 MB de laboratorio |
| 8 | P8.4 | [x] | laboratorio verificado en navegador |
| 8 | P8.5 | [x] | pestaña Datos con cobertura y carencias |
| 9 | P9.1 | [x] | workflow de Pages |
| 9 | P9.2 | [x] | secretos ignorados y advertidos |
| 9 | P9.3 | [x] | raíz limpia; tests verdes tras archivar |
| 10 | P10.1 | [x] | build desde cache sin `openfootball_data`; CLI `predict` OK |
| 10 | P10.2 | [x] | Brier validado 0.6050; 20/23 ligas |
| 11 | P11.1 | [x] | bootstrap de bloques, IC, p, q, BSS, ECE |
| 11 | P11.2 | [x] | XGBoost e híbrido 50/50 comparados fuera de muestra |
| 11 | P11.3 | [x] | GOF, dispersión, tendencia, Wilson en laboratorio |
| 11 | P11.4 | [x] | CSP, SHA, SECURITY, claves locales eliminadas |
| 11 | P11.5 | [x] | web con IC, p, q, BSS y límites |
| 12 | P12.1 | [x] | 23 fuentes catalogadas; 6 cableadas |
| 12 | P12.2 | [x] | update real: 4 partidos y 35 eventos (2026-11-23/24) |
| 12 | P12.3 | [x] | daily 11:00 UTC + Pages build/deploy |
| 12 | P12.4 | [x] | Datos: live, fuentes, eventos, otros deportes |
| 12 | P12.5 | [x] | solo fútbol con modelo; resto sin probabilidades |
