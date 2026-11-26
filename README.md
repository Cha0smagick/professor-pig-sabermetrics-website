# Professor Pigskin's Sports Predictor

Motor predictivo y laboratorio estadístico de fútbol. Aprende fuerzas de ataque y defensa por equipo desde resultados reales, valida cada temporada con walk-forward estricto y publica todo como una web estática: sin backend, sin claves API y sin llamadas externas en runtime.

## Estado actual

- **Datos hasta**: 2026-11-24 (histórico base + actualización diaria).
- **Cobertura**: 87.224 partidos · 39 competiciones · 1.200 equipos · 19 temporadas.
- **Modelo publicado**: híbrido **50% Dixon-Coles + 50% XGBoost**, con el peso fijado a priori.
- **Walk-forward** (n=26.838, 5 temporadas por liga): Brier **0.6126** frente a 0.6512 del baseline; log-loss 1.0216; acierto 48.4%.
- **Subconjunto con evidencia** (20 de 23 ligas, FDR 5%): Brier **0.6050**, IC95% [0.6013–0.6091], BSS 7,0%, ECE 0,73%, n=22.085. Está seleccionado por resultado: no debe usarse como estimación insesgada.
- **Comparación a corte fijo** (n=9.592): DC 0.6153 · XGBoost 0.6096 · híbrido 0.6085 · baseline 0.6476. Híbrido frente a DC: Δ Brier +0.0069, IC95% [0.0054–0.0084], p=0.001. Híbrido frente a XGBoost: +0.0011, IC95% [−0.0004–0.0025], p=0.070 (empate estadístico).
- **Inferencia**: intervalos por bootstrap de bloques semanales, p unilateral, Benjamini-Hochberg, descomposición de Brier y ECE.
- **Medio tiempo**: 92% observado; los 0-0 finales se imputan como 0-0 al descanso (cobertura con imputación: 99,7%) (base para tiempos y remontadas).
- **Extra**: 1.332 partidos de Colombia 2022-2024 desde API-Football (marcador, medio tiempo, árbitro y sede).

Las cifras se regeneran en `site/data/metrics.json` con cada build. No hay números escritos a mano en la web.

## Actualización diaria

- Workflow `daily.yml`: todos los días a las **11:00 UTC = 06:00 America/Bogota** descarga los resultados del día anterior.
- Fuentes cableadas: openfootball, OpenLigaDB, TheSportsDB, MLB StatsAPI, NHL API y nflverse.
- Los partidos de fútbol se guardan en `data/live_matches.jsonl` y entran al modelo en el siguiente build.
- Los eventos de otros deportes se guardan en `data/live_events.jsonl` y se muestran en la pestaña Datos.
- El commit diario solo toca esos dos ficheros. `pages.yml` reconstruye y publica la web; si no hay resultados nuevos, no hay commit ni deploy.
- Manual: `python -m sabermetria update --days 2`.

## Fuentes gratuitas

El catálogo está en `sabermetria/sources.py` y se publica en `site/data/sources.json` (23 fuentes; 6 cableadas). Incluye openfootball, TheSportsDB, OpenLigaDB, MLB StatsAPI, NHL API, nflverse, football-data.org, API-Football, Football-Data.co.uk, FBref, Understat, StatsBomb Open Data, NBA Stats, BallDontLie, ESPN, Retrosheet, Lahman, Sackmann Tennis, The Odds API, Kaggle, Wikidata, Hugging Face y BigQuery público.

## Deportes disponibles

Solo **fútbol** tiene modelo predictivo publicado. Hay resultados diarios cableados de NHL, MLB, NFL y TheSportsDB (baloncesto, tenis, etc.), pero no histórico suficiente ni una distribución validada para esos deportes. La pestaña **Datos** muestra el estado y los últimos eventos. No se inventan probabilidades.

## Qué incluye la web

1. **Predecir**: probabilidades 1X2, goles esperados, marcadores más probables, doble oportunidad, over/under, BTTS, hándicap asiático, cuotas justas, valor esperado y Kelly contra cuotas propias. Incluye forma reciente e historial directo.
2. **Laboratorio**: KPIs por liga, reparto de resultados por temporada, distribución de goles, marcadores frecuentes, goles de 1ª/2ª parte, remontadas, tabla de equipos ordenable (histórico o última temporada), perfiles individuales (local/visitante, Pythagorean, suerte, Elo) e historial contra cualquier rival. Árbitros cuando la fuente los trae.
3. **Datos**: cobertura por competición, registros descartados, fuentes, muestra avanzada (tiros, posesión, pases) y lista explícita de lo que no existe.
4. **Modelo**: Brier, log-loss, acierto, baseline, calibración y límites.

## Instalación

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Requiere Python 3.10+. Probado con 3.12, numpy 2.0, scipy 1.17, pandas 2.2.

## Construir

```powershell
python -m sabermetria build
```

El comando:

1. Normaliza los partidos (crudo si existe; si no, usa `data/matches.jsonl.gz` + `data/live_matches.jsonl`).
2. Ajusta Dixon-Coles por liga con decaimiento temporal.
3. Valida walk-forward, elige hiperparámetros y calcula intervalos por bootstrap de bloques.
4. Entrena XGBoost con features pre-partido y compara DC, XGBoost e híbrido.
5. Exporta `site/data/*.json`, `site/data/hybrid.json`, `site/data/lab/*.json`, `site/data/sources.json` y `site/data/live.json`.

Opcional:

```powershell
python -m sabermetria update --days 2   # descarga resultados diarios
python -m sabermetria ingest            # reconstruye la cache normalizada
python -m sabermetria backtest          # solo métricas
python -m sabermetria predict --league en.1 --home "Arsenal FC" --away "Chelsea FC"
```

## Ver en local

```powershell
python -m http.server 8000 -d site
```

Abre `http://localhost:8000`. No abras `index.html` con `file://`: el navegador bloquea la carga de los JSON.

## Publicar en GitHub Pages

El repositorio incluye dos workflows:

- `daily.yml`: descarga resultados a las 06:00 de Bogotá y commitea solo `data/live_matches.jsonl` y `data/live_events.jsonl`.
- `pages.yml`: en cada push (incluido el commit diario) instala dependencias, reconstruye la web y publica `site/` en Pages.

```powershell
git add .
git commit -m "Professor Pigskin's: motor y laboratorio"
git branch -M main
git remote add origin https://github.com/<usuario>/<repo>.git
git push -u origin main
```

Después: **Settings → Pages → Source: GitHub Actions**. El workflow hace el resto.

> Si publicas desde este mismo repositorio, crea uno nuevo: el historial antiguo contiene claves API reales (ver Seguridad).

## Refrescar datos

La cache versionada permite construir sin red. Para actualizar:

```powershell
git clone https://github.com/openfootball/football.json openfootball_data/football.json
python -m sabermetria ingest
python -m sabermetria build
```

`openfootball_data/` está ignorado por git: solo se versiona la cache normalizada (`data/matches.jsonl.gz`, ~700 KB).

## Estructura

```
sabermetria/            motor Python puro
  ingest.py             openfootball + API-Football -> partidos normalizados
  ratings.py            Dixon-Coles, Elo, matriz de marcadores
  model.py              mercados y valor
  backtest.py           walk-forward, métricas, calibración
  inference.py          bootstrap de bloques, Brier decomposition, FDR, Wilson
  benchmark.py          XGBoost, comparación de modelos y matriz híbrida
  live.py               actualización diaria multi-fuente
  sources.py            catálogo de fuentes gratuitas
  stats.py              laboratorio: ligas, equipos, tiempos, árbitros
  export.py             artefactos JSON
  cli.py                build / ingest / update / backtest / predict
site/                   web estática (GitHub Pages)
  index.html app.js styles.css
  data/                 leagues, ratings, metrics, hybrid, sources, live, lab/
tests/                  pytest
data/                   cache normalizada, alias y resultados diarios
legacy/                 código histórico archivado (no se usa)
```

## Modelo

Para el partido local `h` contra el visitante `a`:

```
λ_h = exp(μ + γ + α_h − β_a)
λ_a = exp(μ + α_a − β_h)
P(x,y) = Pois(x; λ_h) · Pois(y; λ_a) · τ(x,y)
```

- `μ`: nivel medio de la liga.
- `γ`: ventaja local.
- `α`: fuerza de ataque.
- `β`: debilidad defensiva.
- `τ`: corrección Dixon-Coles en (0,0), (1,0), (0,1) y (1,1), con `ρ` estimado.
- Pesos temporales: `0.5^(días / semivida)`, semivida elegida por liga.
- Ridge sobre ataque y defensa.
- Calibración: contracción al baseline de la liga, elegida en validación.

El modelo publicado combina:

```
p_híbrido = 0.5 · p_DC + 0.5 · p_XGBoost
```

XGBoost usa solo información anterior al partido: Elo previo, forma de los últimos 5, goles recientes, descanso y experiencia. El peso 50/50 se fija a priori. La matriz de marcadores se recalibra para que su 1X2 coincida con la probabilidad híbrida, de modo que todos los mercados sigan siendo coherentes.

Todos los mercados salen de la misma matriz. Los hándicaps con línea entera modelan devolución.

## Validación e inferencia

- Walk-forward: para predecir una temporada solo se entrena con partidos anteriores.
- Baselines: frecuencias históricas y "local siempre gana".
- Métricas: Brier multiclase, log-loss, acierto y curva de fiabilidad.
- **Intervalos**: bootstrap de bloques semanales (no i.i.d.), 1.000 réplicas.
- **p-valor unilateral** para la mejora en Brier y **Benjamini-Hochberg** sobre las ligas comparadas (FDR 5%).
- **Brier Skill Score** y descomposición de Murphy (fiabilidad, resolución, incertidumbre).
- **ECE** y curva de fiabilidad con recuentos por bin.
- **Diagnóstico**: chi-cuadrado de goles frente a Poisson e índice de dispersión.
- Solo se publican predicciones de ligas con al menos 200 partidos de test y q≤0,05.
- Copas y eliminatorias quedan fuera: la ventaja local y los partidos a doble vuelta rompen el supuesto.

## Límites

- Sin xG, alineaciones, lesiones, sanciones, minutos de gol ni cuotas históricas.
- Los recién ascendidos arrancan con rating medio.
- El fútbol es ruidoso: un 60% falla 4 de cada 10 veces.
- Nada de esto es consejo financiero ni garantía de rentabilidad.
- La muestra avanzada (tiros/posesión) tiene 24 partidos: describe, no infiere.

## Seguridad

- `config/` está en `.gitignore`. Los ficheros locales `config/config.ini` y `config/keys.json` contenían claves API reales y **se han eliminado del árbol local**. Siguen en el historial del repositorio antiguo y en el remoto antiguo.
- **Rota esas 5 claves en los proveedores** (football-data.org, API-Sports/API-Football, RapidAPI, Scorebat, TheSportsDB). Están en historial público: asume copias.
- El repositorio nuevo se creó con un commit limpio. El remoto antiguo `Sabermetria-futbol` y la rama local `legacy-main` ya no están enlazados; aun así, borra o haz privado el repo antiguo.
- La web no lee claves ni hace peticiones externas. CSP restrictiva, `referrer: no-referrer`, sin cookies ni analytics.
- Las GitHub Actions están fijadas por SHA y con permisos mínimos.
- Consulta `SECURITY.md` para reportar vulnerabilidades.

## Límites estadísticos

- No se publican intervalos por parámetro de ataque/defensa: la dirección ataque+defensa no está identificada y un IC individual sería falsa precisión.
- El diagnóstico chi-cuadrado se vuelve muy sensible con miles de partidos; el índice de dispersión es la lectura práctica.
- La tabla histórica incluye equipos descendidos: es un histórico, no una foto actual.
- El test de tendencia es exploratorio (Spearman); no implica causalidad.
- La muestra avanzada (tiros/posesión) tiene 24 partidos: describe, no infiere.
- Los árbitros se muestran con mínimo 20 partidos y IC Wilson; es descriptivo, no causal.

## Datos y licencia

- Resultados: [openfootball/football.json](https://github.com/openfootball/football.json), CC BY-SA.
- Muestra avanzada y Colombia 2022-2024: cache local de API-Football, sujeta a sus términos.
- Código de este repositorio: MIT.
