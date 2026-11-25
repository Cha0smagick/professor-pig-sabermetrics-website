# Sabermetría

Motor predictivo y laboratorio estadístico de fútbol. Aprende fuerzas de ataque y defensa por equipo desde resultados reales, valida cada temporada con walk-forward estricto y publica todo como una web estática: sin backend, sin claves API y sin llamadas externas en runtime.

## Estado actual

- **Datos hasta**: 2026-03-09 (último partido con marcador en la base).
- **Cobertura**: 87.460 partidos · 40 competiciones · 1.213 equipos · 19 temporadas.
- **Modelo validado** (ligas con ≥200 partidos de test, n=9.858):
  - Brier **0.6105** frente a 0.6489 del baseline.
  - Log-loss **1.0189**.
  - Acierto 1X2 **49.3%**.
  - **21 de 24** competiciones baten al baseline de frecuencias.
- **Medio tiempo**: 92% de los partidos (base para analizar tiempos y remontadas).
- **Extra**: 1.332 partidos de Colombia 2022-2024 desde API-Football (marcador, medio tiempo, árbitro y sede).

Las cifras se regeneran en `site/data/metrics.json` con cada build. No hay números escritos a mano en la web.

## Deportes disponibles

Solo **fútbol**. Es el único deporte con partidos en el repositorio: 0 registros de baloncesto, béisbol, NFL, NHL o tenis. La arquitectura está parametrizada por `sport` y el esquema exporta ese campo, pero inventar un modelo sin datos sería mentir. La pestaña **Datos** muestra el estado de cada deporte.

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

1. Normaliza los partidos (crudo si existe; si no, usa `data/matches.jsonl.gz`).
2. Ajusta Dixon-Coles por liga con decaimiento temporal.
3. Valida walk-forward y elige hiperparámetros por liga.
4. Exporta `site/data/*.json` y `site/data/lab/*.json`.

Opcional:

```powershell
python -m sabermetria ingest     # reconstruye la cache normalizada
python -m sabermetria backtest   # solo métricas
python -m sabermetria predict --league en.1 --home "Arsenal FC" --away "Chelsea FC"
```

## Ver en local

```powershell
python -m http.server 8000 -d site
```

Abre `http://localhost:8000`. No abras `index.html` con `file://`: el navegador bloquea la carga de los JSON.

## Publicar en GitHub Pages

El repositorio ya incluye `.github/workflows/pages.yml`, que publica `site/` sin ejecutar Python (los JSON van versionados).

```powershell
git init
git add .
git commit -m "Sabermetria: motor y laboratorio"
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
  stats.py              laboratorio: ligas, equipos, tiempos, árbitros
  export.py             artefactos JSON
  cli.py                build / ingest / backtest / predict
site/                   web estática (GitHub Pages)
  index.html app.js styles.css
  data/                 leagues, ratings, metrics, lab/
tests/                  pytest
data/                   cache normalizada y alias de equipos
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
- Calibración final: contracción al baseline de la liga, elegida en validación.

Todos los mercados salen de la misma matriz. Los hándicaps con línea entera modelan devolución.

## Validación

- Walk-forward: para predecir una temporada solo se entrena con partidos anteriores.
- Baselines: frecuencias históricas y "local siempre gana".
- Métricas: Brier multiclase, log-loss, acierto y curva de fiabilidad.
- Solo se publican predicciones de ligas que baten al baseline con al menos 200 partidos de test.
- Copas y eliminatorias quedan fuera: la ventaja local y los partidos a doble vuelta rompen el supuesto.

## Límites

- Sin xG, alineaciones, lesiones, sanciones, minutos de gol ni cuotas históricas.
- Los recién ascendidos arrancan con rating medio.
- El fútbol es ruidoso: un 60% falla 4 de cada 10 veces.
- Nada de esto es consejo financiero ni garantía de rentabilidad.
- La muestra avanzada (tiros/posesión) tiene 24 partidos: describe, no infiere.

## Seguridad

- `config/` está en `.gitignore`. Los ficheros locales `config/config.ini` y `config/keys.json` contenían claves API reales y siguen en el historial antiguo de git.
- **Rota esas claves en los proveedores** antes de publicar cualquier copia de este repositorio.
- No reutilices el historial antiguo: crea un repositorio nuevo con el árbol limpio.
- El motor y la web no leen claves ni hacen peticiones externas.

## Datos y licencia

- Resultados: [openfootball/football.json](https://github.com/openfootball/football.json), CC BY-SA.
- Muestra avanzada y Colombia 2022-2024: cache local de API-Football, sujeta a sus términos.
- Código de este repositorio: MIT.
