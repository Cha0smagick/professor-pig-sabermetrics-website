# Seguridad

## Versiones soportadas

Solo la rama `main` y el último commit publicado.

## Qué es y qué no es

- Web estática: no hay servidor propio, base de datos, login ni cookies.
- No se recogen datos personales ni se usan analytics.
- El navegador solo descarga los JSON del propio sitio.
- El motor Python se ejecuta en local para construir los artefactos.

## Secretos

- `config/`, `.env`, claves y tokens están en `.gitignore`.
- El historial del repositorio **antiguo** (`Sabermetria-futbol`) y la rama local `legacy-main` contienen claves API reales.
- Esas claves deben rotarse en los proveedores. No deben reutilizarse.
- El repositorio publicado (`professor-pig-sabermetrics-website`) se creó con un commit limpio y no contiene secretos.

## Cadena de suministro

- Las GitHub Actions del workflow están fijadas por SHA, no por tag móvil.
- Permisos del token: `contents: read`, `pages: write`, `id-token: write` en Pages; `contents: write` solo en `daily.yml` para commitear los dos JSONL de resultados.
- `daily.yml` no recibe secretos: consulta endpoints públicos (openfootball, OpenLigaDB, TheSportsDB, MLB, NHL, nflverse).
- La web no carga scripts, fuentes ni imágenes de terceros.
- `site/index.html` incluye una Content Security Policy restrictiva.
- `dependabot.yml` vigila dependencias Python y acciones.

## Superficie de ataque

- Entrada de usuario: solo números de cuota (se parsean como `Number` y se validan).
- Los nombres de equipos, ligas y árbitros se escapan antes de insertarse en HTML.
- No hay `eval`, `Function`, `document.write`, `innerHTML` con entrada no escapada ni peticiones a dominios externos.
- Los JSON son datos del repositorio; se tratan como datos, no como instrucciones.
- La actualización diaria descarga datos públicos; si una fuente falla o devuelve basura, se descarta y se registra.

## Reportar una vulnerabilidad

Abre un **GitHub Security Advisory** privado en el repositorio:

https://github.com/Cha0smagick/professor-pig-sabermetrics-website/security/advisories/new

Incluye reproducción, impacto y versión. No publiques detalles explotables en un issue público.
