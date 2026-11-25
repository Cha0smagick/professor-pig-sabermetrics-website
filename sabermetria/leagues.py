"""Catalogo de competiciones soportadas.

Solo fútbol: es el único deporte con datos históricos reales en el repositorio.
Para añadir otro deporte hace falta (a) un adaptador de datos y (b) una
distribución de marcador adecuada; el esquema de artefactos ya incluye `sport`.
"""

SPORTS = {
    "football": {"name": "Fútbol", "score_unit": "goles"},
}

# code: (pais, nombre, prioridad para la UI)
LEAGUES = {
    "en.1": ("Inglaterra", "Premier League", 1),
    "es.1": ("España", "LaLiga", 2),
    "it.1": ("Italia", "Serie A", 3),
    "de.1": ("Alemania", "Bundesliga", 4),
    "fr.1": ("Francia", "Ligue 1", 5),
    "pt.1": ("Portugal", "Primeira Liga", 10),
    "nl.1": ("Países Bajos", "Eredivisie", 11),
    "be.1": ("Bélgica", "Pro League", 12),
    "tr.1": ("Turquía", "Süper Lig", 13),
    "gr.1": ("Grecia", "Super League", 14),
    "sco.1": ("Escocia", "Premiership", 15),
    "at.1": ("Austria", "Bundesliga", 16),
    "ch.1": ("Suiza", "Super League", 17),
    "ru.1": ("Rusia", "Premier League", 18),
    "cz.1": ("Chequia", "First League", 19),
    "hu.1": ("Hungría", "NB I", 20),
    "en.2": ("Inglaterra", "Championship", 30),
    "es.2": ("España", "LaLiga 2", 31),
    "it.2": ("Italia", "Serie B", 32),
    "de.2": ("Alemania", "2. Bundesliga", 33),
    "fr.2": ("Francia", "Ligue 2", 34),
    "at.2": ("Austria", "Erste Liga", 35),
    "ch.2": ("Suiza", "Challenge League", 36),
    "ru.2": ("Rusia", "FNL", 37),
    "tr.2": ("Turquía", "1. Lig", 38),
    "en.3": ("Inglaterra", "League One", 40),
    "en.4": ("Inglaterra", "League Two", 41),
    "de.3": ("Alemania", "3. Liga", 42),
    "br.1": ("Brasil", "Serie A", 50),
    "br.2": ("Brasil", "Serie B", 51),
    "ar.1": ("Argentina", "Primera División", 52),
    "mx.1": ("México", "Liga MX", 53),
    "mls": ("Estados Unidos", "MLS", 54),
    "co.1": ("Colombia", "Primera A", 55),
    "jp.1": ("Japón", "J1 League", 56),
    "cn.1": ("China", "Super League", 57),
    "au.1": ("Australia", "A-League", 58),
    "dz.1": ("Argelia", "Ligue 1", 60),
    "eg.1": ("Egipto", "Premier League", 61),
    "ma.1": ("Marruecos", "Botola Pro", 62),
}

# Competiciones de copa presentes en los datos pero excluidas del modelo:
# eliminatorias a doble partido y finales en sede neutral rompen el supuesto
# de ventaja local constante. Se pueden añadir cuando el formato se modele.
EXCLUDED = {
    "at.cup": "copa nacional (eliminatorias)",
    "de.cup": "copa nacional (eliminatorias)",
    "copa.l": "copa continental (eliminatorias)",
    "uefa.cl": "copa continental (eliminatorias)",
}
