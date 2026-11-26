/* Professor Pigskin's Sports Predictor — cálculo íntegro en el navegador. */
"use strict";

const MAXG = 15;
const LOGFACT = [0];
for (let i = 1; i <= MAXG; i++) LOGFACT[i] = LOGFACT[i - 1] + Math.log(i);

const $ = (id) => document.getElementById(id);
const pct = (value, digits = 1) => `${(value * 100).toFixed(digits)}%`;
const num = (value, digits = 2) => Number(value).toFixed(digits);
const esc = (text) => String(text).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const signed = (value, digits = 2) => `${value > 0 ? "+" : ""}${Number(value).toFixed(digits)}`;
const fmtP = (value) => value === null || value === undefined ? "—" : Number(value) < 0.001 ? "<0.001" : Number(value).toFixed(3);
const fmtCI = (ci, digits = 3) => ci ? `${Number(ci[0]).toFixed(digits)}–${Number(ci[1]).toFixed(digits)}` : "—";
const fmtPctCI = (ci) => ci ? `${pct(ci[0], 0)}–${pct(ci[1], 0)}` : "—";
const state = {
  league: null,
  home: null,
  away: null,
  neutral: false,
  odds: { home: null, draw: null, away: null },
  lab: null,
  labCode: null,
  labView: null,
  labViewCode: null,
  labCache: new Map(),
  labSort: { key: "pts", dir: -1 },
  labScope: "all",
  team: null,
  rival: null,
};
const DATA = { leagues: [], ratings: null, metrics: null, labIndex: null, hybrid: null, live: null, sources: null };

/* ---------- matemática del modelo ---------- */
const poisson = (k, lam) => Math.exp(-lam + k * Math.log(lam) - LOGFACT[k]);

function teamRow(league, name) {
  const row = league.teams.find((t) => t[0] === name);
  if (!row) throw new Error(`Equipo desconocido: ${name}`);
  return [row[1], row[2]];
}

function lambdas(league, home, away, neutral) {
  const [attH, defH] = teamRow(league, home);
  const [attA, defA] = teamRow(league, away);
  const venue = neutral ? 0 : league.home_adv;
  return [
    Math.min(Math.max(Math.exp(league.mu + venue + attH - defA), 0.05), 8),
    Math.min(Math.max(Math.exp(league.mu + attA - defH), 0.05), 8),
  ];
}

function normalizeMatrix(matrix) {
  let total = 0;
  for (const row of matrix) for (const value of row) total += Math.max(value, 1e-12);
  return matrix.map((row) => row.map((value) => Math.max(value, 1e-12) / total));
}

function rawMatrix(league, home, away, neutral) {
  const [lh, la] = lambdas(league, home, away, neutral);
  const m = [];
  for (let x = 0; x <= MAXG; x++) {
    m.push([]);
    for (let y = 0; y <= MAXG; y++) m[x].push(poisson(x, lh) * poisson(y, la));
  }
  m[0][0] *= 1 - lh * la * league.rho;
  m[1][0] *= 1 + la * league.rho;
  m[0][1] *= 1 + lh * league.rho;
  m[1][1] *= 1 - league.rho;
  return normalizeMatrix(m);
}

function outcomeProbs(matrix) {
  let home = 0, draw = 0, away = 0;
  for (let x = 0; x <= MAXG; x++) {
    for (let y = 0; y <= MAXG; y++) {
      if (x > y) home += matrix[x][y];
      else if (x === y) draw += matrix[x][y];
      else away += matrix[x][y];
    }
  }
  const total = home + draw + away;
  return { home: home / total, draw: draw / total, away: away / total };
}

function blendMatrix(matrix, league) {
  const alpha = league.alpha || 0;
  if (!alpha) return matrix;
  const current = outcomeProbs(matrix);
  const base = { home: league.base[0], draw: league.base[1], away: league.base[2] };
  const target = {};
  for (const key of ["home", "draw", "away"]) target[key] = (1 - alpha) * current[key] + alpha * base[key];
  return calibrateMatrix(matrix, target);
}

function calibrateMatrix(matrix, target) {
  const current = outcomeProbs(matrix);
  const factor = {
    home: current.home > 1e-9 ? target.home / current.home : 1,
    draw: current.draw > 1e-9 ? target.draw / current.draw : 1,
    away: current.away > 1e-9 ? target.away / current.away : 1,
  };
  const scaled = matrix.map((row, x) => row.map((value, y) => value * (x > y ? factor.home : x === y ? factor.draw : factor.away)));
  return normalizeMatrix(scaled);
}

function hybridProbs(code, home, away) {
  const hybrid = DATA.hybrid;
  if (!hybrid) return null;
  if (hybrid.generated_at && DATA.metrics && hybrid.generated_at < DATA.metrics.data_through) return null;
  const league = hybrid.leagues[code];
  if (!league) return null;
  if (!Array.isArray(league.teams) || league.teams.length !== league.n) return null;
  if (!Array.isArray(league.pairs) || league.pairs.length !== league.n * league.n * 3) return null;
  const hi = league.teams.indexOf(home);
  const ai = league.teams.indexOf(away);
  if (hi < 0 || ai < 0) return null;
  const index = (hi * league.n + ai) * 3;
  const values = [league.pairs[index], league.pairs[index + 1], league.pairs[index + 2]].map((value) => value / 1000);
  const total = values[0] + values[1] + values[2];
  if (!(total > 0)) return null;
  return { home: values[0] / total, draw: values[1] / total, away: values[2] / total };
}

function markets(matrix) {
  const probs = outcomeProbs(matrix);
  const dc = { "1X": probs.home + probs.draw, "12": probs.home + probs.away, "X2": probs.draw + probs.away };
  let goalsHome = 0, goalsAway = 0, btts = 0;
  const scores = [];
  const over = { 0.5: 0, 1.5: 0, 2.5: 0, 3.5: 0, 4.5: 0 };
  for (let x = 0; x <= MAXG; x++) {
    for (let y = 0; y <= MAXG; y++) {
      const p = matrix[x][y];
      goalsHome += x * p; goalsAway += y * p;
      if (x > 0 && y > 0) btts += p;
      scores.push([x, y, p]);
      for (const line of Object.keys(over)) if (x + y > Number(line)) over[line] += p;
    }
  }
  scores.sort((a, b) => b[2] - a[2]);
  const handicap = {};
  for (const line of [-1, -0.5, 0.5, 1]) {
    let win = 0, push = 0, lose = 0;
    for (let x = 0; x <= MAXG; x++)
      for (let y = 0; y <= MAXG; y++) {
        const d = x - y + line;
        if (d > 0) win += matrix[x][y]; else if (d === 0) push += matrix[x][y]; else lose += matrix[x][y];
      }
    handicap[line] = { win, push, lose };
  }
  const ordered = [probs.home, probs.draw, probs.away].sort((a, b) => b - a);
  return { probs, dc, btts, goalsHome, goalsAway, scores: scores.slice(0, 5), over, handicap, margin: ordered[0] - ordered[1] };
}

function valueRows(probs, odds) {
  const labels = { home: "Gana local", draw: "Empate", away: "Gana visitante" };
  return ["home", "draw", "away"].map((key) => {
    const price = odds[key];
    const p = probs[key];
    const ev = price ? p * price - 1 : null;
    const kelly = price && price > 1 ? Math.min(Math.max((p * price - 1) / (price - 1), 0) * 0.25, 0.05) : null;
    return { key, label: labels[key], p, fair: 1 / p, price, ev, kelly };
  });
}

/* ---------- helpers de datos ---------- */
const leagueByCode = (code) => DATA.ratings.leagues[code];
const metaByCode = (code) => DATA.leagues.find((item) => item.code === code);
const labMetaByCode = (code) => DATA.labIndex.leagues.find((item) => item.code === code);
const playableLeagues = () => {
  const latest = DATA.ratings.data_through;
  const cutoff = new Date(latest);
  cutoff.setDate(cutoff.getDate() - 450);
  return DATA.leagues.filter((item) => item.validated && leagueByCode(item.code) && item.teams >= 8 && item.last && new Date(item.last) >= cutoff);
};

function sortedTeams(code) {
  return leagueByCode(code).teams.slice().sort((a, b) => a[0].localeCompare(b[0], "es"));
}

function defaultPair(code) {
  const league = leagueByCode(code);
  const maxLast = league.teams.reduce((latest, row) => (row[4] > latest ? row[4] : latest), "");
  const cutoff = new Date(maxLast);
  cutoff.setDate(cutoff.getDate() - 400);
  let pool = league.teams.filter((row) => row[4] && new Date(row[4]) >= cutoff);
  if (pool.length < 2) pool = league.teams;
  const ranked = pool.slice().sort((a, b) => (b[1] - b[2]) - (a[1] - a[2]));
  return [ranked[0][0], ranked[1][0]];
}

function daysBetween(a, b) { return Math.abs(new Date(b) - new Date(a)) / 86400000; }

function fillLeagueSelect(select, list) {
  select.innerHTML = "";
  for (const league of list) {
    const option = document.createElement("option");
    option.value = league.code;
    option.textContent = `${league.country} · ${league.name}`;
    select.appendChild(option);
  }
}

function fillTeamSelect(select, code, selected) {
  const teams = sortedTeams(code);
  const latest = DATA.ratings.data_through;
  select.innerHTML = "";
  for (const row of teams) {
    const option = document.createElement("option");
    option.value = row[0];
    const stale = row[4] && daysBetween(row[4], latest) > 400;
    option.textContent = stale ? `${row[0]} · histórico` : row[0];
    select.appendChild(option);
  }
  if (selected && teams.some((t) => t[0] === selected)) select.value = selected;
}

async function fetchJSON(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`${path} ${response.status}`);
  return response.json();
}

async function loadLab(code) {
  if (state.labCache.has(code)) return state.labCache.get(code);
  const promise = fetchJSON(`data/lab/${code}.json`);
  state.labCache.set(code, promise);
  return promise;
}

/* ---------- pestañas ---------- */
function setupTabs() {
  const tabs = [
    ["tab-predict", "panel-predict", "#predecir"],
    ["tab-lab", "panel-lab", "#laboratorio"],
    ["tab-data", "panel-data", "#datos"],
    ["tab-model", "panel-model", "#modelo"],
  ];
  const buttons = tabs.map(([buttonId]) => $(buttonId));
  const activate = (tabId) => {
    for (const [buttonId, panelId] of tabs) {
      const selected = buttonId === tabId;
      $(buttonId).setAttribute("aria-selected", String(selected));
      $(buttonId).tabIndex = selected ? 0 : -1;
      $(panelId).hidden = !selected;
    }
  };
  for (const [buttonId, , hash] of tabs) {
    $(buttonId).addEventListener("click", () => {
      activate(buttonId);
      history.replaceState(null, "", hash);
    });
  }
  buttons.forEach((button) => { button.tabIndex = button.getAttribute("aria-selected") === "true" ? 0 : -1; });
  document.querySelector("nav.tabs").addEventListener("keydown", (event) => {
    if (event.key !== "ArrowRight" && event.key !== "ArrowLeft") return;
    const index = buttons.indexOf(document.activeElement);
    if (index < 0) return;
    event.preventDefault();
    const step = event.key === "ArrowRight" ? 1 : -1;
    const next = buttons[(index + step + buttons.length) % buttons.length];
    next.focus();
    next.click();
  });
  const fromHash = tabs.find(([, , hash]) => hash === location.hash);
  if (fromHash) activate(fromHash[0]);
  window.addEventListener("hashchange", () => {
    const match = tabs.find(([, , hash]) => hash === location.hash);
    if (match) activate(match[0]);
  });
}

/* ---------- predecir ---------- */
function setLeague(code, home, away) {
  state.league = code;
  const league = leagueByCode(code);
  const pair = defaultPair(code);
  state.home = home && league.teams.some((t) => t[0] === home) ? home : pair[0];
  state.away = away && league.teams.some((t) => t[0] === away) ? away : pair[1];
  $("league").value = code;
  fillTeamSelect($("home"), code, state.home);
  fillTeamSelect($("away"), code, state.away);
  state.home = $("home").value;
  state.away = $("away").value;
  if (state.home === state.away) {
    state.away = league.teams.find((t) => t[0] !== state.home)[0];
    $("away").value = state.away;
  }
  renderPrediction();
  syncContext();
}

function syncContext() {
  if (!state.league) return;
  $("contextTag").textContent = "cargando…";
  loadLab(state.league)
    .then((lab) => {
      if (state.league === lab.code) {
        state.lab = lab;
        state.labCode = lab.code;
        renderContext();
      }
    })
    .catch(() => {
      $("contextTag").textContent = "sin datos";
      $("contextBody").innerHTML = `<p class="note">No se pudo cargar el historial de la competición.</p>`;
    });
}

function renderPrediction() {
  const empty = $("predictEmpty");
  const result = $("predictResult");
  if (!state.league || !state.home || !state.away) {
    empty.hidden = false; result.hidden = true; return;
  }
  let matrix, stats, xgb;
  try {
    const league = leagueByCode(state.league);
    const dc = rawMatrix(league, state.home, state.away, state.neutral);
    xgb = state.neutral ? null : hybridProbs(state.league, state.home, state.away);
    matrix = xgb ? calibrateMatrix(dc, xgb) : blendMatrix(dc, league);
    stats = markets(matrix);
  } catch (error) {
    empty.hidden = false; result.hidden = true;
    empty.textContent = `No se pudo calcular: ${error.message}`;
    return;
  }
  empty.hidden = true; result.hidden = false;

  const meta = metaByCode(state.league);
  const league = leagueByCode(state.league);
  $("matchTitle").textContent = `${state.home} vs ${state.away}`;
  const mode = state.neutral ? "sede neutral · Dixon-Coles" : (xgb ? "híbrido DC+XGB 50/50" : "Dixon-Coles");
  $("leagueTag").textContent = `${meta.country} · ${meta.name} · ${mode}`;

  const probs = stats.probs;
  $("outcomes").innerHTML = [["Local", probs.home, "home"], ["Empate", probs.draw, "draw"], ["Visitante", probs.away, "away"]]
    .map(([label, value, key]) => `
      <div class="outcome ${key}">
        <span class="label">${label}</span>
        <strong>${pct(value)}</strong>
        <div class="bar"><i data-width="${(value * 100).toFixed(1)}"></i></div>
      </div>`).join("");
  document.querySelectorAll("#outcomes .bar i").forEach((bar) => { bar.style.width = `${bar.dataset.width}%`; });

  $("xg").textContent = `${num(stats.goalsHome)} — ${num(stats.goalsAway)}`;
  $("confidence").textContent = `${(stats.margin * 100).toFixed(1)} pts`;
  $("homeAdv").textContent = state.neutral ? "neutralizada" : `+${Math.round(100 * (Math.exp(league.home_adv) - 1))}% goles`;
  $("fairOdds").innerHTML = `<span>Cuota justa</span><b>1</b> ${num(1 / probs.home)}<b>X</b> ${num(1 / probs.draw)}<b>2</b> ${num(1 / probs.away)}`;
  $("confidenceNote").textContent = "Margen = diferencia entre el resultado más probable y el segundo. Un margen bajo indica un partido abierto.";

  $("scores").innerHTML = stats.scores.map(([x, y, p]) => `<li><span class="score">${x} — ${y}</span><span class="p">${pct(p)}</span></li>`).join("");
  $("doubleChance").innerHTML = Object.entries(stats.dc).map(([label, value]) => `<span class="chip"><b>${label}</b> ${pct(value)}</span>`).join("");
  $("btts").innerHTML = `<span class="chip"><b>Sí</b> ${pct(stats.btts)}</span><span class="chip"><b>No</b> ${pct(1 - stats.btts)}</span>`;

  $("ouBody").innerHTML = Object.entries(stats.over).map(([line, value]) => `<tr><td>${line}</td><td>${pct(value)}</td><td>${pct(1 - value)}</td><td>${num(1 / value)}</td></tr>`).join("");
  $("ahBody").innerHTML = ["-1", "-0.5", "0.5", "1"].map((line) => {
    const h = stats.handicap[line];
    const fair = h.win > 0 ? (1 - h.push) / h.win : null;
    return `<tr><td>${Number(line) > 0 ? "+" : ""}${line}</td><td>${pct(h.win)}</td><td>${pct(h.push)}</td><td>${pct(h.lose)}</td><td>${fair ? num(fair) : "—"}</td></tr>`;
  }).join("");

  renderValue(probs);
  renderContext();
}

function renderValue(probs) {
  $("valueBody").innerHTML = valueRows(probs, state.odds).map((row) => {
    const evClass = row.ev === null ? "" : row.ev > 0 ? "pos" : "neg";
    return `<tr><td>${row.label}</td><td>${pct(row.p)}</td><td>${num(row.fair)}</td><td>${row.price ? num(row.price) : "—"}</td><td class="${evClass}">${row.ev === null ? "—" : (row.ev > 0 ? "+" : "") + pct(row.ev)}</td><td>${row.kelly === null ? "—" : pct(row.kelly, 2)}</td></tr>`;
  }).join("");
}

function teamLab(lab, name) { return lab.teams.find((team) => team.name === name); }

function h2h(lab, team, rival, limit = 6) {
  const ti = lab.teamNames.indexOf(team);
  const ri = lab.teamNames.indexOf(rival);
  if (ti < 0 || ri < 0) return { rows: [], w: 0, d: 0, l: 0, gf: 0, ga: 0, n: 0 };
  const all = lab.matches.filter((m) => (m[1] === ti && m[2] === ri) || (m[1] === ri && m[2] === ti));
  let w = 0, d = 0, l = 0, gf = 0, ga = 0;
  for (const m of all) {
    const home = m[1] === ti;
    const mine = home ? m[3] : m[4];
    const theirs = home ? m[4] : m[3];
    gf += mine; ga += theirs;
    if (mine > theirs) w++; else if (mine === theirs) d++; else l++;
  }
  const rows = all.slice(-limit).reverse().map((m) => {
    const home = m[1] === ti;
    return { date: m[0], home: lab.teamNames[m[1]], away: lab.teamNames[m[2]], hg: m[3], ag: m[4], mine: home ? m[3] : m[4], theirs: home ? m[4] : m[3] };
  });
  return { rows, w, d, l, gf, ga, n: all.length };
}

function renderContext() {
  if (!state.lab || state.labCode !== state.league) {
    $("contextBody").innerHTML = `<p class="note">Cargando historial…</p>`;
    return;
  }
  const lab = state.lab;
  const home = teamLab(lab, state.home);
  const away = teamLab(lab, state.away);
  if (!home || !away) { $("contextBody").innerHTML = `<p class="note">Sin historial para este cruce.</p>`; return; }
  $("contextTag").textContent = `${lab.totals.matches} partidos · ${lab.teams.length} equipos`;
  const formRow = (team) => `
    <div class="ctx-team">
      <div><strong>${esc(team.name)}</strong><span class="note">${team.m} PJ · ${team.pts} pts · Elo ${team.elo}</span></div>
      <div class="form">${team.form.split("").map((r) => `<i class="${esc(r)}">${esc(formEs(r))}</i>`).join("") || "—"}</div>
      <div class="ctx-metrics">
        <span>Local ${team.home.pts} pts</span>
        <span>Visitante ${team.away.pts} pts</span>
        <span>+2.5 ${pct(team.over25_rate)}</span>
        <span>BTTS ${pct(team.btts_rate)}</span>
        <span>Suerte ${signed(team.luck, 1)}</span>
      </div>
    </div>`;
  const record = h2h(lab, state.home, state.away, 4);
  const rows = record.rows.map((r) => `<li><span class="h2h-date">${esc(r.date)}</span><span>${esc(r.home)}</span><b>${r.hg}–${r.ag}</b><span>${esc(r.away)}</span></li>`).join("");
  $("contextBody").innerHTML = `
    <div>
      <h3>Forma y perfil</h3>
      ${formRow(home)}${formRow(away)}
    </div>
    <div>
      <h3>Historial directo</h3>
      <p class="note">${record.n} partidos: ${record.w}V ${record.d}E ${record.l}D para ${esc(state.home)} · goles ${record.gf}–${record.ga}</p>
      <ul class="h2h">${rows || "<li class='note'>Sin enfrentamientos previos en la base.</li>"}</ul>
    </div>`;
}

/* ---------- laboratorio ---------- */
function scopedTeams(lab) { return state.labScope === "recent" && lab.recent ? lab.recent.teams : lab.teams; }
function scopedTotals(lab) { return state.labScope === "recent" && lab.recent ? lab.recent.totals : lab.totals; }

function renderLabKpis(lab) {
  const t = scopedTotals(lab);
  const items = [
    ["Partidos", t.matches, ""],
    ["Goles por partido", t.avg_goals, ""],
    ["Victoria local", pct(t.home_win), `IC95 ${fmtPctCI(t.home_win_ci)}`],
    ["Empate", pct(t.draw), `IC95 ${fmtPctCI(t.draw_ci)}`],
    ["Victoria visitante", pct(t.away_win), ""],
    ["Over 2.5", pct(t.over25), ""],
    ["Ambos marcan", pct(t.btts), ""],
    ["Remontadas", pct(t.comeback_rate), ""],
  ];
  $("labKpis").innerHTML = items.map(([label, value, note]) => `<div class="card metric"><span class="k">${label}</span><strong>${value}</strong>${note ? `<span class="note">${note}</span>` : ""}</div>`).join("");
  $("labScopeNote").textContent = state.labScope === "recent" && lab.recent
    ? `Vista: temporada ${lab.recent.season} · ${t.matches} partidos.`
    : `Vista: histórico completo · ${t.matches} partidos.`;
}

function renderSeasonChart(lab) {
  const seasons = lab.seasons;
  if (!seasons.length) { $("seasonChart").textContent = "Sin temporadas"; return; }
  const w = 520, h = 280, padL = 34, padB = 42, padT = 12;
  const bw = (w - padL - 10) / seasons.length;
  const bars = seasons.map((s, i) => {
    const x = padL + i * bw;
    const total = s.home_win + s.draw + s.away_win || 1;
    let y = h - padB;
    const seg = (value, cls) => {
      const height = (value / total) * (h - padB - padT);
      y -= height;
      return `<rect x="${x + 2}" y="${y}" width="${bw - 4}" height="${height}" class="${cls}"/>`;
    };
    return seg(s.home_win, "bar-home") + seg(s.draw, "bar-draw") + seg(s.away_win, "bar-away");
  }).join("");
  const labels = seasons.map((s, i) => {
    const every = seasons.length > 12 ? 3 : seasons.length > 8 ? 2 : 1;
    if (i % every !== 0 && i !== seasons.length - 1) return "";
    const x = padL + i * bw + bw / 2;
    const short = s.season.includes("-") ? s.season.slice(2).replace("-", "–") : s.season.slice(2);
    return `<text x="${x}" y="${h - padB + 14}" class="tick" text-anchor="middle">${esc(short)}</text>`;
  }).join("");
  $("seasonChart").innerHTML = `
    <svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Reparto de resultados por temporada">
      ${bars}
      <line x1="${padL}" y1="${h - padB}" x2="${w - 6}" y2="${h - padB}" class="axis"/>
      ${labels}
    </svg>
    <div class="legend"><span class="sw home"></span>Local<span class="sw draw"></span>Empate<span class="sw away"></span>Visitante</div>`;
}

function renderGoalsChart(lab) {
  const dist = lab.goals_distribution;
  const total = dist.reduce((a, b) => a + b, 0) || 1;
  const max = Math.max(...dist) || 1;
  const w = 520, h = 250, padL = 34, padB = 34, padT = 12;
  const bw = (w - padL - 10) / dist.length;
  const bars = dist.map((count, i) => {
    const height = (count / max) * (h - padB - padT);
    const x = padL + i * bw + 3;
    const y = h - padB - height;
    return `<rect x="${x}" y="${y}" width="${bw - 6}" height="${height}" class="bar-total"/><text x="${x + (bw - 6) / 2}" y="${y - 4}" class="tick" text-anchor="middle">${count}</text>`;
  }).join("");
  const labels = dist.map((_, i) => `<text x="${padL + i * bw + bw / 2}" y="${h - padB + 16}" class="tick" text-anchor="middle">${i === dist.length - 1 ? "8+" : i}</text>`).join("");
  $("goalsChart").innerHTML = `<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Distribución de goles">
    ${bars}${labels}
    <line x1="${padL}" y1="${h - padB}" x2="${w - 6}" y2="${h - padB}" class="axis"/>
  </svg><div class="legend"><span class="sw goals"></span>Partidos por total de goles (n=${total})</div>`;
}

function renderSeasonTable(lab) {
  $("seasonBody").innerHTML = lab.seasons.map((s) => `<tr>
    <td>${esc(s.season)}</td><td>${s.matches}</td><td>${num(s.avg_goals, 2)}</td>
    <td>${pct(s.home_win)}</td><td>${pct(s.draw)}</td><td>${pct(s.away_win)}</td>
    <td>${pct(s.over25)}</td><td>${pct(s.btts)}</td>
    <td>${num(s.avg_goals_1h, 2)}</td><td>${num(s.avg_goals_2h, 2)}</td>
  </tr>`).join("");
  const trend = lab.trend ? lab.trend.avg_goals : null;
  $("trendNote").textContent = trend && trend.rho !== null
    ? `Tendencia de goles por temporada: Spearman ρ = ${trend.rho}, p = ${fmtP(trend.p)} (${trend.n} temporadas). No implica causalidad.`
    : "Sin temporadas suficientes para estimar tendencia.";
  const t = lab.totals;
  $("timeStats").innerHTML = [
    ["Cobertura ht (observada)", pct(t.ht_observed_coverage)],
    ["Cobertura ht (con 0-0 imputado)", pct(t.ht_coverage)],
    ["0-0 imputados", t.ht_imputed ?? 0],
    ["Empate al descanso", pct(t.ht_draw)],
    ["Over 0.5 al descanso", pct(t.ht_over05)],
    ["Goles 1ª parte", num(t.avg_goals_1h, 2)],
    ["Goles 2ª parte", num(t.avg_goals_2h, 2)],
    ["Remontadas", `${t.comebacks} (${pct(t.comeback_rate)})`],
    ["Ventajas perdidas", `${t.leads_lost} (${pct(t.lead_lost_rate)})`],
    ["0-0", pct(t.nil_nil)],
  ].map(([k, v]) => `<div><span>${k}</span><strong>${v}</strong></div>`).join("");
  const total = lab.totals.matches || 1;
  $("scorelines").innerHTML = lab.scorelines.slice(0, 12).map(([hg, ag, count]) => `<span class="chip"><b>${hg}–${ag}</b> ${pct(count / total)}</span>`).join("");
}

function labTeamRows(lab) {
  const query = $("labSearch").value.trim().toLowerCase();
  const rows = scopedTeams(lab)
    .map((team) => ({ ...team, rank: 0 }))
    .filter((team) => !query || team.name.toLowerCase().includes(query));
  const key = state.labSort.key === "rank" ? "pts" : state.labSort.key;
  const dir = state.labSort.dir;
  rows.sort((a, b) => {
    if (key === "name") return a.name.localeCompare(b.name, "es") * dir;
    if (key === "form") return (a.form || "").localeCompare(b.form || "") * dir;
    return ((a[key] ?? 0) - (b[key] ?? 0)) * dir;
  });
  rows.forEach((row, index) => { row.rank = index + 1; });
  return rows;
}

function formEs(form) {
  return String(form || "").replace(/W/g, "V").replace(/D/g, "E").replace(/L/g, "D");
}

function renderLabTeams(lab) {
  const rows = labTeamRows(lab);
  $("labTeamCount").textContent = `${rows.length} de ${scopedTeams(lab).length} equipos`;
  const head = $("labTeamsTable").querySelector("thead");
  for (const th of head.querySelectorAll("th[data-sort]")) {
    const active = th.dataset.sort === state.labSort.key;
    th.setAttribute("aria-sort", active ? (state.labSort.dir === 1 ? "ascending" : "descending") : "none");
  }
  $("labTeams").innerHTML = rows.map((team) => `
    <tr>
      <td class="rank">${team.rank}</td>
      <td><button class="link" type="button" data-team="${esc(team.name)}">${esc(team.name)}</button></td>
      <td>${team.m}</td><td>${team.w}</td><td>${team.d}</td><td>${team.l}</td>
      <td>${team.gf}</td><td>${team.ga}</td><td class="${team.gd >= 0 ? "pos" : "neg"}">${signed(team.gd, 0)}</td>
      <td><strong>${team.pts}</strong></td><td>${num(team.ppg, 2)}</td>
      <td>${pct(team.over25_rate)}</td><td>${pct(team.btts_rate)}</td>
      <td>${team.elo}</td><td class="${team.luck >= 0 ? "pos" : "neg"}">${signed(team.luck, 1)}</td>
      <td><span class="form">${team.form.split("").map((r) => `<i class="${esc(r)}">${esc(formEs(r))}</i>`).join("")}</span></td>
    </tr>`).join("") || `<tr><td colspan="16" class="muted">Sin resultados</td></tr>`;
}

function renderTeamDetail(lab, name) {
  const scoped = scopedTeams(lab);
  const team = scoped.find((t) => t.name === name) || lab.teams.find((t) => t.name === name);
  const card = $("teamDetail");
  if (!team) { card.hidden = true; return; }
  const rivals = scoped.filter((t) => t.name !== name);
  if (!state.rival || !rivals.some((r) => r.name === state.rival)) state.rival = rivals[0]?.name;
  const record = state.rival ? h2h(lab, name, state.rival, 8) : { rows: [], w: 0, d: 0, l: 0, gf: 0, ga: 0, n: 0 };
  const rows = record.rows.map((r) => `<li><span class="h2h-date">${esc(r.date)}</span><span>${esc(r.home)}</span><b>${r.hg}–${r.ag}</b><span>${esc(r.away)}</span></li>`).join("");
  const inScope = scoped.some((t) => t.name === name);
  const scopeLabel = inScope && state.labScope === "recent" && lab.recent ? `Temporada ${lab.recent.season}` : "Histórico completo";
  card.hidden = false;
  card.innerHTML = `
    <div class="card-head"><h2>${esc(team.name)}</h2><span class="tag">${esc(scopeLabel)} · Elo ${team.elo} · ${team.m} PJ</span></div>
    <div class="grid four">
      <div class="metric"><span class="k">Puntos</span><strong>${team.pts}</strong><span class="note">${num(team.ppg, 2)} por partido</span></div>
      <div class="metric"><span class="k">Goles</span><strong>${team.gf}–${team.ga}</strong><span class="note">DG ${signed(team.gd, 0)}</span></div>
      <div class="metric"><span class="k">Pythagorean</span><strong>${num(team.pyth, 1)}</strong><span class="note">Suerte ${signed(team.luck, 1)} pts</span></div>
      <div class="metric"><span class="k">Forma</span><strong>${esc(formEs(team.streak) || "—")}</strong><span class="note">${esc(team.form ? formEs(team.form) : "sin partidos")}</span></div>
    </div>
    <div class="grid two">
      <div>
        <h3>Local / visitante</h3>
        <table class="mini"><thead><tr><th></th><th>PJ</th><th>G</th><th>E</th><th>P</th><th>GF</th><th>GC</th><th>Pts</th></tr></thead><tbody>
          ${["home", "away"].map((side) => `<tr><td>${side === "home" ? "Local" : "Visitante"}</td><td>${team[side].m}</td><td>${team[side].w}</td><td>${team[side].d}</td><td>${team[side].l}</td><td>${team[side].gf}</td><td>${team[side].ga}</td><td>${team[side].pts}</td></tr>`).join("")}
        </tbody></table>
        <h3>Perfil</h3>
        <div class="chips">
          <span class="chip"><b>+2.5</b> ${pct(team.over25_rate)}</span>
          <span class="chip"><b>BTTS</b> ${pct(team.btts_rate)}</span>
          <span class="chip"><b>Portería a cero</b> ${pct(team.cs_rate)}</span>
          <span class="chip"><b>Sin marcar</b> ${pct(team.fts_rate)}</span>
          <span class="chip"><b>Remontadas</b> ${pct(team.comeback_rate)}</span>
          <span class="chip"><b>Colapsos</b> ${pct(team.collapse_rate)}</span>
          <span class="chip"><b>1ª parte</b> ${team.gf1h}–${team.ga1h}</span>
          <span class="chip"><b>2ª parte</b> ${team.gf2h}–${team.ga2h}</span>
        </div>
        <p class="note">Pythagorean = puntos esperados por goles a favor y en contra (exponente 1.7). Suerte = puntos reales − esperados.</p>
      </div>
      <div>
        <h3>Historial contra</h3>
        <div class="field"><select id="rivalSelect">${rivals.map((r) => `<option value="${esc(r.name)}" ${r.name === state.rival ? "selected" : ""}>${esc(r.name)}</option>`).join("")}</select></div>
        <p class="note">${record.n} partidos: ${record.w}V ${record.d}E ${record.l}D · goles ${record.gf}–${record.ga}</p>
        <ul class="h2h">${rows || "<li class='note'>Sin enfrentamientos.</li>"}</ul>
      </div>
    </div>`;
  card.querySelectorAll("th").forEach((th) => { th.scope = "col"; });
  $("rivalSelect").addEventListener("change", (event) => {
    state.rival = event.target.value;
    renderTeamDetail(lab, name);
  });
}

function renderReferees(lab) {
  const card = $("refCard");
  if (!lab.referees || !lab.referees.length) { card.hidden = true; return; }
  card.hidden = false;
  $("refBody").innerHTML = lab.referees.map((ref) => `<tr>
    <td>${esc(ref.name)}</td><td>${ref.matches}</td><td>${num(ref.avg_goals, 2)}</td>
    <td>${pct(ref.home_win)} <span class="note">[${fmtPctCI(ref.home_win_ci)}]</span></td>
    <td>${pct(ref.draw)}</td>
    <td>${pct(ref.over25)} <span class="note">[${fmtPctCI(ref.over25_ci)}]</span></td>
  </tr>`).join("");
}

function renderDiagnostics(lab) {
  const diagnostics = lab.diagnostics;
  if (!diagnostics) {
    $("diagTag").textContent = "sin datos";
    $("diagnostics").innerHTML = `<p class="note">No hay diagnóstico disponible.</p>`;
    return;
  }
  const gof = diagnostics.goals_gof || {};
  const pearson = diagnostics.pearson_dispersion || {};
  const ratios = diagnostics.goals_ratios || [];
  const dispersion = diagnostics.dispersion_total || {};
  $("diagTag").textContent = "descriptivo";
  $("diagTag").className = "tag";
  $("diagnostics").innerHTML = [
    ["Chi-cuadrado", gof.chi2 ?? "—"],
    ["Grados de libertad", gof.df ?? "—"],
    ["p (no ajustada por parámetros)", fmtP(gof.p)],
    ["Dispersión Pearson", pearson.ratio ?? "—"],
    ["1 gol: observado/esperado", ratios[1] ?? "—"],
    ["8+ goles: observado/esperado", ratios[8] ?? "—"],
    ["Varianza / media", dispersion.index ?? "—"],
    ["Media de goles", dispersion.mean ?? "—"],
  ].map(([k, v]) => `<div><span>${k}</span><strong>${v}</strong></div>`).join("");
}

function renderLab(lab) {
  const meta = labMetaByCode(lab.code);
  $("labTag").textContent = meta && meta.validated ? "liga validada" : "sin ventaja demostrada";
  $("labTag").className = `tag ${meta && meta.validated ? "ok" : "warn"}`;
  renderLabKpis(lab);
  renderSeasonChart(lab);
  renderGoalsChart(lab);
  renderSeasonTable(lab);
  renderLabTeams(lab);
  renderReferees(lab);
  renderDiagnostics(lab);
  if (state.team && scopedTeams(lab).some((t) => t.name === state.team)) renderTeamDetail(lab, state.team);
  else if (state.team && lab.teams.some((t) => t.name === state.team)) renderTeamDetail(lab, state.team);
  else { state.team = null; $("teamDetail").hidden = true; }
}

async function setLabLeague(code) {
  $("labTag").textContent = "cargando…";
  $("labTag").className = "tag";
  try {
    const lab = await loadLab(code);
    const recentOption = $("labScope").querySelector('option[value="recent"]');
    recentOption.textContent = lab.recent ? `Temporada ${lab.recent.season}` : "Última temporada";
    state.labView = lab;
    state.labViewCode = code;
    renderLab(lab);
  } catch (error) {
    $("labTag").textContent = "error";
    $("labKpis").innerHTML = `<p class="note">No se pudo cargar ${esc(code)}: ${esc(error.message)}</p>`;
  }
}

function setupLab() {
  const list = DATA.labIndex.leagues.slice().sort((a, b) => a.priority - b.priority);
  fillLeagueSelect($("labLeague"), list);
  $("labLeague").addEventListener("change", () => {
    state.team = null; state.rival = null;
    setLabLeague($("labLeague").value);
  });
  $("labSearch").addEventListener("input", () => {
    if (state.labView) renderLabTeams(state.labView);
  });
  $("labScope").addEventListener("change", () => {
    state.labScope = $("labScope").value;
    state.rival = null;
    if (state.labView) {
      renderLabKpis(state.labView);
      renderLabTeams(state.labView);
      if (state.team) renderTeamDetail(state.labView, state.team);
    }
  });
  const table = $("labTeamsTable");
  const head = table.querySelector("thead");
  for (const th of table.querySelectorAll("th")) {
    th.scope = "col";
    if (th.dataset.sort) th.tabIndex = 0;
  }
  head.addEventListener("click", (event) => {
    const th = event.target.closest("th[data-sort]");
    if (!th) return;
    const key = th.dataset.sort;
    if (state.labSort.key === key) state.labSort.dir *= -1;
    else state.labSort = { key, dir: key === "name" ? 1 : -1 };
    if (state.labView) renderLabTeams(state.labView);
  });
  head.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" && event.key !== " ") return;
    const th = event.target.closest("th[data-sort]");
    if (!th) return;
    event.preventDefault();
    th.click();
  });
  $("labTeams").addEventListener("click", (event) => {
    const button = event.target.closest("button[data-team]");
    if (!button || !state.labView) return;
    state.team = button.dataset.team;
    state.rival = null;
    renderTeamDetail(state.labView, state.team);
    $("teamDetail").scrollIntoView({ behavior: "smooth", block: "start" });
  });
  if (list.length) setLabLeague(list[0].code);
}

/* ---------- datos ---------- */
function renderData() {
  const index = DATA.labIndex;
  const live = DATA.live;
  const summary = (live && live.summary) || {};
  const liveSports = summary.sports || {};
  $("sportsGrid").innerHTML = index.sports.map((sport) => {
    const count = liveSports[String(sport.name).toLowerCase()] || 0;
    const active = sport.status === "active";
    return `<div class="card metric">
      <span class="k">${esc(sport.name)}</span>
      <strong class="${active ? "pos" : count ? "warn" : "muted"}">${active ? "activo" : count ? "datos diarios" : "sin datos"}</strong>
      <span class="note">${active ? esc(sport.reason) : count ? `${count} eventos descargados; modelo pendiente` : esc(sport.reason)}</span>
    </div>`;
  }).join("");

  $("liveTag").textContent = live && live.generated_at ? `actualizado ${live.generated_at}` : "sin actualización";
  $("liveSummary").innerHTML = [
    ["Partidos de fútbol en vivo", summary.matches ?? 0],
    ["Eventos de otros deportes", summary.events ?? 0],
    ["Último partido", summary.last_match_date ?? "—"],
    ["Último evento", summary.last_event_date ?? "—"],
    ["Fuentes usadas", Object.keys(summary.sources || {}).length],
    ["Deportes con datos", Object.keys(liveSports).length],
  ].map(([label, value]) => `<div><span>${label}</span><strong>${value}</strong></div>`).join("");

  const other = index.sports.filter((sport) => sport.status !== "active");
  $("otherSports").innerHTML = other.map((sport) => {
    const count = liveSports[String(sport.name).toLowerCase()] || 0;
    return `<div><span>${esc(sport.name)}</span><strong class="${count ? "pos" : "muted"}">${count ? `${count} eventos` : "sin datos"}</strong></div>`;
  }).join("");

  const catalog = (DATA.sources && DATA.sources.sources) || (live && live.sources) || [];
  $("sourcesCount").textContent = `${catalog.length} fuentes catalogadas`;
  $("sourcesBody").innerHTML = catalog.map((source) => {
    const status = source.status === "wired" ? ["cableada", "ok"] : source.status === "available" ? ["disponible", "warn"] : ["manual", ""];
    return `<tr>
      <td><a href="${esc(source.url)}" target="_blank" rel="noopener noreferrer">${esc(source.name)}</a></td>
      <td>${esc((source.sports || []).join(", "))}</td>
      <td>${esc(source.auth)}</td>
      <td>${esc(source.cadence)}</td>
      <td>${esc(source.license)}</td>
      <td><span class="tag ${status[1]}">${status[0]}</span></td>
    </tr>`;
  }).join("");

  const latest = summary.latest || [];
  $("liveEvents").innerHTML = latest.map((row) => `<tr>
    <td>${esc(row.date)}</td><td>${esc(row.sport)}</td><td>${esc(row.league)}</td>
    <td>${esc(row.home)}</td><td>${esc(row.away)}</td>
    <td>${row.hg ?? "—"}–${row.ag ?? "—"}</td><td>${esc(row.source)}</td>
  </tr>`).join("") || `<tr><td colspan="7" class="muted">Sin eventos descargados todavía.</td></tr>`;

  const t = index.totals;
  $("dataKpis").innerHTML = [
    ["Partidos", t.matches.toLocaleString("es-ES")],
    ["Competiciones", t.leagues],
    ["Equipos", t.teams],
    ["Goles", t.goals.toLocaleString("es-ES")],
    ["Media", num(t.avg_goals, 2)],
    ["Medio tiempo", pct(t.ht_coverage)],
    ["Over 2.5", pct(t.over25)],
    ["Remontadas", pct(t.comeback_rate)],
  ].map(([label, value]) => `<div class="card metric"><span class="k">${label}</span><strong>${value}</strong></div>`).join("");

  $("coverageBody").innerHTML = index.leagues.map((league) => {
    const status = { predictable: "suficiente", limited: "muestra corta", insufficient: "insuficiente" }[league.status] || "—";
    return `<tr>
      <td>${esc(league.country)} · ${esc(league.name)}</td>
      <td>${status}</td><td>${league.seasons}</td><td>${league.matches}</td><td>${league.teams}</td>
      <td>${esc(league.first)}</td><td>${esc(league.last)}</td><td>${pct(league.ht_coverage)}</td>
      <td><span class="tag ${league.validated ? "ok" : "warn"}">${league.validated ? "validado" : "no"}</span></td>
    </tr>`;
  }).join("");

  const rejected = index.data_quality.rejected || {};
  const rejectedRows = Object.entries(rejected).map(([key, value]) => `<tr><td>${esc(key)}</td><td>${value}</td></tr>`).join("");
  $("quality").innerHTML = `
    <div class="meta-list">
      <div><span>Fuente principal</span><strong>openfootball</strong></div>
      <div><span>Fuente extra</span><strong>API-Football</strong></div>
      <div><span>Competiciones</span><strong>${t.leagues}</strong></div>
      <div><span>Partidos válidos</span><strong>${t.matches.toLocaleString("es-ES")}</strong></div>
    </div>
    <h3>Registros descartados</h3>
    <table class="mini"><thead><tr><th scope="col">Motivo</th><th scope="col">N</th></tr></thead><tbody>${rejectedRows}</tbody></table>
    <p class="note">Se descartan partidos sin marcador final, con goles fuera de rango, fechas inválidas o duplicados.</p>`;

  const advanced = index.data_quality.advanced_sample || { n: 0, rows: [] };
  $("advancedTag").textContent = `n=${advanced.n}`;
  const correlationText = advanced.correlation_sot_goals === null || advanced.correlation_sot_goals === undefined
    ? "sin datos"
    : `${advanced.correlation_sot_goals} (IC95% ${fmtCI(advanced.correlation_ci, 3)}; p=${fmtP(advanced.correlation_p)})`;
  $("advancedSummary").innerHTML = `
    <div class="meta-list">
      <div><span>Partidos con estadísticas</span><strong>${advanced.n}</strong></div>
      <div><span>Correlación tiros a puerta − goles</span><strong>${correlationText}</strong></div>
      <div><span>xG disponible</span><strong>no</strong></div>
      <div><span>Lectura</span><strong>${advanced.correlation_p !== null && advanced.correlation_p !== undefined && advanced.correlation_p < 0.05 ? "asociación exploratoria" : "sin evidencia"}</strong></div>
    </div>
    <p class="note">${esc(advanced.note || "")} n=24: describe, no infiere. La correlación es descriptiva y su intervalo es muy amplio.</p>`;
  $("advancedBody").innerHTML = advanced.rows.map((row) => `<tr>
    <td>${esc(row.date)}</td><td>${esc(row.home)}</td><td>${esc(row.away)}</td>
    <td>${row.hg ?? "—"}–${row.ag ?? "—"}</td>
    <td>${row.shots_home ?? "—"}</td><td>${row.shots_away ?? "—"}</td>
    <td>${row.sot_home ?? "—"}</td><td>${row.sot_away ?? "—"}</td>
    <td>${row.pos_home !== null ? `${row.pos_home}%` : "—"}</td>
    <td>${row.pass_home !== null ? `${row.pass_home}%` : "—"}</td>
  </tr>`).join("");
}

/* ---------- modelo ---------- */
function drawReliability(container, bins, label) {
  const points = (bins || []).filter((b) => b.predicted !== null && b.count > 0);
  if (!points.length) { container.textContent = "Sin datos"; return; }
  const w = 340, h = 260, pad = 40;
  const scaleX = (v) => pad + v * (w - pad - 12);
  const scaleY = (v) => h - pad - v * (h - pad - 12);
  const path = points.map((b, i) => `${i ? "L" : "M"}${scaleX(b.predicted).toFixed(1)},${scaleY(b.observed).toFixed(1)}`).join(" ");
  const dots = points.map((b) => `<circle cx="${scaleX(b.predicted).toFixed(1)}" cy="${scaleY(b.observed).toFixed(1)}" r="${Math.min(3 + Math.sqrt(b.count) / 18, 8).toFixed(1)}"><title>${esc(b.bin)}: pred ${pct(b.predicted)} · obs ${pct(b.observed)} · n=${b.count}</title></circle>`).join("");
  container.innerHTML = `<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Curva de fiabilidad ${label}">
    <line x1="${pad}" y1="${h - pad}" x2="${w - 12}" y2="${h - pad}" class="axis"/>
    <line x1="${pad}" y1="${pad}" x2="${pad}" y2="${h - pad}" class="axis"/>
    <line x1="${pad}" y1="${h - pad}" x2="${w - 12}" y2="${pad}" class="diag"/>
    <path d="${path}" class="line"/>${dots}
    <text x="${w / 2}" y="${h - 8}" class="axis-label">probabilidad predicha</text>
    <text x="12" y="${h / 2}" transform="rotate(-90 12 ${h / 2})" class="axis-label">frecuencia observada</text>
    <text x="${pad}" y="${h - pad + 14}" class="tick">0</text>
    <text x="${w - 24}" y="${h - pad + 14}" class="tick">1</text>
    <text x="${pad - 10}" y="${pad + 4}" class="tick">1</text>
  </svg>`;
}

function renderModel() {
  const metrics = DATA.metrics;
  const agg = metrics.aggregate;
  const inf = agg.inference || {};
  const delta = agg.brier !== null && agg.base_brier !== null ? agg.base_brier - agg.brier : null;

  $("metricCards").innerHTML = [
    ["Brier", agg.brier !== null ? agg.brier.toFixed(4) : "—", "menor es mejor"],
    ["Log-loss", agg.logloss !== null ? agg.logloss.toFixed(4) : "—", "menor es mejor"],
    ["Acierto 1X2", agg.accuracy !== null ? pct(agg.accuracy) : "—", "resultado más probable"],
    ["Brier Skill Score", inf.bss !== null && inf.bss !== undefined ? pct(inf.bss) : "—", "1 − Brier/Brier baseline"],
  ].map(([label, value, note]) => `<div class="card metric"><span class="k">${label}</span><strong>${value}</strong><span class="note">${note}</span></div>`).join("");

  $("mMatches").textContent = metrics.totals ? metrics.totals.matches.toLocaleString("es-ES") : "—";
  $("mLeagues").textContent = metrics.totals ? metrics.totals.leagues : DATA.leagues.length;

  const bench = metrics.benchmark || {};
  if (bench.available) {
    const labels = { baseline: "Baseline (frecuencias)", dc: "Dixon-Coles", xgboost: "XGBoost", hybrid: "Híbrido 50/50" };
    $("benchmarkBody").innerHTML = ["baseline", "dc", "xgboost", "hybrid"].map((key) => {
      const model = bench.models[key] || {};
      const modelInf = model.inference || {};
      const gap = modelInf.delta_brier;
      return `<tr>
        <td>${labels[key]}</td><td>${model.n ?? "—"}</td>
        <td>${model.brier !== null ? model.brier.toFixed(4) : "—"}</td>
        <td>${fmtCI(modelInf.brier_ci, 4)}</td>
        <td>${model.logloss !== null ? model.logloss.toFixed(4) : "—"}</td>
        <td>${model.accuracy !== null ? pct(model.accuracy) : "—"}</td>
        <td class="${gap > 0 ? "pos" : "neg"}">${gap !== null && gap !== undefined ? signed(gap, 4) : "—"}</td>
        <td>${gap !== null && gap !== undefined ? fmtP(modelInf.p_value) : "—"}</td>
      </tr>`;
    }).join("");
  } else {
    $("benchmarkBody").innerHTML = `<tr><td colspan="8" class="muted">XGBoost no disponible; solo Dixon-Coles.</td></tr>`;
  }

  const decomposition = inf.brier_decomposition || {};
  $("inferenceCards").innerHTML = [
    ["Brier IC95%", fmtCI(inf.brier_ci, 4)],
    ["BSS IC95%", inf.bss_ci ? fmtCI(inf.bss_ci.map((value) => value * 100), 1) + "%" : "—"],
    ["ECE", inf.ece !== null && inf.ece !== undefined ? `${(inf.ece * 100).toFixed(2)}%` : "—"],
    ["Fiabilidad", decomposition.reliability ?? "—"],
    ["Resolución", decomposition.resolution ?? "—"],
    ["Incertidumbre", decomposition.uncertainty ?? "—"],
    ["Bloques bootstrap", inf.blocks ?? "—"],
    ["p vs baseline", fmtP(inf.p_value)],
  ].map(([label, value]) => `<div><span>${label}</span><strong>${value}</strong></div>`).join("");

  const vsDc = bench.models ? bench.models.hybrid_vs_dc?.inference : null;
  $("inferenceNote").textContent = (vsDc
    ? `IC95% por bootstrap de bloques semanales. BSS = 1 − Brier/Brier baseline. Híbrido frente a Dixon-Coles: Δ Brier ${signed(vsDc.delta_brier, 4)} (${fmtCI(vsDc.delta_brier_ci, 4)}), p ${fmtP(vsDc.p_value)}. `
    : "IC95% por bootstrap de bloques semanales. BSS = 1 − Brier/Brier baseline. ")
    + "El agregado incluye todas las ligas evaluadas. El subconjunto validado se usa solo para decidir qué ligas publicar y está seleccionado por resultado: no es una estimación insesgada.";

  drawReliability($("relChartModel"), metrics.aggregate.reliability, "agregada");

  $("metricsBody").innerHTML = DATA.leagues.map((league) => {
    const m = metrics.leagues[league.code] || {};
    if (!m.n) return null;
    const gap = m.brier !== null && m.base_brier !== null ? m.base_brier - m.brier : null;
    const ci = m.inference ? m.inference.brier_ci : null;
    return `<tr>
      <td>${esc(league.country)} · ${esc(league.name)}</td>
      <td>${esc(league.last)}</td>
      <td><span class="tag ${m.validated ? "ok" : "warn"}">${m.validated ? "validada" : "sin evidencia"}</span></td>
      <td>${m.n}</td>
      <td>${m.brier !== null ? m.brier.toFixed(4) : "—"}</td>
      <td>${m.dc_brier !== null && m.dc_brier !== undefined ? Number(m.dc_brier).toFixed(4) : "—"}</td>
      <td>${fmtCI(ci, 4)}</td>
      <td class="${gap !== null && gap > 0 ? "pos" : "neg"}">${gap !== null ? signed(gap, 4) : "—"}</td>
      <td>${fmtP(m.p_value)}</td>
      <td>${fmtP(m.q_value)}</td>
      <td>${m.accuracy !== null ? pct(m.accuracy) : "—"}</td>
      <td>${m.coverage !== null ? pct(m.coverage) : "—"}</td>
    </tr>`;
  }).filter(Boolean).join("");
}

function renderMarketing() {
  const metrics = DATA.metrics;
  const agg = metrics.aggregate || {};
  const totals = metrics.totals || {};
  const lab = (DATA.labIndex && DATA.labIndex.totals) || {};
  const ci = agg.inference && agg.inference.brier_ci;
  const matches = (totals.matches || 0).toLocaleString("es-ES");
  const leagues = totals.leagues || 0;
  const teams = (lab.teams || 0).toLocaleString("es-ES");
  const brier = agg.brier !== null && agg.brier !== undefined ? Number(agg.brier).toFixed(3) : "—";

  const micro = document.querySelector(".hero-micro");
  if (micro && metrics.data_through) {
    micro.textContent = `Datos hasta ${metrics.data_through} · gratis, sin registro, sin cookies. El cálculo ocurre en tu navegador.`;
  }

  $("proofMatches").textContent = matches;
  $("proofLeagues").textContent = leagues;
  $("proofTeams").textContent = teams;
  $("proofBrier").textContent = `Brier ${brier}`;
  $("proofCI").textContent = ci ? `[${Number(ci[0]).toFixed(3)}–${Number(ci[1]).toFixed(3)}]` : "—";
}

function scrollToSection(id, focusId) {
  const target = document.getElementById(id);
  if (!target) return;
  target.scrollIntoView({ behavior: "smooth", block: "start" });
  if (focusId) {
    const field = document.getElementById(focusId);
    if (field) setTimeout(() => field.focus({ preventScroll: true }), 350);
  }
}

function setupMarketing() {
  $("heroCta").addEventListener("click", () => scrollToSection("predictControls", "league"));
  $("heroHow").addEventListener("click", () => scrollToSection("como-se-mide"));
  $("finalCta").addEventListener("click", () => scrollToSection("predictControls", "league"));
}

/* ---------- arranque ---------- */
function setupPredict() {
  fillLeagueSelect($("league"), playableLeagues());
  $("league").addEventListener("change", () => setLeague($("league").value));
  $("home").addEventListener("change", () => { state.home = $("home").value; renderPrediction(); });
  $("away").addEventListener("change", () => { state.away = $("away").value; renderPrediction(); });
  $("swap").addEventListener("click", () => setLeague(state.league, state.away, state.home));
  $("neutral").addEventListener("change", () => { state.neutral = $("neutral").checked; renderPrediction(); });
  for (const key of ["home", "draw", "away"]) {
    const input = $("odds" + key[0].toUpperCase() + key.slice(1));
    input.addEventListener("input", () => {
      const value = Number(input.value);
      state.odds[key] = Number.isFinite(value) && value > 1 ? value : null;
      if (state.league) renderPrediction();
    });
  }
  $("clearOdds").addEventListener("click", () => {
    for (const key of ["home", "draw", "away"]) {
      state.odds[key] = null;
      $("odds" + key[0].toUpperCase() + key.slice(1)).value = "";
    }
    if (state.league) renderPrediction();
  });
  const playable = playableLeagues();
  if (playable.length) setLeague(playable[0].code);
  else {
    $("predictEmpty").hidden = false;
    $("predictEmpty").textContent = "Aún no hay competiciones validadas. Solo se publican ligas con evidencia suficiente.";
  }
}

async function boot() {
  document.querySelectorAll("th").forEach((th) => { if (!th.scope) th.scope = "col"; });
  setupTabs();
  try {
    const [leagues, ratings, metrics, labIndex, hybrid, live, sources] = await Promise.all([
      fetchJSON("data/leagues.json"),
      fetchJSON("data/ratings.json"),
      fetchJSON("data/metrics.json"),
      fetchJSON("data/lab/index.json"),
      fetchJSON("data/hybrid.json").catch(() => null),
      fetchJSON("data/live.json").catch(() => null),
      fetchJSON("data/sources.json").catch(() => null),
    ]);
    DATA.leagues = leagues;
    DATA.ratings = ratings;
    DATA.metrics = metrics;
    DATA.labIndex = labIndex;
    DATA.hybrid = hybrid;
    DATA.live = live;
    DATA.sources = sources;
    $("dataStamp").textContent = `datos hasta ${ratings.data_through}`;

    setupPredict();
    setupLab();
    setupMarketing();
    renderMarketing();
    renderData();
    renderModel();
  } catch (error) {
    $("dataStamp").textContent = "datos no disponibles";
    $("predictEmpty").hidden = false;
    $("predictEmpty").textContent = "No se pudieron cargar los datos. Revisa tu conexión y recarga la página.";
    console.error(error);
  }
}

boot();
