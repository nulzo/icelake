"""Render a ``GraphSnapshot`` as a self-contained HTML explorer.

The template is the product UI (search, filters, inspector, Sigma canvas).
Splitting CSS/JS would require a frontend build we explicitly rejected.
"""

from __future__ import annotations

import json
from pathlib import Path

from icelake.visualizer.models import GraphSnapshot

# Placeholder must not collide with CSS percentages.
_PAYLOAD = "%%ICELAKE_SNAPSHOT%%"


def render_html(snapshot: GraphSnapshot) -> str:
    payload = json.dumps(snapshot.model_dump(mode="json"), ensure_ascii=False)
    safe = payload.replace("<", "\\u003c").replace(">", "\\u003e")
    return _TEMPLATE.replace(_PAYLOAD, safe)


def write_html(path: Path, snapshot: GraphSnapshot) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(snapshot), encoding="utf-8")


_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>icelake</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Instrument+Sans:ital,wght@0,400;0,500;0,600;0,700;1,400&family=Instrument+Serif:ital@0;1&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet" />
  <style>
    :root {
      --bg: #05080c;
      --bg-2: #0a1018;
      --panel: rgba(10, 16, 24, 0.88);
      --panel-solid: #0c141e;
      --line: rgba(125, 211, 252, 0.14);
      --text: #e8f4fc;
      --muted: #7d93a8;
      --accent: #7dd3fc;
      --accent-2: #22d3ee;
      --user: #38bdf8;
      --entity-person: #fbbf24;
      --entity-place: #34d399;
      --entity-concept: #a78bfa;
      --entity-org: #fb7185;
      --server: #f8fafc;
      --pos: #4ade80;
      --neg: #f87171;
      --neu: #64748b;
      --id-edge: #fbbf24;
    }
    * { box-sizing: border-box; }
    html, body { margin: 0; height: 100%; overflow: hidden; background: var(--bg); color: var(--text);
      font-family: "Instrument Sans", system-ui, sans-serif; }
    #shell {
      display: grid;
      grid-template-columns: minmax(220px, 280px) minmax(0, 1fr);
      grid-template-rows: auto minmax(0, 1fr) minmax(200px, 30vh);
      height: 100vh; position: relative;
    }
    header {
      grid-column: 1 / -1; display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
      padding: 10px 16px; border-bottom: 1px solid var(--line);
      background: linear-gradient(180deg, rgba(14, 24, 36, 0.95), rgba(8, 14, 22, 0.92));
    }
    .mark { font-family: "Instrument Serif", serif; font-size: 22px; letter-spacing: -0.03em; }
    .mark span { color: var(--accent); font-style: italic; }
    .guild { font-family: "JetBrains Mono", monospace; font-size: 11px; color: var(--muted); }
    .pills { display: flex; gap: 6px; flex: 1; flex-wrap: wrap; }
    .pill {
      font-size: 11px; color: var(--muted); border: 1px solid var(--line);
      background: rgba(125, 211, 252, 0.06); border-radius: 999px; padding: 4px 10px;
    }
    .pill b { color: var(--text); font-weight: 600; }
    header button, .ghost {
      background: transparent; border: 1px solid var(--line); color: var(--text);
      border-radius: 8px; padding: 6px 12px; font-size: 12px; cursor: pointer;
      font-family: inherit; transition: border-color .15s, background .15s;
    }
    header button:hover, .ghost:hover { border-color: var(--accent); color: var(--accent); }
    header button.on { background: rgba(34, 211, 238, 0.16); border-color: var(--accent-2); color: #fff; }
    #sidebar, #inspector {
      background: var(--panel); backdrop-filter: blur(16px); overflow: auto;
      padding: 16px 16px 24px; min-height: 0; font-size: 13px;
    }
    #sidebar { border-right: 1px solid var(--line); grid-row: 2 / 4; }
    #viewport {
      grid-column: 2; grid-row: 2; position: relative; min-width: 0; min-height: 0;
      background:
        radial-gradient(ellipse at 50% 30%, rgba(34, 211, 238, 0.07), transparent 52%),
        radial-gradient(ellipse at 80% 80%, rgba(167, 139, 250, 0.05), transparent 40%),
        var(--bg);
    }
    #inspector {
      grid-column: 2; grid-row: 3; width: auto; height: auto;
      z-index: 4; border-top: 1px solid var(--line); border-left: none;
      background: rgba(10, 16, 24, 0.94);
    }
    #canvas { position: absolute; inset: 0; }
    #status {
      position: absolute; left: 14px; bottom: 14px; z-index: 4;
      font-family: "JetBrains Mono", monospace; font-size: 11px; color: var(--muted);
      background: rgba(5, 8, 12, 0.78); border: 1px solid var(--line);
      padding: 6px 12px; border-radius: 8px; pointer-events: none;
    }
    h2 {
      margin: 18px 0 8px; font-size: 10px; letter-spacing: 0.14em; text-transform: uppercase;
      color: var(--muted); font-weight: 600;
    }
    h2:first-child { margin-top: 0; }
    input[type=search], input[type=text] {
      width: 100%; padding: 10px 12px; border-radius: 10px; border: 1px solid var(--line);
      background: rgba(5, 8, 12, 0.65); color: var(--text); font: inherit;
    }
    input[type=search]:focus { outline: none; border-color: var(--accent); }
    input[type=range] { width: 100%; accent-color: var(--accent); }
    label.check { display: flex; align-items: center; gap: 8px; margin: 6px 0; cursor: pointer; color: var(--text); }
    .chips { display: flex; flex-wrap: wrap; gap: 6px; }
    .chip {
      font-size: 11px; padding: 3px 9px; border-radius: 999px; border: 1px solid var(--line);
      background: transparent; color: var(--muted); cursor: pointer; font-family: inherit;
    }
    .chip.on { color: var(--text); border-color: var(--accent); background: rgba(125, 211, 252, 0.12); }
    .row {
      padding: 8px 8px; border-radius: 8px; cursor: pointer; display: flex; gap: 8px; align-items: baseline;
    }
    .row:hover, .row.sel { background: rgba(125, 211, 252, 0.08); }
    .row .dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; margin-top: 4px; }
    .row .name { font-weight: 550; }
    .row .meta { color: var(--muted); font-size: 11px; margin-left: auto; font-family: "JetBrains Mono", monospace; }
    .hint { color: var(--muted); font-size: 12px; line-height: 1.45; }
    .legend { display: flex; flex-wrap: wrap; gap: 10px; font-size: 11px; color: var(--muted); }
    .legend i { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 5px; }
    #hits { max-height: 220px; overflow: auto; margin-top: 8px; }
    #catalog { max-height: 280px; overflow: auto; }
    #detail h1 {
      font-family: "Instrument Serif", serif; font-size: 26px; font-weight: 400;
      margin: 0 0 6px; letter-spacing: -0.03em;
    }
    .kicker { font-size: 11px; letter-spacing: 0.12em; text-transform: uppercase; color: var(--accent); }
    .fact {
      padding: 10px 0; border-bottom: 1px solid var(--line);
    }
    .fact .body { line-height: 1.45; }
    .fact .meta { color: var(--muted); font-size: 11px; margin-top: 4px; font-family: "JetBrains Mono", monospace; }
    .fact.dead { opacity: 0.45; }
    a { color: var(--accent); }
    .empty { color: var(--muted); font-style: italic; }
    .tag {
      display: inline-block; font-size: 10px; padding: 1px 7px; border-radius: 999px;
      border: 1px solid var(--line); margin-right: 4px; color: var(--muted);
    }
    .pol-positive { color: var(--pos); }
    .pol-negative { color: var(--neg); }
    .pol-neutral { color: var(--muted); }
  </style>
</head>
<body>
  <div id="shell">
    <header>
      <div class="mark">ice<span>lake</span></div>
      <div class="guild" id="guild"></div>
      <div class="pills" id="pills"></div>
      <button type="button" id="preset-people">People</button>
      <button type="button" id="preset-relations" class="on">Relations</button>
      <button type="button" id="preset-all">Everything</button>
      <button type="button" id="layout">Re-cluster</button>
      <button type="button" id="fit">Fit</button>
    </header>
    <aside id="sidebar">
      <h2>Search</h2>
      <input id="search" type="search" placeholder="People, entities, facts…" />
      <div id="hits"></div>
      <h2>Layers</h2>
      <label class="check"><input type="checkbox" class="layer" data-type="user" checked /> Users</label>
      <label class="check"><input type="checkbox" class="layer" data-type="entity" checked /> Entities</label>
      <label class="check"><input type="checkbox" class="layer" data-type="server" checked /> Server</label>
      <h2>Entity kind</h2>
      <div class="chips" id="kind-chips"></div>
      <h2>Polarity</h2>
      <label class="check"><input type="checkbox" class="pol" data-pol="positive" checked /> Positive</label>
      <label class="check"><input type="checkbox" class="pol" data-pol="negative" checked /> Negative</label>
      <label class="check"><input type="checkbox" class="pol" data-pol="neutral" checked /> Neutral</label>
      <h2>Edge kind</h2>
      <label class="check"><input type="checkbox" id="rel-edges" checked /> Relations</label>
      <label class="check"><input type="checkbox" id="id-edges" checked /> Identity (entity is member)</label>
      <h2>Verbs</h2>
      <div class="chips" id="verb-chips"></div>
      <h2>Min weight <span id="wval">0.00</span></h2>
      <input id="weight" type="range" min="0" max="5" step="0.05" value="0" />
      <h2>View</h2>
      <label class="check"><input type="checkbox" id="hide-isolates" checked /> Hide isolated entities</label>
      <label class="check"><input type="checkbox" id="focus-mode" /> 2-hop focus on select</label>
      <h2>Legend</h2>
      <div class="legend">
        <span><i style="background:var(--user)"></i>User</span>
        <span><i style="background:var(--entity-person)"></i>Person</span>
        <span><i style="background:var(--entity-place)"></i>Place</span>
        <span><i style="background:var(--entity-concept)"></i>Concept</span>
        <span><i style="background:var(--entity-org)"></i>Org</span>
        <span><i style="background:var(--server)"></i>Server</span>
      </div>
      <h2>Catalog</h2>
      <div class="chips">
        <button type="button" class="chip on" data-cat="user" id="tab-user">People</button>
        <button type="button" class="chip" data-cat="entity" id="tab-entity">Entities</button>
        <button type="button" class="chip" data-cat="fact" id="tab-fact">Facts</button>
      </div>
      <div id="catalog"></div>
    </aside>
    <main id="viewport">
      <div id="canvas"></div>
      <div id="status">Loading libraries…</div>
    </main>
    <aside id="inspector">
      <div id="detail"><p class="empty">Click a person, entity, the server, or an edge.</p></div>
    </aside>
  </div>
  <script type="module">
    const SNAPSHOT = %%ICELAKE_SNAPSHOT%%;
    const ALL_NODES = SNAPSHOT.nodes || [];
    const ALL_EDGES = SNAPSHOT.edges || [];
    const ALL_FACTS = SNAPSHOT.facts || [];
    const nodeById = new Map(ALL_NODES.map(n => [n.id, n]));
    const factById = new Map(ALL_FACTS.map(f => [f.id, f]));
    const adj = new Map();
    ALL_NODES.forEach(n => adj.set(n.id, []));
    ALL_EDGES.forEach(e => {
      (adj.get(e.source) || adj.set(e.source, []).get(e.source)).push(e);
      (adj.get(e.target) || adj.set(e.target, []).get(e.target)).push(e);
    });

    const verbs = [...new Set(ALL_EDGES.filter(e => e.kind === "relation").map(e => e.verb))].sort();
    const kinds = [...new Set(ALL_NODES.filter(n => n.type === "entity").map(n => n.entity_kind).filter(Boolean))].sort();
    const enabledKinds = new Set(kinds);
    const enabledVerbs = new Set(verbs);
    const enabledTypes = new Set(["user", "entity", "server"]);
    const enabledPol = new Set(["positive", "negative", "neutral"]);
    let showRel = true, showId = true, hideIsolates = true, focusMode = false;
    let minWeight = 0, query = "", selected = null, hovered = null, catalog = "user";
    const statusEl = document.getElementById("status");
    const USE_WORKER = location.protocol !== "file:" && location.protocol !== "null:";

    document.getElementById("guild").textContent = SNAPSHOT.guild_id;
    const s = SNAPSHOT.stats || {};
    document.getElementById("pills").innerHTML = [
      ["facts", s.active_facts ?? ALL_FACTS.length],
      ["people", ALL_NODES.filter(n => n.type === "user").length],
      ["entities", ALL_NODES.filter(n => n.type === "entity").length],
      ["relations", ALL_EDGES.filter(e => e.kind === "relation").length],
    ].map(([k, v]) => `<span class="pill"><b>${v}</b> ${k}</span>`).join("");

    const kindChips = document.getElementById("kind-chips");
    kinds.forEach(k => {
      const b = document.createElement("button");
      b.className = "chip on"; b.textContent = k; b.dataset.kind = k;
      b.onclick = () => { b.classList.toggle("on"); enabledKinds.has(k) ? enabledKinds.delete(k) : enabledKinds.add(k); refresh(); };
      kindChips.append(b);
    });
    const verbChips = document.getElementById("verb-chips");
    verbs.forEach(v => {
      const b = document.createElement("button");
      b.className = "chip on"; b.textContent = v; b.dataset.verb = v;
      b.onclick = () => { b.classList.toggle("on"); enabledVerbs.has(v) ? enabledVerbs.delete(v) : enabledVerbs.add(v); refresh(); };
      verbChips.append(b);
    });
    if (!verbs.length) verbChips.innerHTML = '<span class="hint">No typed relations in this snapshot.</span>';

    let Graph, Sigma, forceAtlas2, FA2Layout;
    try {
      Graph = (await import("https://esm.sh/graphology@0.26.0")).default;
      Sigma = (await import("https://esm.sh/sigma@4.0.0-alpha.5")).default;
      forceAtlas2 = (await import("https://esm.sh/graphology-layout-forceatlas2@0.10.1")).default;
      if (USE_WORKER) FA2Layout = (await import("https://esm.sh/graphology-layout-forceatlas2@0.10.1/worker")).default;
    } catch (err) {
      statusEl.textContent = "Need network for graph libraries, or serve this file over HTTP.";
      throw err;
    }

    let graph = new Graph({ multi: true, type: "directed" });
    let renderer = null, fa2 = null;
    const pos = {};

    function nodeColor(n) {
      if (n.type === "user") return getComputedStyle(document.documentElement).getPropertyValue("--user").trim();
      if (n.type === "server") return "#f8fafc";
      return ({ person: "#fbbf24", place: "#34d399", concept: "#a78bfa", org: "#fb7185" }[n.entity_kind] || "#a78bfa");
    }
    function edgeColor(e) {
      if (e.kind === "identity") return "#fbbf24";
      if (e.polarity === "positive") return "#4ade80";
      if (e.polarity === "negative") return "#f87171";
      return "#64748b";
    }
    function degree(id) {
      return (adj.get(id) || []).length;
    }
    function ego(center, hops) {
      const keep = new Set([center]);
      let frontier = new Set([center]);
      for (let i = 0; i < hops; i++) {
        const next = new Set();
        for (const id of frontier) for (const e of adj.get(id) || []) { next.add(e.source); next.add(e.target); }
        next.forEach(id => keep.add(id));
        frontier = next;
      }
      return keep;
    }

    function visible() {
      const q = query.trim().toLowerCase();
      let nodes = ALL_NODES.filter(n => {
        if (!enabledTypes.has(n.type)) return false;
        if (n.type === "entity" && n.entity_kind && !enabledKinds.has(n.entity_kind)) return false;
        if (hideIsolates && n.type === "entity" && degree(n.id) === 0) return false;
        if (q && !(n.search_text || n.label || "").toLowerCase().includes(q)) {
          const factHit = (n.fact_ids || []).some(id => (factById.get(id)?.text || "").toLowerCase().includes(q));
          if (!factHit) return false;
        }
        return true;
      });
      const ids = new Set(nodes.map(n => n.id));
      if (focusMode && selected && ids.has(selected)) {
        const keep = ego(selected, 2);
        nodes = nodes.filter(n => keep.has(n.id));
        ids.clear(); nodes.forEach(n => ids.add(n.id));
      }
      const edges = ALL_EDGES.filter(e => {
        if (!ids.has(e.source) || !ids.has(e.target)) return false;
        if (e.kind === "identity") return showId;
        if (e.kind === "relation") {
          if (!showRel) return false;
          if (!enabledPol.has(e.polarity)) return false;
          if (verbs.length && !enabledVerbs.has(e.verb)) return false;
          if ((e.weight || 0) < minWeight) return false;
        }
        return true;
      });
      return { nodes, edges };
    }

    function seed(nodes) {
      const users = nodes.filter(n => n.type === "user");
      const ents = nodes.filter(n => n.type === "entity");
      nodes.forEach((n, i) => {
        if (pos[n.id]) return;
        if (n.type === "server") { pos[n.id] = { x: 0, y: 0 }; return; }
        if (n.type === "user") {
          const a = (2 * Math.PI * users.indexOf(n)) / Math.max(1, users.length);
          pos[n.id] = { x: Math.cos(a) * 80, y: Math.sin(a) * 80 };
          return;
        }
        const a = (2 * Math.PI * ents.indexOf(n)) / Math.max(1, ents.length);
        pos[n.id] = { x: Math.cos(a) * 160 + (Math.random() - 0.5) * 20, y: Math.sin(a) * 160 };
      });
    }

    function rebuild() {
      const { nodes, edges } = visible();
      cachePos();
      seed(nodes);
      if (fa2) { try { fa2.kill(); } catch (_) {} fa2 = null; }
      if (renderer) { renderer.kill(); renderer = null; }
      graph.clear();
      nodes.forEach(n => {
        const p = pos[n.id] || { x: Math.random(), y: Math.random() };
        const deg = degree(n.id);
        graph.addNode(n.id, {
          label: n.label, x: p.x, y: p.y,
          size: n.type === "server" ? 16 : n.type === "user" ? 8 + Math.min(10, deg) : 5 + Math.min(6, deg),
          color: nodeColor(n), type: "circle",
        });
      });
      edges.forEach(e => {
        if (!graph.hasNode(e.source) || !graph.hasNode(e.target)) return;
        graph.addEdgeWithKey(e.id, e.source, e.target, {
          label: e.verb, size: e.kind === "identity" ? 0.6 : Math.max(0.4, Math.min(2.4, (e.weight || 0.4) * 0.7)),
          color: edgeColor(e), type: "arrow",
        });
      });
      renderer = new Sigma(graph, document.getElementById("canvas"), {
        labelFont: "Instrument Sans", labelWeight: "500", labelColor: { color: "#e8f4fc" },
        labelSize: 13, labelRenderedSizeThreshold: 0,
        defaultEdgeColor: "#334155", renderEdgeLabels: false,
      });
      renderer.on("clickNode", ({ node }) => select("node", node));
      renderer.on("clickEdge", ({ edge }) => select("edge", edge));
      renderer.on("clickStage", () => select(null, null));
      renderer.on("enterNode", ({ node }) => { hovered = node; paint(); });
      renderer.on("leaveNode", () => { hovered = null; paint(); });
      renderer.setSetting("nodeReducer", (id, attr) => {
        const hi = highlightSet();
        const dim = hi && !hi.has(id);
        return { ...attr, color: dim ? "#1e293b" : attr.color, label: dim ? "" : attr.label, zIndex: dim ? 0 : 1 };
      });
      renderer.setSetting("edgeReducer", (id, attr) => {
        const e = ALL_EDGES.find(x => x.id === id);
        const hi = highlightSet();
        const dim = hi && e && !hi.has(e.source) && !hi.has(e.target);
        return { ...attr, hidden: !!dim, color: dim ? "#0f172a" : attr.color };
      });
      statusEl.textContent = `${nodes.length} nodes · ${edges.length} edges` + (USE_WORKER ? "" : " · serve over HTTP for live layout");
      startLayout();
      renderCatalog();
      paint();
      setTimeout(() => { try { renderer && renderer.getCamera().animatedReset({ duration: 700 }); } catch (_) {} }, 400);
      setTimeout(() => { try { renderer && renderer.getCamera().animatedReset({ duration: 700 }); } catch (_) {} }, 2600);
    }

    function highlightSet() {
      const id = hovered || selected;
      if (!id || !nodeById.has(id)) return null;
      const keep = new Set([id]);
      for (const e of adj.get(id) || []) { keep.add(e.source); keep.add(e.target); }
      return keep;
    }
    function paint() { if (renderer) renderer.refresh(); }
    function cachePos() {
      if (!graph.order) return;
      graph.forEachNode((id, a) => { pos[id] = { x: a.x, y: a.y }; });
    }
    function startLayout() {
      if (!graph.order) return;
      const settings = forceAtlas2.inferSettings(graph);
      Object.assign(settings, { gravity: 0.8, slowDown: 2, barnesHutOptimize: graph.order > 200 });
        if (FA2Layout) {
        fa2 = new FA2Layout(graph, { settings });
        fa2.start();
        setTimeout(() => { try { fa2 && fa2.stop(); } catch (_) {} }, 2400);
      } else {
        forceAtlas2.assign(graph, { iterations: 120, settings });
        paint();
      }
    }

    function esc(s) {
      return String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
    }
    function factCard(f) {
      if (!f) return "";
      const cites = (f.citations || []).filter(c => c.message_url || c.content_snippet).map(c =>
        c.message_url ? `<a href="${esc(c.message_url)}" target="_blank" rel="noreferrer">${esc(c.author_name || "source")}</a>` : esc(c.content_snippet)
      ).join(" · ");
      return `<div class="fact ${f.active ? "" : "dead"}"><div class="body">${esc(f.text)}</div>
        <div class="meta"><span class="tag">${esc(f.category)}</span><span class="tag">${esc(f.tier)}</span>
        ×${f.occurrences} · ${(f.confidence * 100).toFixed(0)}%${f.active ? "" : " · inactive"}${cites ? " · " + cites : ""}</div></div>`;
    }

    function similarUsers(userId) {
      const mine = new Set();
      for (const e of ALL_EDGES) {
        if (e.kind !== "relation") continue;
        if (e.source === userId && nodeById.get(e.target)?.type === "entity") mine.add(e.target);
        if (e.target === userId && nodeById.get(e.source)?.type === "entity") mine.add(e.source);
      }
      if (!mine.size) return [];
      const scored = [];
      for (const n of ALL_NODES) {
        if (n.type !== "user" || n.id === userId) continue;
        const theirs = new Set();
        for (const e of ALL_EDGES) {
          if (e.kind !== "relation") continue;
          if (e.source === n.id && nodeById.get(e.target)?.type === "entity") theirs.add(e.target);
          if (e.target === n.id && nodeById.get(e.source)?.type === "entity") theirs.add(e.source);
        }
        let inter = 0; for (const x of mine) if (theirs.has(x)) inter++;
        const union = mine.size + theirs.size - inter;
        if (inter && union) scored.push({ n, score: inter / union });
      }
      scored.sort((a, b) => b.score - a.score);
      return scored.slice(0, 8);
    }

    function inspectNode(id) {
      const n = nodeById.get(id);
      if (!n) return;
      const facts = (n.fact_ids || []).map(fid => factById.get(fid)).filter(Boolean);
      const aliases = (n.aliases || []).map(a => `<span class="tag">${esc(a.alias)}${a.source ? " · " + esc(a.source) : ""}</span>`).join(" ");
      const edges = (adj.get(id) || []);
      const relHtml = edges.map(e => {
        const other = e.source === id ? e.target : e.source;
        const o = nodeById.get(other);
        return `<div class="row" data-jump="${esc(e.id)}" data-kind="edge"><span class="dot" style="background:${edgeColor(e)}"></span>
          <span class="name pol-${esc(e.polarity)}">${esc(e.verb)}</span>
          <span class="meta">${esc(o?.label || other)}</span></div>`;
      }).join("");
      let extra = "";
      if (n.type === "user") {
        const sim = similarUsers(id);
        extra = sim.length ? `<h2>Similar members (entity overlap)</h2>` + sim.map(({ n: o, score }) =>
          `<div class="row" data-jump="${esc(o.id)}" data-kind="node"><span class="dot" style="background:${nodeColor(o)}"></span>
           <span class="name">${esc(o.label)}</span><span class="meta">${(score * 100).toFixed(0)}%</span></div>`
        ).join("") : "";
      }
      if (n.linked_user_id) {
        const uid = "user:" + n.linked_user_id;
        const u = nodeById.get(uid);
        extra += `<h2>Linked member</h2><div class="row" data-jump="${esc(uid)}" data-kind="node"><span class="name">${esc(u?.label || n.linked_user_id)}</span></div>`;
      }
      document.getElementById("detail").innerHTML = `
        <div class="kicker">${esc(n.type)}${n.entity_kind ? " · " + esc(n.entity_kind) : ""}</div>
        <h1>${esc(n.label)}</h1>
        ${n.user_id ? `<div class="hint" style="font-family:JetBrains Mono,monospace">${esc(n.user_id)}</div>` : ""}
        ${n.entity_slug ? `<div class="hint">${esc(n.entity_slug)}</div>` : ""}
        ${aliases ? `<h2>Aliases</h2><div>${aliases}</div>` : ""}
        <h2>Relationships <span class="tag">${edges.length}</span></h2>
        ${relHtml || '<p class="empty">No typed edges.</p>'}
        ${extra}
        <h2>Facts <span class="tag">${facts.length}</span></h2>
        ${facts.map(factCard).join("") || '<p class="empty">No facts on this node.</p>'}
      `;
      bindJumps();
    }

    function inspectEdge(id) {
      const e = ALL_EDGES.find(x => x.id === id);
      if (!e) return;
      const a = nodeById.get(e.source), b = nodeById.get(e.target);
      const facts = (e.evidence_fact_ids || []).map(fid => factById.get(fid)).filter(Boolean);
      document.getElementById("detail").innerHTML = `
        <div class="kicker">${esc(e.kind)} · <span class="pol-${esc(e.polarity)}">${esc(e.polarity)}</span></div>
        <h1>${esc(e.verb)}</h1>
        <p>${esc(a?.label || e.source)} → ${esc(b?.label || e.target)}</p>
        <div class="hint">weight ${Number(e.weight || 0).toFixed(2)} · ×${e.occurrences} · ${(e.confidence * 100).toFixed(0)}%</div>
        <h2>Evidence</h2>
        ${facts.map(factCard).join("") || '<p class="empty">No evidence facts attached.</p>'}
      `;
    }

    function inspectFact(id) {
      const f = factById.get(id);
      if (!f) return;
      const subject = f.subject_id ? nodeById.get("user:" + f.subject_id) : nodeById.get("server:guild");
      document.getElementById("detail").innerHTML = `
        <div class="kicker">fact${f.subject_id ? "" : " · server"}</div>
        <h1>Memory</h1>
        ${factCard(f)}
        ${subject ? `<h2>Subject</h2><div class="row" data-jump="${esc(subject.id)}" data-kind="node"><span class="name">${esc(subject.label)}</span></div>` : ""}
      `;
      bindJumps();
    }

    function bindJumps() {
      document.querySelectorAll("#detail [data-jump]").forEach(el => {
        el.onclick = () => {
          const kind = el.dataset.kind, id = el.dataset.jump;
          if (kind === "node" && nodeById.has(id)) select("node", id);
          if (kind === "edge") select("edge", id);
        };
      });
    }

    function select(kind, id) {
      selected = kind === "node" ? id : (kind === "edge" ? null : null);
      if (!kind) {
        document.getElementById("detail").innerHTML = '<p class="empty">Click a person, entity, the server, or an edge.</p>';
      } else if (kind === "node") inspectNode(id);
      else inspectEdge(id);
      if (focusMode) rebuild(); else paint();
    }

    function renderHits() {
      const q = query.trim().toLowerCase();
      const box = document.getElementById("hits");
      if (!q) { box.innerHTML = ""; return; }
      const rows = [];
      for (const n of ALL_NODES) {
        if ((n.search_text || n.label).toLowerCase().includes(q))
          rows.push({ kind: "node", id: n.id, label: n.label, meta: n.type, color: nodeColor(n) });
      }
      for (const f of ALL_FACTS) {
        if ((f.text || "").toLowerCase().includes(q))
          rows.push({ kind: "fact", id: f.id, label: f.text.slice(0, 80), meta: "fact", color: "#7dd3fc" });
      }
      box.innerHTML = rows.slice(0, 24).map(r =>
        `<div class="row" data-kind="${r.kind}" data-id="${esc(r.id)}"><span class="dot" style="background:${r.color || "var(--accent)"}"></span>
         <span class="name">${esc(r.label)}</span><span class="meta">${esc(r.meta)}</span></div>`
      ).join("") || '<p class="empty">No matches.</p>';
      box.querySelectorAll(".row").forEach(el => {
        el.onclick = () => {
          if (el.dataset.kind === "node") select("node", el.dataset.id);
          else inspectFact(el.dataset.id);
        };
      });
    }

    function renderCatalog() {
      const box = document.getElementById("catalog");
      if (catalog === "fact") {
        box.innerHTML = ALL_FACTS.map(f =>
          `<div class="row" data-id="${esc(f.id)}"><span class="name">${esc(f.text.slice(0, 90))}</span></div>`
        ).join("") || '<p class="empty">No facts.</p>';
        box.querySelectorAll(".row").forEach(el => el.onclick = () => inspectFact(el.dataset.id));
        return;
      }
      const items = ALL_NODES.filter(n => n.type === catalog).sort((a, b) => a.label.localeCompare(b.label));
      box.innerHTML = items.map(n =>
        `<div class="row" data-id="${esc(n.id)}"><span class="dot" style="background:${nodeColor(n)}"></span>
         <span class="name">${esc(n.label)}</span><span class="meta">${n.fact_ids?.length || 0}</span></div>`
      ).join("") || '<p class="empty">None.</p>';
      box.querySelectorAll(".row").forEach(el => el.onclick = () => select("node", el.dataset.id));
    }

    function refresh() { renderHits(); rebuild(); }

    document.getElementById("search").addEventListener("input", e => { query = e.target.value; refresh(); });
    document.querySelectorAll(".layer").forEach(el => el.addEventListener("change", () => {
      enabledTypes.clear();
      document.querySelectorAll(".layer:checked").forEach(x => enabledTypes.add(x.dataset.type));
      refresh();
    }));
    document.querySelectorAll(".pol").forEach(el => el.addEventListener("change", () => {
      enabledPol.clear();
      document.querySelectorAll(".pol:checked").forEach(x => enabledPol.add(x.dataset.pol));
      refresh();
    }));
    document.getElementById("rel-edges").onchange = e => { showRel = e.target.checked; refresh(); };
    document.getElementById("id-edges").onchange = e => { showId = e.target.checked; refresh(); };
    document.getElementById("hide-isolates").onchange = e => { hideIsolates = e.target.checked; refresh(); };
    document.getElementById("focus-mode").onchange = e => { focusMode = e.target.checked; refresh(); };
    document.getElementById("weight").oninput = e => {
      minWeight = Number(e.target.value); document.getElementById("wval").textContent = minWeight.toFixed(2); refresh();
    };
    document.getElementById("fit").onclick = () => { if (renderer) renderer.getCamera().animatedReset({ duration: 500 }); };
    document.getElementById("layout").onclick = () => { Object.keys(pos).forEach(k => delete pos[k]); rebuild(); };
    document.getElementById("preset-people").onclick = () => {
      document.querySelectorAll(".layer").forEach(el => { el.checked = el.dataset.type !== "entity"; });
      enabledTypes.clear(); enabledTypes.add("user"); enabledTypes.add("server");
      hideIsolates = true; document.getElementById("hide-isolates").checked = true;
      document.getElementById("preset-people").classList.add("on");
      document.getElementById("preset-relations").classList.remove("on");
      document.getElementById("preset-all").classList.remove("on");
      refresh();
    };
    document.getElementById("preset-relations").onclick = () => {
      document.querySelectorAll(".layer").forEach(el => { el.checked = true; });
      enabledTypes.clear(); ["user", "entity", "server"].forEach(t => enabledTypes.add(t));
      hideIsolates = true; document.getElementById("hide-isolates").checked = true;
      document.getElementById("preset-relations").classList.add("on");
      document.getElementById("preset-people").classList.remove("on");
      document.getElementById("preset-all").classList.remove("on");
      refresh();
    };
    document.getElementById("preset-all").onclick = () => {
      document.querySelectorAll(".layer").forEach(el => { el.checked = true; });
      enabledTypes.clear(); ["user", "entity", "server"].forEach(t => enabledTypes.add(t));
      hideIsolates = false; document.getElementById("hide-isolates").checked = false;
      document.getElementById("preset-all").classList.add("on");
      document.getElementById("preset-people").classList.remove("on");
      document.getElementById("preset-relations").classList.remove("on");
      refresh();
    };
    document.querySelectorAll("[data-cat]").forEach(el => el.onclick = () => {
      catalog = el.dataset.cat;
      document.querySelectorAll("[data-cat]").forEach(x => x.classList.toggle("on", x === el));
      renderCatalog();
    });
    window.addEventListener("keydown", e => {
      if (e.key === "/" && document.activeElement?.tagName !== "INPUT") {
        e.preventDefault(); document.getElementById("search").focus();
      }
      if (e.key === "Escape") { document.getElementById("search").blur(); query = ""; document.getElementById("search").value = ""; select(null); refresh(); }
    });

    function canvasReady() {
      const el = document.getElementById("canvas");
      return el && el.clientWidth >= 8 && el.clientHeight >= 8;
    }
    async function waitForCanvas() {
      if (canvasReady()) return;
      await new Promise(resolve => {
        const el = document.getElementById("canvas");
        const ro = new ResizeObserver(() => {
          if (canvasReady()) { ro.disconnect(); resolve(); }
        });
        ro.observe(el);
        setTimeout(() => { ro.disconnect(); resolve(); }, 4000);
      });
    }
    await waitForCanvas();
    rebuild();
  </script>
</body>
</html>
"""
