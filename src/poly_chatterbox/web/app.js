// Poly Chatterbox web UI. Every action goes through the REST API under /api.
"use strict";

const STORAGE_KEY = "polychatterbox.workspaces.v1";

const MODES = [
  { id: "turbo", label: "Turbo", model: "turbo", hint: "Chatterbox-Turbo (English, 350M): fast, natural speech with paralinguistic tags such as [laugh], [sigh] and [cough]. Clone a voice from a clip of 5 seconds or more." },
  { id: "nano", label: "Nano", model: "nano", hint: "Chatterbox-Nano (English, 110M): Turbo's architecture in an even smaller model that runs well on CPU, with the same tags." },
  { id: "multilingual", label: "Multilingual", model: "multilingual", hint: "Chatterbox Multilingual: speech in 23 languages. Each language has a native sample speaker, or clone any voice across languages." },
  { id: "original", label: "Original", model: "original", hint: "The original Chatterbox (English, 500M), with creative control: raise exaggeration for drama and lower CFG / pace to slow it down." },
  { id: "vc", label: "Voice Conversion", hint: "Keep the words, timing and intonation of a recording but change the voice: upload or record speech, then pick the target voice." },
  { id: "watermark", label: "Watermark", hint: "Everything Chatterbox generates carries an imperceptible Perth watermark. Check whether a clip has one." },
  { id: "whisper", label: "Whisper", hint: "Speech to text with Whisper: upload or record audio and copy the transcript. The language is detected unless you pick one." },
];

const SAMPLE_TEXTS = {
  turbo: "Oh, that's hilarious! [chuckle] Um anyway, we do have a new model in store. It's the SkyNet T-800 series and it's got basically everything. Would you like me to get some prices for you?",
  original: "Now let's make my mum's favourite. So three mars bars into the pan. Then we add the tuna and just stir for a bit, just let the chocolate and fish infuse. Oh boy, this is going to be incredible.",
};

// Voice sources each TTS mode offers ("sample" = the multilingual language's own speaker).
const VOICE_SOURCES = {
  sample: "Language sample", builtin: "Built-in voice", library: "Library voice", upload: "Upload / record",
};
const sourcesFor = (mode) => (mode === "multilingual" ? ["sample", "builtin", "library", "upload"] : ["builtin", "library", "upload"]);

const app = {
  info: null,
  voices: [],
  prompts: [],      // saved prompts (GET /api/prompts)
  workspaces: [],   // persisted state objects
  runtime: {},      // id -> { el, files: {ref, source}, results, busy, recorder, recSlot }
  activeId: null,
};

// ------------------------------------------------------------------ utilities

const $ = (sel, root = document) => root.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const uid = () => Math.random().toString(36).slice(2, 10);
const modeOf = (ws) => MODES.find((m) => m.id === ws.mode) || MODES[0];
const langName = (code) => app.info?.languages?.[code] || code;

function setPath(obj, path, value) {
  const keys = path.split(".");
  let o = obj;
  for (const k of keys.slice(0, -1)) o = o[k] ??= {};
  o[keys.at(-1)] = value;
}

function newWorkspace(name) {
  return {
    id: uid(), name: name || `Workspace ${app.workspaces.length + 1}`, mode: "turbo",
    text: SAMPLE_TEXTS.turbo, language: "fr", version: "",
    voiceSource: "builtin", voiceId: "podcast", wrapVoice: "", promptId: "", saveName: "",
    voiceFilter: { q: "", kind: "" }, gen: {}, whisperLang: "", preset: {},
  };
}

function save() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ workspaces: app.workspaces, activeId: app.activeId }));
  } catch { /* storage unavailable: state just isn't remembered */ }
}

function load() {
  try {
    const data = JSON.parse(localStorage.getItem(STORAGE_KEY) || "null");
    if (data && Array.isArray(data.workspaces) && data.workspaces.length) {
      app.workspaces = data.workspaces.map((w) => ({ ...newWorkspace(), ...w }));
      app.activeId = data.activeId;
    }
  } catch { /* ignore */ }
  if (!app.workspaces.length) app.workspaces.push(newWorkspace("Workspace 1"));
  if (!app.workspaces.some((w) => w.id === app.activeId)) app.activeId = app.workspaces[0].id;
}

async function api(path, options = {}) {
  const res = await fetch(path, options);
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail, null, 1);
    } catch { /* not JSON */ }
    throw new Error(detail);
  }
  return res;
}

// ------------------------------------------------------------------ tabs

function renderTabs() {
  const tabs = $("#tabs");
  tabs.innerHTML = "";
  for (const ws of app.workspaces) {
    const rt = app.runtime[ws.id];
    const tab = document.createElement("div");
    tab.className = "tab" + (ws.id === app.activeId ? " active" : "") + (rt?.busy ? " busy" : "");
    tab.dataset.id = ws.id;
    tab.setAttribute("role", "tab");
    tab.setAttribute("aria-selected", ws.id === app.activeId);
    tab.title = "Double-click to rename, middle-click to close";
    tab.innerHTML = `<span class="busy-dot"></span><span class="name">${esc(ws.name)}</span>
      <button class="close" title="Close workspace" aria-label="Close ${esc(ws.name)}">×</button>`;
    tab.addEventListener("click", (e) => { if (!e.target.closest(".close")) activate(ws.id); });
    tab.addEventListener("auxclick", (e) => { if (e.button === 1) closeWorkspace(ws.id); });
    tab.addEventListener("dblclick", () => startRename(tab, ws));
    $(".close", tab).addEventListener("click", () => closeWorkspace(ws.id));
    tabs.appendChild(tab);
  }
}

// Update active/busy state without rebuilding the tabs (rebuilding would break double-click rename).
function syncTabs() {
  document.querySelectorAll("#tabs .tab").forEach((tab) => {
    const active = tab.dataset.id === app.activeId;
    tab.classList.toggle("active", active);
    tab.classList.toggle("busy", !!app.runtime[tab.dataset.id]?.busy);
    tab.setAttribute("aria-selected", active);
  });
}

function startRename(tab, ws) {
  const nameEl = $(".name", tab);
  const input = document.createElement("input");
  input.className = "rename";
  input.value = ws.name;
  nameEl.replaceWith(input);
  input.focus();
  input.select();
  const done = (commit) => {
    if (commit && input.value.trim()) ws.name = input.value.trim();
    save();
    renderTabs();
  };
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") done(true);
    if (e.key === "Escape") done(false);
  });
  input.addEventListener("blur", () => done(true));
}

function addWorkspace() {
  const ws = newWorkspace();
  app.workspaces.push(ws);
  mountWorkspace(ws);
  app.activeId = ws.id;
  renderTabs();
  activate(ws.id);
}

function closeWorkspace(id) {
  const rt = app.runtime[id];
  if (rt?.results.length && !confirm("Close this workspace? Its generated audio will be discarded.")) return;
  const idx = app.workspaces.findIndex((w) => w.id === id);
  app.workspaces.splice(idx, 1);
  if (rt) {
    rt.results.forEach((r) => r.url && URL.revokeObjectURL(r.url));
    Object.values(rt.files).forEach((f) => f?.url && URL.revokeObjectURL(f.url));
    rt.el.remove();
    delete app.runtime[id];
  }
  if (!app.workspaces.length) {
    const ws = newWorkspace("Workspace 1");
    app.workspaces.push(ws);
    mountWorkspace(ws);
  }
  if (app.activeId === id) app.activeId = app.workspaces[Math.min(idx, app.workspaces.length - 1)].id;
  renderTabs();
  activate(app.activeId);
}

function activate(id) {
  app.activeId = id;
  for (const [wid, rt] of Object.entries(app.runtime)) rt.el.classList.toggle("active", wid === id);
  syncTabs();
  save();
}

// ------------------------------------------------------------------ workspace

function mountWorkspace(ws) {
  const el = $("#workspace-template").content.firstElementChild.cloneNode(true);
  el.dataset.id = ws.id;
  $("#workspaces").appendChild(el);
  app.runtime[ws.id] = { el, files: { ref: null, source: null }, results: [], busy: false, recorder: null, recSlot: null };
  $(".clear-results", el).addEventListener("click", () => {
    const rt = app.runtime[ws.id];
    rt.results.forEach((r) => r.url && URL.revokeObjectURL(r.url));
    rt.results = [];
    renderResults(ws);
  });
  renderModes(ws);
  renderForm(ws);
  renderResults(ws);
}

function renderModes(ws) {
  const el = app.runtime[ws.id].el;
  const modes = $(".modes", el);
  modes.innerHTML = MODES.map((m) =>
    `<button type="button" class="mode" role="radio" aria-checked="${m.id === modeOf(ws).id}" data-mode="${m.id}">${m.label}</button>`).join("");
  modes.querySelectorAll(".mode").forEach((b) => b.addEventListener("click", () => {
    ws.mode = b.dataset.mode;
    save();
    renderModes(ws);
    renderForm(ws);
  }));
  $(".mode-hint", el).textContent = modeOf(ws).hint;
}

// ---------- form building blocks (return HTML; bindForm wires data-k inputs to state)

function textField(ws, model) {
  const tags = model === "turbo" || model === "nano";
  const max = app.info?.max_chunk_chars ?? 300;
  return `<div class="field"><label for="${ws.id}-text">Text</label>
    <div class="prompt-tools">
      <select data-k="promptId" aria-label="Saved prompts">
        <option value="">${app.prompts.length ? `Saved prompts (${app.prompts.length})…` : "No saved prompts yet"}</option>
        ${app.prompts.map((p) => `<option value="${esc(p.id)}" ${p.id === ws.promptId ? "selected" : ""}>${esc(p.name)}${p.mode ? " · " + esc(p.mode) : ""}</option>`).join("")}
      </select>
      <button type="button" class="btn small" data-action="load-prompt" title="Replace this workspace's text, mode, voice and settings with the saved prompt">Load</button>
      <button type="button" class="btn small" data-action="save-prompt" title="Save the text, mode, language, voice and changed settings as a prompt">Save…</button>
      ${ws.promptId ? `<button type="button" class="link danger" data-action="delete-prompt">Delete</button>` : ""}
    </div>
    <textarea id="${ws.id}-text" class="main-text" data-k="text" placeholder="What should be said?">${esc(ws.text)}</textarea>
    ${tags ? `<div class="chips" aria-label="Insert a paralinguistic tag">${(app.info?.event_tags || []).map((t) =>
      `<button type="button" class="chip tag" data-tag="${esc(t)}">${esc(t)}</button>`).join("")}</div>` : ""}
    <div class="markup-tools">
      <button type="button" class="btn small" data-action="wrap-settings" title="Wraps the highlighted text so only it uses your changed generation settings">Wrap selection in settings</button>
      <span class="sep"></span>
      <select data-k="wrapVoice" aria-label="Voice for the selection">${voiceOptions(ws.wrapVoice || ws.voiceId)}</select>
      <button type="button" class="btn small" data-action="wrap-voice" title="Wraps the highlighted text so only it is spoken by the chosen voice">Wrap selection in voice</button>
    </div>
    <div class="help settings-preview">${settingsPreview(ws)}</div>
    <div class="help"><code>[pause 800ms]</code> inserts silence. Text longer than ${max} characters is split at sentences and joined.
      ${tags ? "Click a tag to insert it at the cursor. " : ""}Highlight some text and use the buttons above to give just that part
      its own generation settings (<code>[settings …]…[/settings]</code>) or voice (<code>[voice: id]…[/voice]</code>).</div></div>`;
}

// All voices as <option>s (value = voice id) for the "wrap selection in voice" picker.
function voiceOptions(selected) {
  return app.voices.map((v) => `<option value="${esc(v.id)}" ${v.id === selected ? "selected" : ""}>${esc(v.name)}${v.language ? " — " + esc(langName(v.language)) : ""}</option>`).join("");
}

// The opening [settings …] tag for the generation options the user has changed in this mode ("" if none).
function settingsTag(ws) {
  const g = ws.gen[ws.mode] || {};
  const parts = genOptions(modeOf(ws).model)
    .filter(([k]) => k !== "seed" && g[k] != null && g[k] !== "" && !Number.isNaN(g[k]))
    .map(([k]) => `${k}=${g[k]}`);
  return parts.length ? `[settings ${parts.join(" ")}]` : "";
}

function settingsPreview(ws) {
  const tag = settingsTag(ws);
  return tag ? `Settings block you have set up: <code>${esc(tag)}</code>`
    : "Change a slider or a Generation option below, then wrap the text that should use it. (The seed can't be set per span.)";
}

// Wrap the highlighted text of the text box in open/close markup (with no selection: an empty pair
// at the cursor). Whitespace at the edges of the selection stays outside the tags.
function wrapSelection(ws, form, open, close) {
  const area = form.querySelector("textarea[data-k='text']");
  const text = area.value;
  let start = area.selectionStart ?? text.length, end = area.selectionEnd ?? text.length;
  const sel = text.slice(start, end);
  start += sel.length - sel.trimStart().length;
  end -= sel.length - sel.trimEnd().length;
  if (end < start) end = start;
  ws.text = text.slice(0, start) + open + text.slice(start, end) + close + text.slice(end);
  save();
  renderForm(ws);
  const again = form.querySelector("textarea[data-k='text']");
  const pos = end === start ? start + open.length : end + open.length + close.length;
  again.focus();
  again.setSelectionRange(pos, pos);
}

function languageField(ws) {
  const langs = Object.entries(app.info?.languages || { en: "English" });
  const versions = app.info?.multilingual_versions || { default: "v3", choices: ["v2", "v3"] };
  return `<div class="row"><div class="field"><label>Language</label><select data-k="language">
      ${langs.map(([code, name]) => `<option value="${code}" ${code === ws.language ? "selected" : ""}>${esc(name)} (${code})</option>`).join("")}
    </select><button type="button" class="link" data-action="sample-text">Insert sample sentence</button></div>
    <div class="field"><label>Model version</label><select data-k="version">
      <option value="">Server default (${esc(versions.default)})</option>
      ${versions.choices.map((v) => `<option value="${v}" ${ws.version === v ? "selected" : ""}>${v}${v === "v3" ? " (recommended)" : ""}</option>`).join("")}
    </select></div></div>`;
}

function voicePicker(ws) {
  const f = ws.voiceFilter;
  const q = (f.q || "").toLowerCase();
  const voices = app.voices.filter((v) => (!f.kind || v.kind === f.kind)
    && (!q || `${v.id} ${v.name} ${v.description} ${langName(v.language)}`.toLowerCase().includes(q)));
  const groups = { builtin: "Built-in", sample: "Demo voices", cloned: "My cloned voices" };
  const options = Object.entries(groups).map(([kind, title]) => {
    const vs = voices.filter((v) => v.kind === kind);
    if (!vs.length) return "";
    return `<optgroup label="${title} (${vs.length})">` + vs.map((v) =>
      `<option value="${esc(v.id)}" ${v.id === ws.voiceId ? "selected" : ""}>${esc(v.name)}${v.language ? " — " + esc(langName(v.language)) : ""}${v.gender !== "unknown" ? " · " + v.gender : ""}</option>`).join("") + "</optgroup>";
  }).join("");
  const current = app.voices.find((v) => v.id === ws.voiceId);
  return `<div class="field voice-picker"><label>Voice</label>
    <div class="voice-filter">
      <input type="text" data-filter="q" placeholder="Search ${app.voices.length} voices…" value="${esc(f.q)}">
      <select data-filter="kind"><option value="">All kinds</option>${Object.entries(groups).map(([k, t]) => `<option value="${k}" ${f.kind === k ? "selected" : ""}>${t}</option>`).join("")}</select>
    </div>
    <select size="8" data-k="voiceId" aria-label="Voice">${options || "<option disabled>No matching voices</option>"}</select>
    <div class="voice-desc"><span class="grow">${current ? esc(current.description) : "Select a voice."}</span>
      ${current?.kind === "cloned" ? `<button type="button" class="link danger" data-action="delete-voice">Delete</button>` : ""}</div>
  </div>`;
}

function audioField(ws, slot, label, hint) {
  const rt = app.runtime[ws.id];
  const f = rt.files[slot];
  const min = app.info?.reference_audio?.min_seconds ?? 5;
  let info = hint;
  let bad = false;
  if (f) {
    info = esc(f.file.name) + (f.seconds != null ? ` · ${f.seconds.toFixed(1)} s` : "");
    if (slot === "ref" && f.seconds != null && f.seconds < min) { info += ` (needs at least ${min} s)`; bad = true; }
  }
  const recording = rt.recorder && rt.recSlot === slot;
  return `<div class="field"><label>${label}</label><div class="ref-box">
      <div class="ref-actions">
        <input type="file" accept="audio/*" data-file="${slot}" aria-label="${label}">
        <button type="button" class="btn rec ${recording ? "recording" : ""}" data-record="${slot}" ${rt.recorder && !recording ? "disabled" : ""}>${recording ? "■ Stop recording" : "● Record"}</button>
      </div>
      <div class="ref-info ${bad ? "bad" : ""}">${info}</div>
      ${f ? `<audio controls src="${f.url}"></audio>` : ""}
    </div></div>`;
}

function voiceField(ws, { label = "Voice", target = false } = {}) {
  const sources = target ? ["builtin", "library", "upload"] : sourcesFor(ws.mode);
  const source = sources.includes(ws.voiceSource) ? ws.voiceSource : sources[0];
  const seg = (value) => `<button type="button" data-seg="voiceSource" data-value="${value}" class="${source === value ? "on" : ""}">${VOICE_SOURCES[value]}</button>`;
  let body = "";
  if (source === "sample") body = `<div class="help">A native ${esc(langName(ws.language))} speaker from the upstream demo (downloaded on first use).</div>`;
  if (source === "builtin") body = `<div class="help">The voice that ships with the model. No reference clip needed.</div>`;
  if (source === "library") body = voicePicker(ws);
  if (source === "upload") {
    body = audioField(ws, "ref", target ? "Target voice clip" : "Reference clip",
      "No clip yet. Upload or record one: about 10 seconds of clean speech works best (5 s minimum).") +
      `<details><summary>Save to voice library</summary>
        <div class="field"><label>Name</label><input type="text" data-k="saveName" value="${esc(ws.saveName)}" placeholder="e.g. My narrator"></div>
        <button type="button" class="btn" data-action="save-voice">Save cloned voice</button>
        <div class="help">Saved voices appear under Library voice in every mode.</div>
      </details>`;
  }
  return `<div class="field"><span class="label">${label}</span><div class="segmented">${sources.map(seg).join("")}</div></div>${body}`;
}

function genOptions(model) {
  return Object.entries(app.info?.generation_options || {}).filter(([, o]) => model in o.defaults);
}

// Emotion presets (GET /api/info → presets): picking one fills in the generation options the model uses.
function presetField(ws, model) {
  const presets = app.info?.presets || [];
  if (!presets.length) return "";
  const current = presets.find((p) => p.id === ws.preset[ws.mode]);
  const used = genOptions(model).map(([k]) => k);
  return `<div class="field"><label for="${ws.id}-preset">Emotion preset</label>
    <select id="${ws.id}-preset" data-preset>
      <option value="">Custom (your own settings)</option>
      ${presets.map((p) => `<option value="${esc(p.id)}" ${p.id === current?.id ? "selected" : ""}>${esc(p.label)}</option>`).join("")}
    </select>
    <div class="help">${current ? esc(current.description) + " Sets " + Object.keys(current.options).filter((k) => used.includes(k)).map((k) => esc(app.info.generation_options[k]?.label || k)).join(", ") + "; you can still adjust them below."
      : "Starts the settings below from a mood. " + (model === "turbo" || model === "nano" ? "Turbo and Nano have no exaggeration or CFG, so presets only change their sampling." : "")}</div></div>`;
}

function applyPreset(ws, id) {
  const preset = (app.info?.presets || []).find((p) => p.id === id);
  ws.preset[ws.mode] = preset ? preset.id : "";
  if (preset) {
    const model = modeOf(ws).model;
    const used = genOptions(model).map(([k]) => k);
    const gen = {};
    if (ws.gen[ws.mode]?.seed != null) gen.seed = ws.gen[ws.mode].seed;
    for (const [k, v] of Object.entries(preset.options)) if (used.includes(k)) gen[k] = v;
    ws.gen[ws.mode] = gen;
  }
  save();
  renderForm(ws);
}

function slidersField(ws, model) {
  const g = ws.gen[ws.mode] || {};
  const fields = genOptions(model).filter(([, o]) => o.primary).map(([k, o]) => {
    const id = `${ws.id}-gen-${k}`;
    const value = g[k] ?? o.defaults[model];
    return `<div class="slider"><div class="label-row"><label for="${id}">${esc(o.label)}</label><output>${value}</output></div>
      <input id="${id}" type="range" min="${o.min}" max="${o.max}" step="${o.step}" value="${value}" data-k="gen.${ws.mode}.${k}" data-num data-slider aria-describedby="${id}-help">
      <div class="help" id="${id}-help">${esc(o.description)} <span class="default">Default: ${o.defaults[model]}.</span></div></div>`;
  }).join("");
  return fields ? `<div class="sliders">${fields}</div>` : "";
}

function generationField(ws, model) {
  const g = ws.gen[ws.mode] || {};
  // Labels, ranges, defaults and tuning advice come from the server (GET /api/info → generation_options).
  const fields = genOptions(model).filter(([, o]) => !o.primary).map(([k, o]) => {
    const id = `${ws.id}-gen-${k}`;
    const def = o.defaults[model];
    const help = `<div class="help" id="${id}-help">${esc(o.description)}${def != null ? ` <span class="default">Default: ${def}.</span>` : ""}</div>`;
    if (o.type === "bool") {
      const v = g[k] == null ? "" : String(g[k]);
      return `<div class="gen-option"><label class="label" for="${id}">${esc(o.label)}</label>
        <select id="${id}" data-k="gen.${ws.mode}.${k}" data-bool aria-describedby="${id}-help">
          <option value="" ${v === "" ? "selected" : ""}>Default (${def ? "on" : "off"})</option>
          <option value="true" ${v === "true" ? "selected" : ""}>On</option><option value="false" ${v === "false" ? "selected" : ""}>Off</option>
        </select>${help}</div>`;
    }
    return `<div class="gen-option"><label class="label" for="${id}">${esc(o.label)}</label>
      <input id="${id}" type="number" step="${o.step}" min="${o.min}" ${o.max != null ? `max="${o.max}"` : ""} data-k="gen.${ws.mode}.${k}" data-num
        value="${g[k] ?? ""}" placeholder="${def ?? "random"}" aria-describedby="${id}-help">${help}</div>`;
  }).join("");
  return `<details><summary>Generation options</summary>
    <p class="help gen-intro">These change how the model samples speech. Leave a box empty to use the model default (shown in grey).
      Set a seed first, then change one option at a time to hear what it does.</p>
    <div class="gen-grid">${fields || `<p class="help">Generation options are unavailable (the API could not be reached).</p>`}</div>
    <button type="button" class="link" data-action="reset-gen">Reset all to defaults</button></details>`;
}

function renderForm(ws) {
  const rt = app.runtime[ws.id];
  const form = $(".form", rt.el);
  const mode = modeOf(ws);
  let html = "";
  let submit = "Generate";
  if (mode.model) {
    html = (mode.model === "multilingual" ? languageField(ws) : "") + voiceField(ws) + textField(ws, mode.model)
      + presetField(ws, mode.model) + slidersField(ws, mode.model) + generationField(ws, mode.model);
  } else if (mode.id === "vc") {
    html = audioField(ws, "source", "Speech to convert", "No recording yet. Upload or record the speech whose voice you want to change.")
      + voiceField(ws, { label: "Target voice", target: true });
    submit = "Convert";
  } else if (mode.id === "watermark") {
    html = audioField(ws, "source", "Audio to check", "Upload or record the clip to check.");
    submit = "Check for watermark";
  } else if (mode.id === "whisper") {
    const langs = Object.entries(app.info?.languages || {});
    html = audioField(ws, "source", "Audio to transcribe", "Upload or record the speech to transcribe.")
      + `<div class="field"><label for="${ws.id}-wlang">Language spoken</label><select id="${ws.id}-wlang" data-k="whisperLang">
          <option value="">Detect automatically</option>
          ${langs.map(([code, name]) => `<option value="${code}" ${code === ws.whisperLang ? "selected" : ""}>${esc(name)} (${code})</option>`).join("")}
        </select><div class="help">Picking the language avoids mistakes on short or noisy clips.</div></div>`;
    submit = "Transcribe";
  }
  html += `<div class="actions">
      <button type="submit" class="primary" ${rt.busy ? "disabled" : ""}>${rt.busy ? '<span class="spinner"></span> Working…' : submit}</button>
      <span class="kbd">Ctrl+Enter</span></div><div class="error" role="alert"></div>`;
  // Keep collapsible sections open/closed across re-renders.
  const openState = new Map([...form.querySelectorAll("details")].map((d) => [d.querySelector("summary").textContent, d.open]));
  form.innerHTML = html;
  form.querySelectorAll("details").forEach((d) => {
    const prev = openState.get(d.querySelector("summary").textContent);
    if (prev !== undefined) d.open = prev;
  });
  bindForm(ws, form);
}

function bindForm(ws, form) {
  const rt = app.runtime[ws.id];
  form.onsubmit = (e) => { e.preventDefault(); generate(ws); };
  form.onkeydown = (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); generate(ws); } };

  form.querySelectorAll("[data-k]").forEach((input) => {
    const key = input.dataset.k;
    const handler = () => {
      let value = input.value;
      if (input.hasAttribute("data-num")) value = value === "" ? null : Number(value);
      if (input.hasAttribute("data-bool")) value = value === "" ? null : value === "true";
      setPath(ws, key, value);
      if (input.hasAttribute("data-slider")) input.closest(".slider").querySelector("output").textContent = input.value;
      if (key.startsWith("gen.")) {
        if (ws.preset[ws.mode]) {  // the user moved off the preset's values
          ws.preset[ws.mode] = "";
          const picker = form.querySelector("[data-preset]");
          if (picker) picker.value = "";
        }
        const preview = form.querySelector(".settings-preview");
        if (preview) preview.innerHTML = settingsPreview(ws);
      }
      save();
      if (input.tagName === "SELECT" && ["voiceId", "language"].includes(key)) renderForm(ws);
    };
    input.addEventListener(input.tagName === "SELECT" ? "change" : "input", handler);
  });
  form.querySelector("[data-preset]")?.addEventListener("change", (e) => applyPreset(ws, e.target.value));
  form.querySelectorAll("[data-filter]").forEach((input) => {
    input.addEventListener(input.tagName === "SELECT" ? "change" : "input", () => {
      ws.voiceFilter[input.dataset.filter] = input.value;
      save();
      const pos = input.selectionStart;
      renderForm(ws);
      const again = form.querySelector(`[data-filter="${input.dataset.filter}"]`);
      again.focus();
      if (pos != null && again.setSelectionRange) again.setSelectionRange(pos, pos);
    });
  });
  form.querySelectorAll("[data-seg]").forEach((b) => b.addEventListener("click", () => {
    ws[b.dataset.seg] = b.dataset.value;
    save();
    renderForm(ws);
  }));
  form.querySelectorAll("[data-tag]").forEach((chip) => chip.addEventListener("click", () => insertTag(ws, form, chip.dataset.tag)));
  form.querySelectorAll("[data-file]").forEach((input) => input.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) setAudio(ws, input.dataset.file, file);
  }));
  form.querySelectorAll("[data-record]").forEach((b) => b.addEventListener("click", () => toggleRecording(ws, b.dataset.record)));
  form.querySelector("[data-action='reset-gen']")?.addEventListener("click", () => {
    ws.gen[ws.mode] = {};
    ws.preset[ws.mode] = "";
    save();
    renderForm(ws);
  });
  form.querySelector("[data-action='wrap-settings']")?.addEventListener("click", () => {
    const tag = settingsTag(ws);
    if (!tag) return showError(ws, "Change a slider or a Generation option first; the wrapped text will use those settings.");
    showError(ws, "");
    wrapSelection(ws, form, tag, "[/settings]");
  });
  form.querySelector("[data-action='wrap-voice']")?.addEventListener("click", () => {
    const id = ws.wrapVoice || ws.voiceId || app.voices[0]?.id;
    if (!id) return showError(ws, "Pick a voice for the selection first.");
    showError(ws, "");
    wrapSelection(ws, form, `[voice: ${id}]`, "[/voice]");
  });
  form.querySelector("[data-action='load-prompt']")?.addEventListener("click", () => loadPrompt(ws));
  form.querySelector("[data-action='save-prompt']")?.addEventListener("click", () => savePrompt(ws));
  form.querySelector("[data-action='delete-prompt']")?.addEventListener("click", () => deletePrompt(ws));
  form.querySelector("[data-action='save-voice']")?.addEventListener("click", () => saveVoice(ws));
  form.querySelector("[data-action='delete-voice']")?.addEventListener("click", () => deleteVoice(ws));
  form.querySelector("[data-action='sample-text']")?.addEventListener("click", () => {
    ws.text = app.info?.sample_texts?.[ws.language] || ws.text;
    save();
    renderForm(ws);
  });
  rt.errorEl = form.querySelector(".error");
}

// Insert a paralinguistic tag at the cursor, padding it with spaces where needed.
function insertTag(ws, form, tag) {
  const area = form.querySelector("textarea[data-k='text']");
  const text = area.value;
  const start = area.selectionStart ?? text.length, end = area.selectionEnd ?? text.length;
  const prefix = start === 0 || text[start - 1] === " " ? "" : " ";
  const suffix = end < text.length && text[end] === " " ? "" : " ";
  ws.text = text.slice(0, start) + prefix + tag + suffix + text.slice(end);
  save();
  renderForm(ws);
  const again = form.querySelector("textarea[data-k='text']");
  const pos = start + prefix.length + tag.length + suffix.length;
  again.focus();
  again.setSelectionRange(pos, pos);
}

// ------------------------------------------------------------------ audio inputs

async function setAudio(ws, slot, file) {
  const rt = app.runtime[ws.id];
  if (rt.files[slot]?.url) URL.revokeObjectURL(rt.files[slot].url);
  const entry = { file, url: URL.createObjectURL(file), seconds: null };
  rt.files[slot] = entry;
  renderForm(ws);
  try {
    const ctx = new AudioContext();
    const buf = await ctx.decodeAudioData(await file.arrayBuffer());
    entry.seconds = buf.duration;
    ctx.close();
  } catch { /* the server validates anyway */ }
  renderForm(ws);
}

function encodeWav(buffer) {
  // Downmix to mono 16-bit PCM WAV so the server can always decode it.
  const ch = buffer.numberOfChannels, len = buffer.length, sr = buffer.sampleRate;
  const mono = new Float32Array(len);
  for (let c = 0; c < ch; c++) { const d = buffer.getChannelData(c); for (let i = 0; i < len; i++) mono[i] += d[i] / ch; }
  const out = new DataView(new ArrayBuffer(44 + len * 2));
  const str = (o, s) => [...s].forEach((c, i) => out.setUint8(o + i, c.charCodeAt(0)));
  str(0, "RIFF"); out.setUint32(4, 36 + len * 2, true); str(8, "WAVE"); str(12, "fmt ");
  out.setUint32(16, 16, true); out.setUint16(20, 1, true); out.setUint16(22, 1, true);
  out.setUint32(24, sr, true); out.setUint32(28, sr * 2, true); out.setUint16(32, 2, true); out.setUint16(34, 16, true);
  str(36, "data"); out.setUint32(40, len * 2, true);
  for (let i = 0; i < len; i++) out.setInt16(44 + i * 2, Math.max(-1, Math.min(1, mono[i])) * 0x7fff, true);
  return new Blob([out], { type: "audio/wav" });
}

async function toggleRecording(ws, slot) {
  const rt = app.runtime[ws.id];
  if (rt.recorder) { rt.recorder.stop(); return; }
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (e) {
    showError(ws, `Microphone unavailable: ${e.message}. Browsers only allow recording on HTTPS pages or localhost.`);
    return;
  }
  const chunks = [];
  const recorder = new MediaRecorder(stream);
  recorder.ondataavailable = (e) => chunks.push(e.data);
  recorder.onstop = async () => {
    stream.getTracks().forEach((t) => t.stop());
    rt.recorder = null;
    rt.recSlot = null;
    try {
      const ctx = new AudioContext();
      const buf = await ctx.decodeAudioData(await new Blob(chunks).arrayBuffer());
      ctx.close();
      await setAudio(ws, slot, new File([encodeWav(buf)], "recording.wav", { type: "audio/wav" }));
    } catch (e) {
      showError(ws, `Could not process recording: ${e.message}`);
      renderForm(ws);
    }
  };
  rt.recorder = recorder;
  rt.recSlot = slot;
  recorder.start();
  renderForm(ws);
}

// ------------------------------------------------------------------ voices

async function loadVoices() {
  app.voices = await (await api("/api/voices")).json();
}

async function refreshVoices() {
  await loadVoices();
  for (const ws of app.workspaces) renderForm(ws);
}

async function saveVoice(ws) {
  const rt = app.runtime[ws.id];
  showError(ws, "");
  if (!rt.files.ref) return showError(ws, "Add a reference clip first.");
  if (!ws.saveName.trim()) return showError(ws, "Give the voice a name.");
  const fd = new FormData();
  fd.append("name", ws.saveName.trim());
  fd.append("ref_audio", rt.files.ref.file);
  if (ws.mode === "multilingual") fd.append("language", ws.language);
  else if (modeOf(ws).model) fd.append("language", "en");
  try {
    const voice = await (await api("/api/voices", { method: "POST", body: fd })).json();
    ws.voiceId = voice.id;
    ws.voiceSource = "library";
    ws.saveName = "";
    save();
    await refreshVoices();
    alert(`Saved "${voice.name}" to the voice library (id: ${voice.id}).`);
  } catch (e) {
    showError(ws, e.message);
  }
}

async function deleteVoice(ws) {
  const voice = app.voices.find((v) => v.id === ws.voiceId);
  if (!voice || !confirm(`Delete cloned voice "${voice.name}"?`)) return;
  try {
    await api(`/api/voices/${encodeURIComponent(voice.id)}`, { method: "DELETE" });
    await refreshVoices();
  } catch (e) {
    showError(ws, e.message);
  }
}

// ------------------------------------------------------------------ saved prompts

async function refreshPrompts() {
  app.prompts = await (await api("/api/prompts")).json();
  for (const ws of app.workspaces) renderForm(ws);
}

// Which voice source the form is really using (a mode may not offer the stored one).
function effectiveSource(ws) {
  const sources = sourcesFor(ws.mode);
  return sources.includes(ws.voiceSource) ? ws.voiceSource : sources[0];
}

async function savePrompt(ws) {
  showError(ws, "");
  if (!ws.text.trim()) return showError(ws, "There is no text to save.");
  const current = app.prompts.find((p) => p.id === ws.promptId);
  const name = (window.prompt("Name for this prompt (an existing name is replaced):", current?.name || "") || "").trim();
  if (!name) return;
  if (app.prompts.some((p) => p.name.toLowerCase() === name.toLowerCase()) && !confirm(`Replace the saved prompt "${name}"?`)) return;
  const source = effectiveSource(ws);
  const body = { name, text: ws.text, mode: modeOf(ws).model };
  if (modeOf(ws).model === "multilingual") {
    body.language = ws.language;
    if (ws.version) body.version = ws.version;
  }
  if (source === "library") body.voice_id = ws.voiceId;
  if (source === "builtin") body.voice_id = "default";
  const generation = cleanGen(ws.gen[ws.mode]);
  if (generation) body.generation = generation;
  try {
    const saved = await (await api("/api/prompts", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })).json();
    ws.promptId = saved.id;
    save();
    await refreshPrompts();
    if (source === "upload") showError(ws, "Saved. An uploaded clip isn't stored with a prompt: save it to the voice library to include the voice.");
  } catch (e) {
    showError(ws, e.message);
  }
}

function loadPrompt(ws) {
  const p = app.prompts.find((x) => x.id === ws.promptId);
  if (!p) return showError(ws, "Choose a saved prompt to load first.");
  if (p.mode && MODES.some((m) => m.id === p.mode)) ws.mode = p.mode;
  ws.text = p.text;
  if (p.language) ws.language = p.language;
  ws.version = p.version || "";
  if (p.voice_id === "default") ws.voiceSource = "builtin";
  else if (p.voice_id) { ws.voiceSource = "library"; ws.voiceId = p.voice_id; }
  else if (ws.mode === "multilingual") ws.voiceSource = "sample";
  ws.gen[ws.mode] = { ...(p.generation || {}) };
  ws.preset[ws.mode] = "";
  save();
  renderModes(ws);
  renderForm(ws);
  if (p.voice_id && p.voice_id !== "default" && !app.voices.some((v) => v.id === p.voice_id)) {
    showError(ws, `Loaded, but the voice "${p.voice_id}" is no longer in the library.`);
  }
}

async function deletePrompt(ws) {
  const p = app.prompts.find((x) => x.id === ws.promptId);
  if (!p || !confirm(`Delete the saved prompt "${p.name}"?`)) return;
  try {
    await api(`/api/prompts/${encodeURIComponent(p.id)}`, { method: "DELETE" });
    for (const w of app.workspaces) if (w.promptId === p.id) w.promptId = "";
    save();
    await refreshPrompts();
  } catch (e) {
    showError(ws, e.message);
  }
}

// ------------------------------------------------------------------ generation

function cleanGen(gen) {
  const out = {};
  for (const [k, v] of Object.entries(gen || {})) if (v !== null && v !== "" && !Number.isNaN(v)) out[k] = v;
  return Object.keys(out).length ? out : null;
}

// Adds the chosen voice to the form data; returns a label for the result list.
function appendVoice(ws, fd, { audioField, voiceField, target = false }) {
  const rt = app.runtime[ws.id];
  const sources = target ? ["builtin", "library", "upload"] : sourcesFor(ws.mode);
  const source = sources.includes(ws.voiceSource) ? ws.voiceSource : sources[0];
  if (source === "upload") {
    if (!rt.files.ref) throw new Error("Add a reference clip (upload or record), or pick another voice source.");
    fd.append(audioField, rt.files.ref.file);
    return rt.files.ref.file.name;
  }
  if (source === "library") {
    if (!ws.voiceId) throw new Error("Pick a library voice.");
    fd.append(voiceField, ws.voiceId);
    return app.voices.find((v) => v.id === ws.voiceId)?.name || ws.voiceId;
  }
  if (source === "builtin") {
    fd.append(voiceField, "default");
    return "built-in voice";
  }
  return `${langName(ws.language)} sample`;
}

function buildRequest(ws) {
  const rt = app.runtime[ws.id];
  const mode = modeOf(ws);
  const fd = new FormData();
  if (mode.model) {
    fd.append("text", ws.text);
    const voice = appendVoice(ws, fd, { audioField: "ref_audio", voiceField: "voice_id" });
    const generation = cleanGen(ws.gen[ws.mode]);
    if (generation) fd.append("generation", JSON.stringify(generation));
    if (mode.model === "multilingual") {
      fd.append("language", ws.language);
      if (ws.version) fd.append("version", ws.version);
      return { url: "/api/tts/multilingual", init: { method: "POST", body: fd }, label: `Multilingual · ${langName(ws.language)} · ${voice}`, text: ws.text };
    }
    return { url: `/api/tts/${mode.model}`, init: { method: "POST", body: fd }, label: `${mode.label} · ${voice}`, text: ws.text };
  }
  if (!rt.files.source) throw new Error(mode.id === "vc" ? "Add the speech to convert (upload or record)." : mode.id === "whisper" ? "Add the audio to transcribe." : "Add the audio to check.");
  fd.append(mode.id === "vc" ? "source_audio" : "audio", rt.files.source.file);
  if (mode.id === "whisper") {
    if (ws.whisperLang) fd.append("language", ws.whisperLang);
    return { url: "/api/transcribe", init: { method: "POST", body: fd }, label: `Whisper · ${rt.files.source.file.name}`, text: "", json: true };
  }
  if (mode.id === "watermark") {
    return { url: "/api/watermark", init: { method: "POST", body: fd }, label: `Watermark · ${rt.files.source.file.name}`, text: "", json: true };
  }
  const voice = appendVoice(ws, fd, { audioField: "target_audio", voiceField: "target_voice_id", target: true });
  return { url: "/api/vc", init: { method: "POST", body: fd }, label: `Conversion · ${rt.files.source.file.name} → ${voice}`, text: "" };
}

function showError(ws, msg) {
  const el = app.runtime[ws.id].errorEl;
  if (el) el.textContent = msg;
}

async function generate(ws) {
  const rt = app.runtime[ws.id];
  if (rt.busy) return;
  const mode = modeOf(ws);
  showError(ws, "");
  let req;
  try {
    req = buildRequest(ws);
  } catch (e) {
    return showError(ws, e.message);
  }
  const entry = { id: uid(), label: req.label, text: req.text, pending: true, started: Date.now() };
  rt.results.unshift(entry);
  rt.busy = true;
  renderResults(ws);
  renderForm(ws);
  syncTabs();
  try {
    const res = await api(req.url, req.init);
    if (req.json) {
      const body = await res.json();
      Object.assign(entry, { pending: false, ...(mode.id === "whisper" ? { transcript: body } : { watermark: body }) });
    } else {
      const meta = JSON.parse(decodeURIComponent(res.headers.get("X-TTS-Metadata") || "%7B%7D"));
      const blob = await res.blob();
      Object.assign(entry, { pending: false, url: URL.createObjectURL(blob), meta });
    }
  } catch (e) {
    Object.assign(entry, { pending: false, error: e.message });
  } finally {
    rt.busy = false;
    renderResults(ws);
    renderForm(ws);
    syncTabs();
  }
}

function renderResults(ws) {
  const rt = app.runtime[ws.id];
  const list = $(".result-list", rt.el);
  if (!rt.results.length) {
    list.innerHTML = `<p class="empty">Generated audio appears here.</p>`;
    return;
  }
  const head = (r) => `<div class="title"><span>${esc(r.label)}</span><button class="link" data-remove="${r.id}">remove</button></div>
    ${r.text ? `<div class="text">${esc(r.text)}</div>` : ""}`;
  list.innerHTML = rt.results.map((r, i) => {
    if (r.pending) return `<div class="result pending"><div class="title"><span>${esc(r.label)}</span><span class="spinner"></span></div>
      ${r.text ? `<div class="text">${esc(r.text)}</div>` : ""}<div class="meta">Working… the first request may take a while to download and load the model.</div></div>`;
    if (r.error) return `<div class="result">${head(r)}<div class="error">${esc(r.error)}</div></div>`;
    if (r.transcript) {
      const t = r.transcript;
      return `<div class="result">${head(r)}<textarea class="transcript" readonly rows="6" aria-label="Transcript">${esc(t.text)}</textarea>
        <div class="meta"><span>${t.duration} s audio</span><span>${t.elapsed} s to transcribe</span>${t.language ? `<span>${esc(langName(t.language))}</span>` : ""}</div>
        <div class="links"><button type="button" class="link" data-copy="${r.id}">Copy text</button></div></div>`;
    }
    if (r.watermark) {
      const w = r.watermark;
      return `<div class="result">${head(r)}<div class="score ${w.watermarked ? "yes" : ""}">${w.watermarked ? "Watermarked" : "No watermark"}</div>
        <div class="meta"><span>score ${w.score}</span><span>${w.watermarked ? "This clip was generated by Chatterbox." : "No Chatterbox watermark was found."}</span></div></div>`;
    }
    const m = r.meta || {};
    const file = `chatterbox-${m.mode || "audio"}-${rt.results.length - i}.wav`;
    return `<div class="result">${head(r)}
      <audio controls preload="metadata" src="${r.url}" ${i === 0 && !r.played ? "autoplay" : ""}></audio>
      <div class="meta"><span>${m.duration ?? "?"} s audio</span><span>${m.elapsed ?? "?"} s to generate</span>${m.model ? `<span>${esc(m.model)}</span>` : ""}${m.segments > 1 ? `<span>${m.segments} chunks</span>` : ""}</div>
      ${(m.warnings || []).map((w) => `<div class="warn">${esc(w)}</div>`).join("")}
      <div class="links"><a href="${r.url}" download="${file}">Download WAV</a></div></div>`;
  }).join("");
  // Autoplay only once per result.
  rt.results.forEach((r) => { if (r.url) r.played = true; });
  list.querySelectorAll("[data-copy]").forEach((b) => b.addEventListener("click", async () => {
    const r = rt.results.find((x) => x.id === b.dataset.copy);
    try {
      await navigator.clipboard.writeText(r.transcript.text);
      b.textContent = "Copied";
    } catch {
      const area = b.closest(".result").querySelector("textarea");
      area.select();
      b.textContent = "Press Ctrl+C to copy";
    }
    setTimeout(() => { b.textContent = "Copy text"; }, 1800);
  }));
  list.querySelectorAll("[data-remove]").forEach((b) => b.addEventListener("click", () => {
    const idx = rt.results.findIndex((r) => r.id === b.dataset.remove);
    const [r] = rt.results.splice(idx, 1);
    if (r?.url) URL.revokeObjectURL(r.url);
    renderResults(ws);
  }));
}

// ------------------------------------------------------------------ boot

async function boot() {
  load();
  const status = $("#status");
  try {
    app.info = await (await api("/api/info")).json();
    await loadVoices();
    app.prompts = await (await api("/api/prompts")).json().catch(() => []);
    const b = app.info.backend;
    status.textContent = `${b.backend}${b.gpu ? " · " + b.gpu : b.device ? " · " + b.device : ""} · ${app.voices.length} voices`;
    status.className = "status ok";
  } catch (e) {
    status.textContent = `API error: ${e.message}`;
    status.className = "status err";
  }
  app.workspaces.forEach(mountWorkspace);
  renderTabs();
  activate(app.activeId);
  $("#new-tab").addEventListener("click", addWorkspace);
  document.addEventListener("keydown", (e) => {
    if (e.altKey && e.key.toLowerCase() === "n") { e.preventDefault(); addWorkspace(); }
    if (e.altKey && e.key.toLowerCase() === "w") { e.preventDefault(); closeWorkspace(app.activeId); }
  });
}

boot();
