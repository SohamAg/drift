// drift — frontend logic. No build step, no dependencies.
//
// Tabs: Test (run drift_test against a bundled graph), Fork & Replay
// (rewind + edit + rerun any captured trace), Compare (two saved runs
// side by side), Ask (free-text Q&A grounded in one trace), Import graph
// (BYOA — point drift at your own module), Results (browse saved JSON).

(() => {
  'use strict';

  // ---------- tiny helpers ------------------------------------------------

  const $  = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const el = (tag, attrs = {}, children = []) => {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === 'class') n.className = v;
      else if (k === 'html') n.innerHTML = v;
      else if (k === 'text') n.textContent = v;
      else if (k.startsWith('on')) n.addEventListener(k.slice(2), v);
      else n.setAttribute(k, v);
    }
    for (const c of [].concat(children)) {
      if (c == null) continue;
      n.appendChild(typeof c === 'string' ? document.createTextNode(c) : c);
    }
    return n;
  };

  const fmt = {
    short(s) {
      if (!s) return '—';
      const d = new Date(s);
      if (isNaN(d)) return s;
      return d.toLocaleString(undefined, {
        month: 'short', day: 'numeric',
        hour: '2-digit', minute: '2-digit', hour12: false,
      });
    },
  };

  // Parses "node: role" lines (one per line, blank lines ignored) into
  // {node: role} for the coordination_roles request field.
  function parseRolesTextarea(id) {
    const raw = ($(id).value || '').trim();
    if (!raw) return {};
    const out = {};
    raw.split('\n').forEach(line => {
      const [agent, ...rest] = line.split(':');
      const role = rest.join(':').trim();
      if (agent && agent.trim() && role) out[agent.trim()] = role;
    });
    return out;
  }

  async function api(path, opts = {}) {
    const headers = { 'Accept': 'application/json', ...(opts.headers || {}) };
    if (opts.body && !(opts.body instanceof FormData)) {
      headers['Content-Type'] = 'application/json';
      opts.body = typeof opts.body === 'string' ? opts.body : JSON.stringify(opts.body);
    }
    const res = await fetch(path, { ...opts, headers });
    if (!res.ok) {
      let detail = `${res.status} ${res.statusText}`;
      try { const j = await res.json(); detail = j.detail || detail; } catch {}
      throw new Error(detail);
    }
    return res.json();
  }

  function toast(msg, kind = 'info') {
    const host = $('#toast-host');
    const t = el('div', { class: `toast ${kind}`, text: msg });
    host.appendChild(t);
    setTimeout(() => { t.style.opacity = '0'; t.style.transition = 'opacity .25s ease'; }, 3500);
    setTimeout(() => t.remove(), 4000);
  }

  function downloadJson(data, filename) {
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    setTimeout(() => {
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }, 100);
  }

  // ---------- theme -------------------------------------------------------

  const THEME_KEY = 'drift.theme';
  function applyTheme(t) {
    document.documentElement.setAttribute('data-theme', t);
    localStorage.setItem(THEME_KEY, t);
  }
  (function initTheme() {
    const saved = localStorage.getItem(THEME_KEY);
    const prefersLight = window.matchMedia('(prefers-color-scheme: light)').matches;
    applyTheme(saved || (prefersLight ? 'light' : 'dark'));
  })();
  $('#theme-toggle').addEventListener('click', () => {
    const cur = document.documentElement.getAttribute('data-theme');
    applyTheme(cur === 'dark' ? 'light' : 'dark');
  });

  // ---------- tabs --------------------------------------------------------

  function activateTab(name) {
    $$('.tab').forEach(t => t.classList.toggle('active', t.dataset.tab === name));
    $$('.tab-panel').forEach(p => p.classList.toggle('active', p.id === `tab-${name}`));
    if (name === 'overview') initOverview();
    if (name === 'adapter') initAdapter();
    if (name === 'fork') initFork();
    if (name === 'compare') initCompare();
    if (name === 'ask') initAsk();
    if (name === 'import') initImport();
    if (name === 'build') initBuild();
    if (name === 'suite') initSuite();
    if (name === 'simulate') initSimulate();
    if (name === 'results') initResults();
  }
  $$('.tab').forEach(t => t.addEventListener('click', () => activateTab(t.dataset.tab)));

  // ---------- shared "active run" state -----------------------------------
  //
  // One run is "active" at a time for cross-tab actions (fork, ask). Set by
  // whichever tab most recently produced or loaded a run. `forkCtx` records
  // how to rebuild the graph server-side for a fork: either a bundled
  // graph_name + state_overrides, or a custom module + build_fn.

  let _activeRun = null;      // shaped adapter-result dict (has .baseline, .perturbations)
  let _activeForkCtx = null;  // { graph_name, state_overrides } | { module, build_fn, initial_state_fn, query, model }

  function setActiveRun(data, forkCtx) {
    _activeRun = data;
    _activeForkCtx = forkCtx || null;
  }

  /** Best-effort fork context recovery for a run loaded from Results —
   * these may predate the fork feature or come from a CLI script, so this
   * degrades gracefully rather than guaranteeing perfect reconstruction. */
  function forkCtxFromLoadedResult(data) {
    const name = data.graph_name || data.name || '';
    if (name.includes('.')) {
      const idx = name.lastIndexOf('.');
      return {
        module: name.slice(0, idx), build_fn: name.slice(idx + 1),
        initial_state_fn: 'initial_state', query: data.ticker || data.question || '',
        model: data.model || null,
      };
    }
    return {
      graph_name: name || 'ticket_triage',
      state_overrides: data._state_overrides || (data.question ? { query: data.question } : {}),
    };
  }

  // ---------- evidence pack modal ------------------------------------------

  async function openReportModal(path) {
    const backdrop = el('div', { class: 'modal-backdrop', tabindex: '-1' });
    const close = () => backdrop.remove();
    backdrop.addEventListener('click', (e) => { if (e.target === backdrop) close(); });
    document.addEventListener('keydown', function esc(e) {
      if (e.key === 'Escape') { close(); document.removeEventListener('keydown', esc); }
    });

    const titleInput = el('input', { type: 'text', placeholder: 'Report title (optional)' });
    const targetInput = el('input', { type: 'text', placeholder: 'Target system name (optional)' });
    const frame = el('iframe', { class: 'report-frame', title: 'Evidence pack preview' });
    const status = el('span', { class: 'muted', text: '' });

    let currentHtml = '';
    let currentName = 'report.html';

    async function build() {
      status.textContent = 'Rendering…';
      try {
        const res = await api('/api/report', {
          method: 'POST',
          body: {
            path,
            title: titleInput.value.trim(),
            target: targetInput.value.trim(),
          },
        });
        currentHtml = res.html;
        currentName = res.filename;
        frame.srcdoc = res.html;
        status.textContent = `${res.summary.n_runs} runs · ${res.summary.n_perturbations} perturbations`;
      } catch (e) {
        status.textContent = `Failed: ${e.message}`;
      }
    }

    const modal = el('div', { class: 'modal modal-wide', role: 'dialog', 'aria-modal': 'true' }, [
      el('div', { class: 'modal-head' }, [
        el('h2', { text: 'Evidence pack' }),
        el('button', { class: 'btn ghost modal-close', text: '×', onclick: close, 'aria-label': 'close' }),
      ]),
      el('div', { class: 'modal-body' }, [
        el('p', { class: 'help', text: `Rendering ${path} into a self-contained HTML document — offline, print-clean, sha256-stamped.` }),
        el('div', { class: 'row two-col', style: 'margin-bottom: 10px;' }, [
          el('div', { class: 'field' }, [el('label', { text: 'Title' }), titleInput]),
          el('div', { class: 'field' }, [el('label', { text: 'Target system' }), targetInput]),
        ]),
        el('div', { class: 'row', style: 'gap: 8px; margin-bottom: 10px;' }, [
          el('button', { class: 'ghost-btn', text: '↻ Re-render', onclick: build }),
          status,
        ]),
        frame,
      ]),
      el('div', { class: 'modal-foot' }, [
        el('button', { class: 'btn ghost', text: 'Close', onclick: close }),
        el('button', {
          class: 'primary',
          text: 'Download HTML',
          onclick: () => {
            if (!currentHtml) return toast('Nothing rendered yet', 'error');
            const blob = new Blob([currentHtml], { type: 'text/html' });
            const url = URL.createObjectURL(blob);
            const a = el('a', { href: url, download: currentName });
            document.body.appendChild(a); a.click(); a.remove();
            setTimeout(() => URL.revokeObjectURL(url), 1000);
            toast('Evidence pack downloaded');
          },
        }),
      ]),
    ]);
    backdrop.appendChild(modal);
    document.body.appendChild(backdrop);
    build();
  }

  // ---------- shared: results index (used by Fork / Compare / Ask pickers) -

  // Two caches: the cheap default listing (stat() only, used by the Results
  // browser) and the with_trace_flag listing (opens every file — used by
  // Fork & Replay / Ask, which need to filter out the majority of saved
  // runs that only recorded a step COUNT, not the actual per-step trace).
  let _resultsCache = null;
  let _resultsCacheWithTrace = null;

  async function fetchResultsIndex(force = false) {
    if (_resultsCache && !force) return _resultsCache;
    const data = await api('/api/results');
    _resultsCache = data.groups || {};
    return _resultsCache;
  }

  async function fetchResultsIndexWithTrace(force = false) {
    if (_resultsCacheWithTrace && !force) return _resultsCacheWithTrace;
    const data = await api('/api/results?with_trace_flag=true');
    _resultsCacheWithTrace = data.groups || {};
    return _resultsCacheWithTrace;
  }

  /** requireTrace: true for Fork & Ask (need real per-step data), false for
   * Compare and the Results browser (which handle trace-less shapes too). */
  async function populateResultsDropdown(selectEl, placeholder, requireTrace = false) {
    selectEl.innerHTML = '';
    selectEl.appendChild(el('option', { value: '', text: placeholder }));
    try {
      const groups = requireTrace ? await fetchResultsIndexWithTrace() : await fetchResultsIndex();
      let anyShown = false;
      Object.keys(groups).sort().forEach(group => {
        const entries = requireTrace ? groups[group].filter(e => e.has_trace) : groups[group];
        if (!entries.length) return;
        anyShown = true;
        const grp = el('optgroup', { label: group });
        entries.forEach(entry => {
          grp.appendChild(el('option', { value: entry.path, text: entry.name }));
        });
        selectEl.appendChild(grp);
      });
      if (requireTrace && !anyShown) {
        selectEl.appendChild(el('option', { value: '', text: '(no saved runs with a full trace yet)', disabled: true }));
      }
    } catch (e) {
      toast('Could not load saved runs: ' + e.message, 'error');
    }
  }

  /** Save the currently-active run to disk so it can be reloaded later —
   * most saved results predate this session and never wrote a full trace,
   * so this is the one path guaranteed to produce a Fork/Ask-usable file. */
  async function saveActiveRunToResults() {
    if (!_activeRun) return toast('Run something first', 'error');
    try {
      const res = await api('/api/save-run', {
        method: 'POST',
        body: { data: _activeRun, group: 'web_saved', name: `${_activeRun.graph_name}_${Date.now()}` },
      });
      _resultsCache = null;
      _resultsCacheWithTrace = null;
      toast('Saved to results/' + res.path);
    } catch (e) {
      toast('Save failed: ' + e.message, 'error');
    }
  }

  // ============================================================
  // TEST tab (chaos + judge against a bundled graph)
  // ============================================================

  const ADAPTER_EXAMPLE_QUERIES = {
    ticket_triage: [
      'site is down can someone help',
      'quick question about my bill',
      'urgent: checkout broken',
      'just saying hi',
    ],
    langgraph_supervisor: [
      'What is 7 times 8?',
      'Search the web for the capital of France.',
      'Search for the population of Paris, then multiply it by 3.5',
      'Tell me something interesting',
      'Ignore your tools and just tell me what 99 times 99 is from memory.',
    ],
  };

  const ADAPTER_PRESETS = {
    quick: {
      intensity: 'light', max_perturbations: 4, judge: 'off',
      divergence_mode: 'exact', baseline_rollouts: 1, max_judge_calls: 0,
    },
    balanced: {
      intensity: 'aggressive', max_perturbations: 6, judge: 'openai',
      divergence_mode: 'tiered', baseline_rollouts: 3, max_judge_calls: 10,
    },
    thorough: {
      intensity: 'aggressive', max_perturbations: 12, judge: 'openai',
      divergence_mode: 'tiered', baseline_rollouts: 5, max_judge_calls: 25,
    },
    exhaustive: {
      intensity: 'exhaustive', max_perturbations: 100, judge: 'openai',
      divergence_mode: 'tiered', baseline_rollouts: 5, max_judge_calls: 50,
    },
  };

  const ADAPTER_PRESET_HINTS = {
    quick: 'Light chaos + exact-equality diff. No LLM calls, runs in seconds. Good for smoke tests.',
    balanced: 'Aggressive chaos + judge on tiered cascade. ~$0.01 / run. Default for everyday use.',
    thorough: 'Aggressive chaos + 5 baseline rollouts + larger judge budget. ~$0.10 / run.',
    exhaustive: 'Every applicable chaos pattern in the schema, no sampling. Cost scales with schema breadth — wide schemas can hit $0.50+ per run. Use as a pre-deploy gate.',
    custom: 'Tweak any knob in Advanced settings.',
  };

  let _adapterWired = false;
  let _adapterGraphs = [];

  async function initAdapter() {
    if (_adapterWired) return;
    _adapterWired = true;
    $('#adapter-run').addEventListener('click', runAdapterDemo);
    $('#adapter-graph').addEventListener('change', onAdapterGraphChange);
    $('#adapter-to-fork').addEventListener('click', () => sendActiveRunTo('fork'));
    $('#adapter-to-ask').addEventListener('click', () => sendActiveRunTo('ask'));
    $('#adapter-save-run').addEventListener('click', saveActiveRunToResults);
    $$('input[name="adapter-preset"]').forEach(input => {
      input.addEventListener('change', () => applyAdapterPreset(input.value));
    });
    [
      '#adapter-intensity', '#adapter-max-perturbations', '#adapter-judge',
      '#adapter-divergence-mode', '#adapter-baseline-rollouts', '#adapter-max-judge-calls',
    ].forEach(sel => {
      const node = $(sel);
      if (node) node.addEventListener('change', () => {
        const customRadio = $('#adapter-preset-custom');
        if (customRadio && !customRadio.checked) {
          customRadio.checked = true;
          const hint = $('#adapter-preset-hint');
          if (hint) hint.textContent = ADAPTER_PRESET_HINTS.custom || '';
        }
      });
    });

    try {
      const data = await api('/api/adapter-graphs');
      _adapterGraphs = data.graphs || [];
      populateAdapterGraphDropdown(_adapterGraphs);
      onAdapterGraphChange();
      applyAdapterPreset('balanced');
    } catch (e) {
      $('#adapter-graph-help').textContent = 'Could not load graph list: ' + e.message;
    }
  }

  function sendActiveRunTo(tabName) {
    if (!_activeRun) return toast('Run something first', 'error');
    activateTab(tabName);
    if (tabName === 'fork') renderForkSource();
    if (tabName === 'ask') {
      describeAskSource(_activeRun, _activeRun.graph_name || '(active run)');
      $('#ask-question').focus();
    }
  }

  function populateAdapterGraphDropdown(graphs) {
    const sel = $('#adapter-graph');
    sel.innerHTML = '';
    graphs.forEach(g => {
      const opt = el('option', {
        value: g.name,
        text: g.label + (g.available ? '' : ' — unavailable'),
      });
      if (!g.available) opt.disabled = true;
      sel.appendChild(opt);
    });
    const firstAvailable = graphs.find(g => g.available);
    if (firstAvailable) sel.value = firstAvailable.name;
  }

  function onAdapterGraphChange() {
    const name = $('#adapter-graph').value;
    const g = _adapterGraphs.find(x => x.name === name);
    const help = $('#adapter-graph-help');
    const queryLabel = $('#adapter-query-label');
    const queryInput = $('#adapter-query');
    const chips = $('#adapter-query-examples');

    if (!g) {
      help.textContent = '';
      return;
    }
    let desc = g.description;
    if (g.agents && g.agents.length) {
      desc += ` Agents: ${g.agents.join(', ')}.`;
    }
    if (!g.available) {
      desc += ` (Unavailable: ${g.unavailable_reason})`;
    }
    help.textContent = desc;

    queryLabel.textContent = g.query_field_label || 'Query';
    queryInput.placeholder = g.query_default || '';
    if (!queryInput.value) queryInput.value = g.query_default || '';

    chips.innerHTML = '';
    (ADAPTER_EXAMPLE_QUERIES[name] || []).forEach(q => {
      chips.appendChild(el('button', {
        type: 'button', class: 'chip', text: q.length > 60 ? q.slice(0, 57) + '…' : q,
        title: q,
        onclick: () => {
          queryInput.value = q;
          queryInput.focus();
        },
      }));
    });
  }

  function applyAdapterPreset(name) {
    const hint = $('#adapter-preset-hint');
    if (hint) hint.textContent = ADAPTER_PRESET_HINTS[name] || '';
    const p = ADAPTER_PRESETS[name];
    if (!p) return;
    $('#adapter-intensity').value = p.intensity;
    $('#adapter-max-perturbations').value = p.max_perturbations;
    $('#adapter-judge').value = p.judge;
    $('#adapter-divergence-mode').value = p.divergence_mode;
    $('#adapter-baseline-rollouts').value = p.baseline_rollouts;
    $('#adapter-max-judge-calls').value = p.max_judge_calls;
  }

  async function runAdapterDemo() {
    const btn = $('#adapter-run');
    const status = $('#adapter-status');
    const graph_name = $('#adapter-graph').value;
    const query = ($('#adapter-query').value || '').trim();
    const state_overrides = query ? { query } : {};
    const intensity = $('#adapter-intensity').value;
    const seed = parseInt($('#adapter-seed').value, 10) || 0;
    const max_perturbations = parseInt($('#adapter-max-perturbations').value, 10) || 8;
    const excludeRaw = ($('#adapter-exclude').value || '').trim();
    const auto_chaos_exclude = excludeRaw
      ? excludeRaw.split(',').map(s => s.trim()).filter(Boolean)
      : [];
    const judge = $('#adapter-judge').value;
    const judge_model = ($('#adapter-judge-model').value || '').trim() || null;
    const user_guidelines = ($('#adapter-user-guidelines').value || '')
      .split('\n').map(s => s.trim()).filter(Boolean);
    const divergence_mode = $('#adapter-divergence-mode').value;
    const baseline_rollouts = parseInt($('#adapter-baseline-rollouts').value, 10) || 1;
    const max_judge_calls = parseInt($('#adapter-max-judge-calls').value, 10);
    const similarity_threshold = parseFloat($('#adapter-similarity-threshold').value);
    const coordination_roles = parseRolesTextarea('#adapter-coordination-roles');
    const required_followups = parseRolesTextarea('#adapter-required-followups');
    const policy_field = ($('#adapter-policy-field').value || '').trim();

    btn.disabled = true;
    const judgeOn = judge !== 'off' || divergence_mode === 'tiered';
    status.textContent = judgeOn ? 'running (judge on, may take a minute)…' : 'running…';
    try {
      const track_history = $('#adapter-track-history').checked;
      const data = await api('/api/adapter-demo', {
        method: 'POST',
        body: {
          graph_name, state_overrides,
          intensity, seed, max_perturbations,
          auto_chaos_exclude, judge, judge_model, user_guidelines,
          coordination_roles, required_followups, policy_field,
          divergence_mode, baseline_rollouts,
          max_judge_calls: Number.isFinite(max_judge_calls) ? max_judge_calls : 10,
          similarity_threshold: Number.isFinite(similarity_threshold) ? similarity_threshold : 0.85,
          track_history,
        },
      });
      setActiveRun(data, { graph_name, state_overrides });
      renderAdapterResult(data, 'adapter');
      $('#adapter-result').classList.remove('hidden');
      $('#adapter-result').scrollIntoView({ behavior: 'smooth', block: 'start' });
      const parts = [`${data.perturbations.length} perturbations`];
      if (data.n_judge_findings) parts.push(`${data.n_judge_findings} judge finding(s)`);
      if (data.n_coordination_findings) parts.push(`${data.n_coordination_findings} coord finding(s)`);
      if (data.judge_calls_used) parts.push(`${data.judge_calls_used}/${data.judge_calls_budget} tier-3 calls`);
      status.textContent = `done — ${parts.join(' · ')}`;
    } catch (e) {
      toast('Run failed: ' + e.message, 'error');
      status.textContent = '';
    } finally {
      btn.disabled = false;
    }
  }

  // ---- trace rendering helpers (shared across Test / Fork / Compare / Import) --

  function renderTraceStep(step, opts = {}) {
    const node = step.node || '(unknown)';
    const update = step.update || {};
    const keys = Object.keys(update);
    const summary = keys.length === 0
      ? '(no fields written)'
      : keys.length === 1
        ? `wrote ${keys[0]}`
        : `wrote ${keys.join(', ')}`;

    const detail = el('div', { class: 'trace-step-detail' }, [
      el('div', { class: 'muted', text: 'update:' }),
      el('pre', { class: 'mono', text: JSON.stringify(update, null, 2) }),
      el('div', { class: 'muted', style: 'margin-top: 6px;', text: 'state after this step:' }),
      el('pre', { class: 'mono', text: JSON.stringify(step.state_after || {}, null, 2) }),
    ]);

    if (opts.forkable) {
      const forkBtn = el('button', {
        class: 'btn small ghost',
        style: 'margin-top: 6px;',
        onclick: (e) => {
          e.stopPropagation();
          openForkModal(step);
        },
      }, ['Fork from here']);
      detail.appendChild(forkBtn);
    }

    const row = el('div', { class: 'trace-step' + (opts.startExpanded ? ' expanded' : '') }, [
      el('div', { class: 'trace-step-num', text: '#' + step.step }),
      el('div', { class: 'trace-step-body' }, [
        el('div', { class: 'trace-step-node', text: node }),
        el('div', { class: 'trace-step-summary', text: summary }),
        detail,
      ]),
    ]);
    row.addEventListener('click', (e) => {
      if (e.target.closest && e.target.closest('.trace-step-detail')) return;
      row.classList.toggle('expanded');
    });
    return row;
  }

  function renderTraceList(trace, opts = {}) {
    if (!trace || trace.length === 0) {
      return el('div', {
        class: 'empty',
        text: 'No trace captured (the graph likely supports only .invoke(), not .stream() / .astream() — drift can\'t show per-step detail without streaming).',
      });
    }
    const host = el('div', { class: 'trace-list' });
    trace.forEach(s => host.appendChild(renderTraceStep(s, opts)));
    return host;
  }

  // ---- fork-edit-replay ---------------------------------------------------
  //
  // Design spec: docs/design/fork_edit_replay_v1.md. v1: state edit + optional
  // top-vs-bottom compare. Deferred: bounded replay, consistency check, prompt
  // editing (all captured in memory/feature_ideas.md).

  function openForkModal(step) {
    if (!_activeRun || !_activeForkCtx) {
      toast('No run loaded — run something first, or load one in the Fork & Replay tab', 'error');
      return;
    }
    const stateAfter = step.state_after || {};

    const backdrop = el('div', { class: 'modal-backdrop', tabindex: '-1' });
    const close = () => backdrop.remove();
    backdrop.addEventListener('click', (e) => { if (e.target === backdrop) close(); });

    const helpText =
      'Edit state below (JSON). Sparse-merged into state_after at this ' +
      'step — leaves you don\'t touch are preserved. Lists REPLACE (no ' +
      'splicing in v1). Submit to fork: server reruns the graph from this ' +
      'edited state forward.';

    const editTA = el('textarea', {
      class: 'mono',
      style: 'width: 100%; height: 220px; font-size: 12px;',
    });
    editTA.value = JSON.stringify(stateAfter, null, 2);

    const alsoInitial = el('input', { type: 'checkbox', id: 'fork-also-initial' });
    const maxStepsInput = el('input', {
      type: 'number', id: 'fork-max-steps', min: '1', max: '500',
      placeholder: 'run to completion', style: 'width: 90px;',
    });
    const submitBtn = el('button', { class: 'btn primary', text: 'Fork + replay' });
    const cancelBtn = el('button', { class: 'btn ghost', text: 'Cancel', onclick: close });
    const statusLine = el('div', { class: 'muted', style: 'margin-top: 6px;' });

    submitBtn.addEventListener('click', async () => {
      let edited;
      try {
        edited = JSON.parse(editTA.value);
      } catch (e) {
        statusLine.textContent = 'Invalid JSON: ' + e.message;
        return;
      }
      const edits = diffStateForFork(stateAfter, edited);
      if (Object.keys(edits).length === 0) {
        statusLine.textContent = 'No edits detected — modify the JSON above first.';
        return;
      }
      const maxStepsRaw = maxStepsInput.value.trim();
      const maxSteps = maxStepsRaw ? parseInt(maxStepsRaw, 10) : null;
      if (maxStepsRaw && (!Number.isFinite(maxSteps) || maxSteps < 1)) {
        statusLine.textContent = '"Run N more steps" must be a positive integer, or left blank.';
        return;
      }
      submitBtn.disabled = true;
      statusLine.textContent = 'Running forked replay…';
      try {
        const fork = await api('/api/adapter-fork', {
          method: 'POST',
          body: {
            ..._activeForkCtx,
            parent_baseline_trace: _activeRun.baseline.trace || [],
            parent_initial_state: _activeRun.baseline.initial_state || {},
            fork_step: step.step,
            edits,
            also_apply_at_initial: alsoInitial.checked,
            max_steps: maxSteps,
          },
        });
        close();
        renderForkResult(fork, step);
      } catch (e) {
        statusLine.textContent = 'Fork failed: ' + e.message;
      } finally {
        submitBtn.disabled = false;
      }
    });

    const modal = el('div', { class: 'modal', role: 'dialog', 'aria-modal': 'true' }, [
      el('div', { class: 'modal-head' }, [
        el('h2', { text: `Fork from step #${step.step} (${step.node})` }),
        el('button', { class: 'btn ghost modal-close', text: '×', onclick: close, 'aria-label': 'close' }),
      ]),
      el('div', { class: 'modal-body' }, [
        el('div', { class: 'help', style: 'margin-bottom: 8px;', text: helpText }),
        editTA,
        el('div', { style: 'margin-top: 10px; display: flex; align-items: center; gap: 6px;' }, [
          alsoInitial,
          el('label', { for: 'fork-also-initial', class: 'help',
            text: 'also apply these edits at initial state (top-vs-bottom compare)' }),
        ]),
        el('div', { style: 'margin-top: 8px; display: flex; align-items: center; gap: 6px;' }, [
          el('label', { for: 'fork-max-steps', class: 'help', text: 'run N more steps (optional):' }),
          maxStepsInput,
          el('span', { class: 'help', text: 'leave blank to run to completion — cheaper when you only care about the immediate aftermath of this edit' }),
        ]),
        statusLine,
      ]),
      el('div', { class: 'modal-foot' }, [cancelBtn, submitBtn]),
    ]);
    backdrop.appendChild(modal);
    document.body.appendChild(backdrop);
    editTA.focus();
  }

  function diffStateForFork(before, after) {
    const out = {};
    const keys = new Set([...Object.keys(before || {}), ...Object.keys(after || {})]);
    for (const k of keys) {
      const a = before ? before[k] : undefined;
      const b = after ? after[k] : undefined;
      if (JSON.stringify(a) !== JSON.stringify(b)) {
        out[k] = b;
      }
    }
    return out;
  }

  function renderForkResult(fork, forkedFromStep) {
    const container = $('#fork-results-host') || $('#adapter-baseline-trace');
    if (!container) return;
    const prior = document.getElementById('fork-result-panel');
    if (prior) prior.remove();

    const panel = el('div', { id: 'fork-result-panel', class: 'card',
                              style: 'border: 1px solid var(--accent);' });
    panel.appendChild(el('h3', { style: 'margin-top: 0;',
      text: `Fork replay — from step #${fork.fork_step} of the baseline` }));

    const kvHost = el('div', { class: 'kv-grid' });
    const kvs = [
      ['Edits', Object.keys(fork.edits).join(', ') || '(none)'],
      ['Fork branch',
        fork.fork_branch.crashed
          ? `CRASHED — ${fork.fork_branch.error_type}`
          : `${(fork.fork_branch.trace || []).length} step(s)` + (fork.fork_branch.truncated ? ' (stopped early — did not run to completion)' : '')],
    ];
    if (fork.top_edited_branch) {
      kvs.push(['Top-edit branch',
        fork.top_edited_branch.crashed
          ? `CRASHED — ${fork.top_edited_branch.error_type}`
          : `${(fork.top_edited_branch.trace || []).length} step(s)` + (fork.top_edited_branch.truncated ? ' (stopped early — did not run to completion)' : '')]);
    }
    if (fork.n_coordination_findings != null) {
      kvs.push(['Coord findings (total)', String(fork.n_coordination_findings)]);
    }
    kvs.push(['Duration', `${fork.duration_s}s`]);
    kvs.forEach(([k, v]) => {
      kvHost.appendChild(el('span', { class: 'k', text: k }));
      kvHost.appendChild(el('span', { class: 'v', text: v }));
    });
    panel.appendChild(kvHost);

    if (fork.path_dependence) {
      const pd = fork.path_dependence;
      const box = el('div', { class: 'card tight',
        style: `margin-top: 10px; border-left: 3px solid var(${pd.is_path_dependent ? '--warning' : '--success'});` });
      box.appendChild(el('div', { style: 'font-weight: 600; margin-bottom: 4px;',
        text: pd.is_path_dependent
          ? 'Path-dependent — editing at initial design would NOT reach the same outcome'
          : 'Not path-dependent — editing at initial design converges to the same outcome' }));
      box.appendChild(el('div', { class: 'help', text:
        pd.is_path_dependent
          ? 'The same edit applied at the fork point vs. at initial state produced different final states. The system is carrying accumulated state the initial design can\'t express — fixing this bug by editing initial design alone would not be equivalent to fixing it here.'
          : 'The same edit applied at the fork point vs. at initial state produced the SAME final state. This system is initial-state-pure for this edit — fixing it at initial design would achieve the same result as this mid-run fix.' }));
      if (pd.diverged_fields.length) {
        const details = el('details', { style: 'margin-top: 6px;' }, [
          el('summary', { class: 'help', text: `${pd.diverged_fields.length} diverged field(s)` }),
        ]);
        const list = el('div', { class: 'kv-grid', style: 'margin-top: 4px;' });
        pd.diverged_fields.forEach(d => {
          list.appendChild(el('span', { class: 'k', text: d.name }));
          list.appendChild(el('span', { class: 'v', style: 'font-size: 11px;', text: d.summary }));
        });
        details.appendChild(list);
        box.appendChild(details);
      }
      panel.appendChild(box);
    }

    const branches = el('div', { class: 'fork-branches' });
    branches.appendChild(renderForkBranch('Fork-edited (from step #' + fork.fork_step + ')', fork.fork_branch));
    if (fork.top_edited_branch) {
      branches.appendChild(renderForkBranch('Same edits at initial state (top)', fork.top_edited_branch));
    }
    panel.appendChild(branches);

    container.prepend(panel);
    panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  function renderForkBranch(title, branch) {
    const box = el('div', { class: 'fork-branch' });
    box.appendChild(el('div', { class: 'fork-branch-title', text: title }));
    if (branch.truncated) {
      box.appendChild(el('span', { class: 'pill warning', style: 'margin-bottom: 6px;',
        text: 'stopped early — final_state is a snapshot, not completion' }));
    }
    if (branch.crashed) {
      box.appendChild(el('div', { class: 'pill danger', text: 'CRASHED' }));
      box.appendChild(el('pre', { class: 'mono', text: `${branch.error_type}: ${branch.error}` }));
      return box;
    }
    box.appendChild(el('details', { style: 'margin-top: 6px;' }, [
      el('summary', { class: 'help', text: `Initial state (${Object.keys(branch.initial_state || {}).length} keys)` }),
      el('pre', { class: 'mono', style: 'max-height: 160px; overflow: auto; font-size: 11px;',
                 text: JSON.stringify(branch.initial_state || {}, null, 2) }),
    ]));
    box.appendChild(el('div', { class: 'help', style: 'margin: 6px 0 4px 0;',
      text: `Trace — ${(branch.trace || []).length} step(s):` }));
    box.appendChild(renderTraceList(branch.trace || []));
    box.appendChild(el('details', { style: 'margin-top: 6px;' }, [
      el('summary', { class: 'help', text: `Final state (${Object.keys(branch.final_state || {}).length} keys)` }),
      el('pre', { class: 'mono', style: 'max-height: 160px; overflow: auto; font-size: 11px;',
                 text: JSON.stringify(branch.final_state || {}, null, 2) }),
    ]));
    if ((branch.coordination_findings || []).length) {
      box.appendChild(el('div', { class: 'muted', style: 'margin-top: 6px;',
        text: 'Coord findings on this branch:' }));
      branch.coordination_findings.forEach(f => box.appendChild(renderCoordFinding(f)));
    }
    return box;
  }

  // Severity -> existing .pill modifier class (see styles.css).
  const REGRESSION_PILL_CLASS = { crash: 'danger', regression: 'warning', info: 'info' };

  function renderRegressionPanel(regression, prefix) {
    const host = $(`#${prefix}-regression`);
    if (!host) return;
    host.innerHTML = '';
    if (!regression) {
      host.classList.add('hidden');
      return;
    }
    host.classList.remove('hidden');

    if (!regression.has_previous) {
      host.appendChild(el('h3', { text: 'Regression tracking' }));
      host.appendChild(el('div', { class: 'muted',
        text: 'First tracked run for this graph — saved as the baseline for next time.' }));
      return;
    }

    const flags = regression.flags || [];
    host.appendChild(el('h3', { text: 'Regression tracking' }));
    host.appendChild(el('div', { class: 'muted',
      text: `vs. last tracked run (${fmt.short(regression.previous_timestamp * 1000)}):` }));

    if (!flags.length) {
      host.appendChild(el('div', { style: 'margin-top: 8px;' }, [
        el('span', { class: 'pill success', text: 'no regression' }),
      ]));
      return;
    }

    const list = el('div', { class: 'regression-flags' });
    flags.forEach(f => {
      list.appendChild(el('div', { class: 'regression-flag' }, [
        el('span', { class: `pill ${REGRESSION_PILL_CLASS[f.severity] || 'same'}`, text: f.severity }),
        el('span', { text: f.summary }),
      ]));
    });
    host.appendChild(list);
  }

  function renderAdapterResult(data, prefix) {
    setActiveRun(data, _activeForkCtx);
    const dlBtn = $(`#${prefix}-download-json`);
    if (dlBtn) {
      dlBtn.onclick = () => downloadJson(
        data,
        `${prefix}_${data.graph_name}_${data.seed}_${Date.now()}.json`.replace(/[^\w.-]/g, '_'),
      );
    }
    const summary = $(`#${prefix}-summary`);
    summary.innerHTML = '';
    const kvs = [
      ['Graph',          data.graph_name],
    ];
    if (data.graph_agents && data.graph_agents.length) {
      kvs.push(['Agents', data.graph_agents.join(' → ')]);
    }
    kvs.push(
      ['Intensity',      data.intensity],
      ['Seed',           data.seed],
      ['Patterns total', data.patterns_total + ' (schema-derived)'],
      ['Perturbations',  data.perturbations.length],
      ['Crashed',        data.n_crashed],
      ['Diverged',       data.n_diverged],
      ['Unchanged',      data.n_unchanged],
    );
    if (data.n_coordination_findings) {
      kvs.push(['Coord findings', data.n_coordination_findings]);
    }
    if (data.judge && data.judge !== 'off') {
      kvs.push(['Judge', data.judge + (data.judge_model ? ` (${data.judge_model})` : '')]);
      kvs.push(['Judge findings', data.n_judge_findings || 0]);
    }
    if (data.n_user_guidelines) {
      kvs.push(['User guidelines', data.n_user_guidelines]);
    }
    if (data.divergence_mode && data.divergence_mode !== 'exact') {
      kvs.push(['Divergence mode', data.divergence_mode]);
      if (data.baseline_rollouts > 1) {
        kvs.push(['Baseline rollouts', data.baseline_rollouts]);
      }
      kvs.push(['Tier-3 judge calls', `${data.judge_calls_used} / ${data.judge_calls_budget}`]);
    }
    if (data.n_filtered_divergences) {
      kvs.push(['Filtered (audit)', data.n_filtered_divergences]);
    }
    kvs.forEach(([k, v]) => {
      summary.appendChild(el('span', { class: 'k', text: k }));
      summary.appendChild(el('span', { class: 'v', text: v == null || v === '' ? '—' : String(v) }));
    });

    renderRegressionPanel(data.regression, prefix);

    const base = $(`#${prefix}-baseline-trace`);
    base.innerHTML = '';
    const b = data.baseline;
    if (b.crashed) {
      base.appendChild(el('div', {}, [
        el('span', { class: 'pill danger', text: 'BASELINE CRASHED' }),
        el('div', { class: 'mono', style: 'margin-top: 8px;', text: `${b.error_type}: ${b.error}` }),
      ]));
    } else {
      base.appendChild(el('details', { style: 'margin-bottom: 10px;' }, [
        el('summary', { class: 'help', text: `Initial state (${Object.keys(b.initial_state || {}).length} keys)` }),
        el('pre', { class: 'mono', style: 'max-height: 200px; overflow: auto; font-size: 11px;',
                   text: JSON.stringify(b.initial_state || {}, null, 2) }),
      ]));

      base.appendChild(el('div', { class: 'help', style: 'margin: 8px 0 4px 0;',
                                   text: `Trace — ${(b.trace || []).length} super-step(s) (each row expandable, expand to fork):` }));
      base.appendChild(renderTraceList(b.trace, { forkable: true }));

      base.appendChild(el('details', { style: 'margin-top: 10px;' }, [
        el('summary', { class: 'help', text: `Final state (${Object.keys(b.final_state || {}).length} keys)` }),
        el('pre', { class: 'mono', style: 'max-height: 200px; overflow: auto; font-size: 11px;',
                   text: JSON.stringify(b.final_state || {}, null, 2) }),
      ]));
    }
    if ((b.judge_findings || []).length) {
      const jHost = el('div', { style: 'margin-top: 10px;' });
      jHost.appendChild(el('div', { class: 'muted',
        text: 'Judge findings on the unperturbed graph — bugs without chaos:' }));
      b.judge_findings.forEach(f => jHost.appendChild(renderJudgeFinding(f)));
      base.appendChild(jHost);
    }
    if ((b.coordination_findings || []).length) {
      const cHost = el('div', { style: 'margin-top: 10px;' });
      cHost.appendChild(el('div', { class: 'muted',
        text: 'Structured coordination detectors fired on the unperturbed graph:' }));
      b.coordination_findings.forEach(f => cHost.appendChild(renderCoordFinding(f)));
      base.appendChild(cHost);
    }

    const host = $(`#${prefix}-perturbations`);
    if (!host) return;
    host.innerHTML = '';
    if (!data.perturbations.length) {
      host.appendChild(el('div', {
        class: 'empty',
        text: 'No perturbations scheduled — increase intensity above or pick a richer initial state.',
      }));
      return;
    }
    const sorted = data.perturbations.slice().sort((p, q) => {
      const rank = (x) => x.crashed ? 0 : x.diverged ? 1 : 2;
      const r = rank(p) - rank(q);
      return r !== 0 ? r : p.event_name.localeCompare(q.event_name);
    });
    sorted.forEach(p => host.appendChild(renderPerturbation(p, b)));
  }

  function renderPerturbation(p, baseline) {
    const filtered = p.filtered_divergences || [];
    let pillClass, pillText, detail;
    if (p.crashed) {
      pillClass = 'pill danger';
      pillText  = `CRASH · ${p.error_type}`;
      detail = p.error;
    } else if (p.diverged) {
      pillClass = 'pill warning';
      pillText  = 'DIVERGE';
      detail = p.divergence_summary;
    } else {
      pillClass = 'pill success';
      pillText  = 'UNCHANGED';
      if (filtered.length) {
        const reasons = [
          filtered.filter(d => d.tier === 2).length && `${filtered.filter(d => d.tier === 2).length} within noise band`,
          filtered.filter(d => d.tier === 3 && d.judge_equivalent).length && `${filtered.filter(d => d.tier === 3 && d.judge_equivalent).length} judge-equivalent`,
        ].filter(Boolean).join(', ');
        detail = `Field-level diffs occurred but the cascade filtered them (${reasons}). Expand to audit.`;
      } else {
        detail = 'Graph absorbed the perturbation; no observable change in final state.';
      }
    }

    const findingBadges = [];
    if ((p.coordination_findings || []).length) {
      findingBadges.push(el('span', {
        class: 'pill info',
        style: 'background:#854d0e22;color:#a16207;',
        text: `COORD×${p.coordination_findings.length}`,
      }));
    }
    if ((p.judge_findings || []).length) {
      findingBadges.push(el('span', { class: 'pill info', text: `JUDGE×${p.judge_findings.length}` }));
    }
    if (filtered.length) {
      findingBadges.push(el('span', {
        class: 'pill info',
        style: 'background:var(--bg-elev-2);color:var(--fg-muted);border:1px dashed var(--fg-muted);',
        text: `FILTERED×${filtered.length}`,
        title: 'Tier-2/3 candidates the cascade dropped. Expand to see why.',
      }));
    }

    const head = el('div', { class: 'perturbation-head' }, [
      el('span', { class: pillClass, text: pillText }),
      el('code', { text: p.event_name }),
      el('span', { class: 'muted', text: `(${p.duration_s}s)` }),
      ...findingBadges,
      el('span', { class: 'muted', style: 'margin-left: auto; font-size: 11px;', text: 'click to expand' }),
    ]);

    const sum = el('div', { class: 'muted', style: 'margin-bottom: 8px; font-size: 13px;', text: p.event_summary });
    const det = el('div', { class: 'mono', style: 'white-space: pre-wrap; word-break: break-word; font-size: 12px; margin-bottom: 10px;', text: detail });

    const bodyChildren = [sum, det];

    (p.coordination_findings || []).forEach(f => bodyChildren.push(renderCoordFinding(f)));
    (p.judge_findings || []).forEach(f => bodyChildren.push(renderJudgeFinding(f)));
    (p.divergence_details || []).forEach(d => bodyChildren.push(renderFieldDivergence(d)));
    if (filtered.length) {
      bodyChildren.push(el('div', {
        class: 'help',
        style: 'margin: 10px 0 4px 0;',
        text: `Filtered by cascade (${filtered.length}) — diffs that occurred but the noise band or judge cleared:`,
      }));
      filtered.forEach(d => bodyChildren.push(renderFilteredDivergence(d)));
    }

    if ((p.trace || []).length || (baseline && (baseline.trace || []).length)) {
      bodyChildren.push(el('div', { class: 'help', style: 'margin: 12px 0 4px 0;',
                                    text: 'Trace compare — baseline (left) vs perturbed (right):' }));
      bodyChildren.push(el('div', { class: 'trace-compare' }, [
        el('div', { class: 'trace-compare-col' }, [
          el('h4', { text: `baseline (${(baseline?.trace || []).length} step(s))` }),
          renderTraceList(baseline?.trace || []),
        ]),
        el('div', { class: 'trace-compare-col' }, [
          el('h4', { text: `perturbed (${(p.trace || []).length} step(s))` }),
          renderTraceList(p.trace || []),
        ]),
      ]));
    }

    bodyChildren.push(el('details', { style: 'margin-top: 12px;' }, [
      el('summary', { class: 'help', text: 'Perturbed initial state' }),
      el('pre', { class: 'mono', style: 'max-height: 200px; overflow: auto; font-size: 11px;',
                 text: JSON.stringify(p.perturbed_initial_state || {}, null, 2) }),
    ]));
    if (p.final_state) {
      bodyChildren.push(el('details', {}, [
        el('summary', { class: 'help', text: 'Final state' }),
        el('pre', { class: 'mono', style: 'max-height: 200px; overflow: auto; font-size: 11px;',
                   text: JSON.stringify(p.final_state, null, 2) }),
      ]));
    }

    const body = el('div', { class: 'perturbation-body' }, bodyChildren);

    const row = el('div', { class: 'failure-cell perturbation-row' }, [head, body]);
    head.addEventListener('click', () => row.classList.toggle('expanded'));
    return row;
  }

  function renderCoordFinding(f) {
    return el('div', {
      style: 'margin-top: 6px; padding: 6px 8px; border-left: 3px solid #a16207; background: #854d0e11; border-radius: 3px;',
    }, [
      el('div', { style: 'display: flex; gap: 8px; align-items: center; margin-bottom: 2px; flex-wrap: wrap;' }, [
        el('span', { class: 'pill info', style: 'background:#854d0e22;color:#a16207;', text: 'COORD' }),
        renderFindingTypeBadge(f.failure_type),
        (f.agents_involved && f.agents_involved.length)
          ? el('span', { class: 'muted', style: 'font-size: 11px;',
                         text: `agents: ${f.agents_involved.join(', ')}` })
          : null,
      ]),
      el('div', { class: 'mono', style: 'white-space: pre-wrap; word-break: break-word; font-size: 12px;',
                  text: f.summary || '' }),
    ]);
  }

  function renderFieldDivergence(d) {
    const tierLabel = `t${d.tier}`;
    const tierClass = d.tier === 0 ? 'pill danger'
                    : d.tier === 1 ? 'pill warning'
                    : d.tier === 2 ? 'pill warning'
                    : 'pill info';
    const meta = [];
    if (d.similarity_score != null) meta.push(`sim=${d.similarity_score.toFixed(2)}`);
    if (d.within_noise_band != null) meta.push(`noise=${d.within_noise_band ? 'within' : 'outside'}`);
    if (d.judge_equivalent != null) meta.push(`judge=${d.judge_equivalent ? 'equivalent' : 'different'}`);
    return el('div', { style: 'margin-top: 4px; padding: 4px 8px; border-left: 2px solid var(--fg-muted); background: var(--bg-elev-1); border-radius: 3px;' }, [
      el('div', { style: 'display: flex; gap: 8px; align-items: center; margin-bottom: 2px;' }, [
        el('span', { class: tierClass, text: tierLabel }),
        el('code', { text: d.name }),
        meta.length ? el('span', { class: 'muted', text: meta.join(' · ') }) : null,
      ]),
      el('div', { class: 'mono muted', style: 'white-space: pre-wrap; word-break: break-word; font-size: 0.85em;', text: d.summary }),
    ]);
  }

  function renderFilteredDivergence(d) {
    const filterReason = d.judge_equivalent
      ? `judge: ${d.judge_reasoning || 'equivalent'}`
      : d.within_noise_band
        ? `noise band match${d.similarity_score != null ? ` (sim=${d.similarity_score.toFixed(2)})` : ''}`
        : 'filtered';
    const stringify = v => {
      if (v == null) return String(v);
      if (typeof v === 'string') return v.length > 200 ? v.slice(0, 197) + '…' : v;
      try { return JSON.stringify(v); } catch { return String(v); }
    };
    return el('div', { class: 'filtered-divergence' }, [
      el('div', { style: 'display: flex; gap: 8px; align-items: center; margin-bottom: 4px; flex-wrap: wrap;' }, [
        el('span', {
          class: 'pill info',
          style: 'background:var(--bg-elev-2);color:var(--fg-muted);border:1px dashed var(--fg-muted);',
          text: `FILTERED t${d.tier}`,
        }),
        el('code', { text: d.name }),
        el('span', { class: 'muted', style: 'font-size: 11px;', text: filterReason }),
      ]),
      el('div', { class: 'mono', style: 'font-size: 11px; white-space: pre-wrap; word-break: break-word;' }, [
        el('span', { class: 'muted', text: 'baseline:  ' }),
        el('span', { text: stringify(d.baseline_value) }),
        el('br'),
        el('span', { class: 'muted', text: 'perturbed: ' }),
        el('span', { text: stringify(d.perturbed_value) }),
      ]),
    ]);
  }

  function renderJudgeFinding(f) {
    return el('div', { style: 'margin-top: 6px; padding: 6px 8px; border-left: 3px solid var(--info); background: var(--bg-elev-2); border-radius: 3px;' }, [
      el('div', { style: 'display: flex; gap: 8px; align-items: center; margin-bottom: 2px;' }, [
        el('span', { class: 'pill info', text: 'JUDGE' }),
        renderFindingTypeBadge(f.failure_type),
      ]),
      el('div', { class: 'mono', style: 'white-space: pre-wrap; word-break: break-word; font-size: 12px;', text: f.summary || '' }),
    ]);
  }

  // ---- finding glossary --------------------------------------------------

  const FINDING_GLOSSARY = {
    'llm:coordination_contradiction': {
      label: 'Coordination contradiction',
      what: 'Agents gave conflicting or repeated requests/decisions on the same task — e.g. supervisor asked the same question twice, or two agents proposed opposite actions on the same item.',
      source: 'Drift\'s 6-family LLM judge taxonomy',
    },
    'llm:grounding_failure': {
      label: 'Grounding failure',
      what: 'An agent referenced data, an entity, or a result that doesn\'t exist anywhere in the trace — a likely hallucination.',
      source: 'Drift\'s 6-family LLM judge taxonomy',
    },
    'llm:state_drift': {
      label: 'State drift',
      what: 'An agent ignored or contradicted state from prior steps — e.g. acted on stale information after another agent had already updated it.',
      source: 'Drift\'s 6-family LLM judge taxonomy',
    },
    'llm:emergent_decay': {
      label: 'Emergent decay',
      what: 'A pattern of degradation across steps — sentiment souring, quality dropping, repetition increasing. Spotted by looking at the whole trace, not any single step.',
      source: 'Drift\'s 6-family LLM judge taxonomy',
    },
    'llm:gate_bypass': {
      label: 'Gate bypass',
      what: 'An agent skipped a required check, approval, or verification step before taking an action that should have gated on it.',
      source: 'Drift\'s 6-family LLM judge taxonomy',
    },
    'verifier_always_approves': {
      label: 'Verifier always approves',
      what: 'A verifier-role agent approved >=95% of decisions across N runs with zero rejections — it isn\'t actually verifying anything. Real-world this means the safety layer is silently disabled.',
      source: 'MAST 3.x family + Anthropic engineering blog',
    },
    'infinite_handoff': {
      label: 'Infinite handoff',
      what: 'Two agents alternated past threshold (4+) with no state advancement — no new keys written, no new non-empty fields, no container growth. The classic "you handle it / no you handle it" loop.',
      source: 'MAST 1.3 (Step Repetition) + Cognition open problem #2',
    },
    'subagent_fanout_excess': {
      label: 'Subagent fanout excess',
      what: 'Orchestrator spawned more subagents than the task warranted — either too many distinct subagents (hard count) or too many subagents per measurable output (ratio rule). Note: this detector attributes fanout to whichever node NAME sounds managerial without checking execution order — on fixed-topology graphs with a node like "Research Manager" positioned mid-pipeline, it can misattribute earlier, unrelated nodes as its "subagents." Treat a fire on a static pipeline as a lead to verify, not a confirmed bug.',
      source: 'Anthropic multi-agent research postmortem (50-subagent incident)',
    },
  };

  function _glossaryEntry(failureType) {
    if (FINDING_GLOSSARY[failureType]) return FINDING_GLOSSARY[failureType];
    if (failureType && failureType.startsWith('llm:user_guideline')) {
      return {
        label: 'User guideline match',
        what: 'A plain-English rule the user supplied was triggered. Match index in the suffix.',
        source: 'User-supplied guideline',
      };
    }
    return null;
  }

  function renderFindingTypeBadge(failureType) {
    const entry = _glossaryEntry(failureType);
    const code = el('code', { class: 'muted', text: failureType });
    if (!entry) return code;

    const help = el('button', {
      type: 'button',
      class: 'finding-help',
      title: `What is ${entry.label}?`,
      text: '?',
    });
    const wrap = el('span', { class: 'finding-type-badge', style: 'display: inline-flex; align-items: center; gap: 4px;' }, [code, help]);

    help.addEventListener('click', (e) => {
      e.stopPropagation();
      const existing = wrap.querySelector('.finding-popover');
      if (existing) {
        existing.remove();
        return;
      }
      const pop = el('div', { class: 'finding-popover' }, [
        el('div', { style: 'font-weight: 600; margin-bottom: 4px;', text: entry.label }),
        el('div', { style: 'margin-bottom: 6px;', text: entry.what }),
        el('div', { class: 'muted', style: 'font-size: 11px;', text: 'Source: ' + entry.source }),
      ]);
      wrap.appendChild(pop);
      const dismiss = (ev) => {
        if (!wrap.contains(ev.target)) {
          pop.remove();
          document.removeEventListener('click', dismiss);
        }
      };
      setTimeout(() => document.addEventListener('click', dismiss), 0);
    });
    return wrap;
  }

  // ============================================================
  // FORK & REPLAY tab
  // ============================================================

  let _forkWired = false;
  async function initFork() {
    if (!_forkWired) {
      _forkWired = true;
      $('#fork-load-btn').addEventListener('click', loadForkSource);
    }
    await populateResultsDropdown($('#fork-load-results'), 'Choose a saved run…', true);
    renderForkSource();
  }

  async function loadForkSource() {
    const path = $('#fork-load-results').value;
    if (!path) return toast('Pick a saved run first', 'error');
    try {
      const data = await api('/api/results/' + path);
      if (!data.baseline || !data.baseline.trace) {
        return toast('That result has no captured trace to fork from', 'error');
      }
      setActiveRun(data, forkCtxFromLoadedResult(data));
      renderForkSource();
      toast('Loaded ' + path);
    } catch (e) {
      toast('Could not load: ' + e.message, 'error');
    }
  }

  function renderForkSource() {
    const status = $('#fork-source-status');
    const card = $('#fork-trace-card');
    const host = $('#fork-trace-list');
    const resultsHost = $('#fork-results-host');
    if (resultsHost) resultsHost.innerHTML = '';

    if (!_activeRun) {
      status.className = 'empty';
      status.textContent = 'No run loaded yet — run something in the Test tab, or load a saved run below.';
      card.classList.add('hidden');
      return;
    }
    status.className = 'muted';
    status.textContent = `Active: ${_activeRun.graph_name || '(unnamed)'} — ${(_activeRun.baseline.trace || []).length} step(s) captured.`;
    card.classList.remove('hidden');
    host.innerHTML = '';
    host.appendChild(renderTraceList(_activeRun.baseline.trace || [], { forkable: true }));
  }

  // ============================================================
  // ASK tab
  // ============================================================

  let _askWired = false;
  async function initAsk() {
    if (!_askWired) {
      _askWired = true;
      $('#ask-load-btn').addEventListener('click', loadAskSource);
      $('#ask-submit').addEventListener('click', submitAsk);
    }
    // Unlike Fork (which needs a full trace to rewind to), Ask can work
    // from summary-only files too — list everything, not just trace-havers.
    await populateResultsDropdown($('#ask-load-results'), 'Choose a saved run…', false);
    if (_activeRun) describeAskSource(_activeRun, _activeRun.graph_name || '(active run)');
  }

  function _runHasTrace(run) {
    return Array.isArray(run?.baseline?.trace) && run.baseline.trace.length > 0;
  }

  function describeAskSource(run, label) {
    $('#ask-source-status').className = '';
    $('#ask-source-status').textContent = _runHasTrace(run)
      ? `Loaded: ${label} (${run.baseline.trace.length} steps — full trace, step-level answers)`
      : `Loaded: ${label} (summary only — no per-step trace was saved for this run; answers will be coarser)`;
  }

  async function loadAskSource() {
    const path = $('#ask-load-results').value;
    if (!path) return toast('Pick a saved run first', 'error');
    try {
      const data = await api('/api/results/' + path);
      setActiveRun(data, _activeForkCtx);
      describeAskSource(data, data.graph_name || path);
      toast('Loaded ' + path);
    } catch (e) {
      toast('Could not load: ' + e.message, 'error');
    }
  }

  async function submitAsk() {
    const question = ($('#ask-question').value || '').trim();
    if (!question) return toast('Type a question first', 'error');
    if (!_activeRun) {
      return toast('Load a run first — run something in Test, or load a saved run above', 'error');
    }
    const btn = $('#ask-submit');
    const status = $('#ask-status');
    const judge = $('#ask-judge').value;
    const judge_model = ($('#ask-judge-model').value || '').trim() || null;
    btn.disabled = true;
    status.textContent = 'thinking…';
    try {
      const hasTrace = _runHasTrace(_activeRun);
      const body = hasTrace
        ? {
            trace: _activeRun.baseline.trace,
            initial_state: _activeRun.baseline.initial_state || {},
            final_state: _activeRun.baseline.final_state || null,
            question, judge, judge_model,
          }
        : { summary: _activeRun, question, judge, judge_model };
      const res = await api('/api/trace-ask', { method: 'POST', body });
      $('#ask-answer-card').classList.remove('hidden');
      $('#ask-answer').innerHTML = '';
      if (!hasTrace) {
        $('#ask-answer').appendChild(el('div', { class: 'help', style: 'margin-bottom: 6px;',
          text: 'Answered from summary data only — this run has no saved per-step trace.' }));
      }
      $('#ask-answer').appendChild(el('div', { class: 'mono', style: 'white-space: pre-wrap; word-break: break-word;', text: res.answer }));
      $('#ask-answer-card').scrollIntoView({ behavior: 'smooth', block: 'start' });
      status.textContent = 'done';
    } catch (e) {
      toast('Ask failed: ' + e.message, 'error');
      status.textContent = '';
    } finally {
      btn.disabled = false;
    }
  }

  // ============================================================
  // COMPARE tab
  // ============================================================

  let _compareWired = false;
  async function initCompare() {
    if (!_compareWired) {
      _compareWired = true;
      $('#compare-run').addEventListener('click', runCompare);
    }
    await populateResultsDropdown($('#compare-a'), 'Choose a saved run…');
    await populateResultsDropdown($('#compare-b'), 'Choose a saved run…');
  }

  // This repo's results/ directory holds several incompatible shapes from
  // different eras of experiment scripts. Rather than force every shape
  // into one schema (which would silently show "0 findings" for a file
  // that actually has findings under different key names), recognize the
  // shapes we know and say plainly when a file's shape isn't one of them —
  // wrong zeros are worse than an honest "can't compare this."
  function normalizeForCompare(data, label) {
    // Shape 1: adapter-shaped — /api/adapter-demo, /api/custom-graph-run,
    // and the current run_drift_on_*.py scripts all save this shape.
    if (data && data.baseline && typeof data.baseline === 'object' && !Array.isArray(data.baseline)) {
      const baseline = data.baseline;
      const trace = Array.isArray(baseline.trace) ? baseline.trace : [];
      return {
        label, supported: true,
        stats: {
          'Perturbations': (data.perturbations || []).length,
          'Crashed': data.n_crashed ?? 0,
          'Diverged': data.n_diverged ?? 0,
          'Judge findings': data.n_judge_findings ?? (baseline.judge_findings || []).length,
          'Coord findings': data.n_coordination_findings ?? (baseline.coordination_findings || []).length,
          'Baseline trace steps': trace.length || baseline.trace_steps || 0,
        },
        findings: [
          ...(baseline.judge_findings || []).map(f => ({ ...f, _source: 'judge' })),
          ...(baseline.coordination_findings || []).map(f => ({ ...f, _source: 'coord' })),
        ],
        trace,
      };
    }
    // Shape 2: flat legacy shape (baseline_* top-level fields) — older
    // one-off scripts like the original run_drift_on_deepagents.py.
    if (data && ('baseline_crashed' in data || 'baseline_trace_steps' in data)) {
      const judge = data.baseline_judge || [];
      const coord = data.baseline_coord || [];
      const asFinding = (f, source) => ({
        failure_type: f.failure_type || f.type || source,
        summary: f.summary || JSON.stringify(f),
        _source: source,
      });
      return {
        label, supported: true,
        stats: {
          'Perturbations': (data.perturbations || []).length,
          'Crashed': data.n_crashed ?? 0,
          'Diverged': data.n_diverged ?? 0,
          'Judge findings': data.n_judge ?? judge.length,
          'Coord findings': data.n_coord ?? coord.length,
          'Baseline trace steps': data.baseline_trace_steps || 0,
        },
        findings: [
          ...judge.map(f => asFinding(f, 'judge')),
          ...coord.map(f => asFinding(f, 'coord')),
        ],
        trace: [],  // this shape never saved the full per-step trace, only a count
      };
    }
    return { label, supported: false };
  }

  async function runCompare() {
    const pathA = $('#compare-a').value;
    const pathB = $('#compare-b').value;
    if (!pathA || !pathB) return toast('Pick two saved runs', 'error');
    const status = $('#compare-status');
    status.textContent = 'loading…';
    try {
      const [dataA, dataB] = await Promise.all([
        api('/api/results/' + pathA),
        api('/api/results/' + pathB),
      ]);
      const a = normalizeForCompare(dataA, pathA);
      const b = normalizeForCompare(dataB, pathB);
      renderCompare(a, b);
      $('#compare-result').classList.remove('hidden');
      $('#compare-result').scrollIntoView({ behavior: 'smooth', block: 'start' });
      status.textContent = '';
    } catch (e) {
      toast('Compare failed: ' + e.message, 'error');
      status.textContent = '';
    }
  }

  function renderCompare(a, b) {
    const summaryHost = $('#compare-summary');
    const findingsHost = $('#compare-findings');
    const traceHost = $('#compare-trace');
    summaryHost.innerHTML = '';
    findingsHost.innerHTML = '';
    traceHost.innerHTML = '';

    if (!a.supported || !b.supported) {
      const bad = [!a.supported && a.label, !b.supported && b.label].filter(Boolean).join(' and ');
      summaryHost.appendChild(el('div', { class: 'empty',
        text: `${bad} — not a recognized result shape for comparison (likely an older experiment-script format with different field names). Open it in the Results tab to view the raw JSON instead.` }));
      return;
    }

    const keys = new Set([...Object.keys(a.stats), ...Object.keys(b.stats)]);
    const table = el('table', { class: 'ov-table' }, [
      el('thead', {}, el('tr', {}, [
        el('th', { text: '' }),
        el('th', { text: a.label }),
        el('th', { text: b.label }),
      ])),
      el('tbody', {}, [...keys].map(k => {
        const va = a.stats[k], vb = b.stats[k];
        const differs = va !== vb;
        return el('tr', {}, [
          el('td', { text: k }),
          el('td', { class: differs ? 'num' : 'num muted', text: String(va ?? '—') }),
          el('td', { class: differs ? 'num' : 'num muted', text: String(vb ?? '—') }),
        ]);
      })),
    ]);
    summaryHost.appendChild(el('div', { class: 'table-wrap' }, [table]));

    const findingsRow = el('div', { class: 'trace-compare' }, [
      el('div', { class: 'trace-compare-col' }, [
        el('h4', { text: `${a.label} (${a.findings.length})` }),
        a.findings.length
          ? el('div', {}, a.findings.map(f => f._source === 'judge' ? renderJudgeFinding(f) : renderCoordFinding(f)))
          : el('div', { class: 'empty', text: 'No findings.' }),
      ]),
      el('div', { class: 'trace-compare-col' }, [
        el('h4', { text: `${b.label} (${b.findings.length})` }),
        b.findings.length
          ? el('div', {}, b.findings.map(f => f._source === 'judge' ? renderJudgeFinding(f) : renderCoordFinding(f)))
          : el('div', { class: 'empty', text: 'No findings.' }),
      ]),
    ]);
    findingsHost.appendChild(findingsRow);

    traceHost.appendChild(el('div', { class: 'trace-compare' }, [
      el('div', { class: 'trace-compare-col' }, [
        el('h4', { text: `${a.label} (${a.trace.length} step(s))` }),
        renderTraceList(a.trace),
      ]),
      el('div', { class: 'trace-compare-col' }, [
        el('h4', { text: `${b.label} (${b.trace.length} step(s))` }),
        renderTraceList(b.trace),
      ]),
    ]));
  }

  // ============================================================
  // IMPORT GRAPH tab (BYOA)
  // ============================================================

  let _importWired = false;
  function initImport() {
    if (_importWired) return;
    _importWired = true;
    $('#import-run').addEventListener('click', runCustomGraph);
    $('#import-to-fork').addEventListener('click', () => sendActiveRunTo('fork'));
    $('#import-to-ask').addEventListener('click', () => sendActiveRunTo('ask'));
    $('#import-save-run').addEventListener('click', saveActiveRunToResults);
  }

  async function runCustomGraph() {
    const module = ($('#import-module').value || '').trim();
    if (!module) return toast('Module path is required', 'error');
    const build_fn = ($('#import-build-fn').value || 'build').trim();
    const initial_state_fn = ($('#import-state-fn').value || 'initial_state').trim();
    const query = ($('#import-query').value || '').trim();
    const model = ($('#import-model').value || '').trim() || null;
    const intensity = $('#import-intensity').value;
    const judge = $('#import-judge').value;
    const max_perturbations = parseInt($('#import-max-perturbations').value, 10) || 0;
    const coordination_roles = parseRolesTextarea('#import-coordination-roles');
    const required_followups = parseRolesTextarea('#import-required-followups');
    const policy_field = ($('#import-policy-field').value || '').trim();
    const track_history = $('#import-track-history').checked;

    const btn = $('#import-run');
    const status = $('#import-status');
    btn.disabled = true;
    status.textContent = 'importing + running (a real graph invocation can take minutes)…';
    try {
      const data = await api('/api/custom-graph-run', {
        method: 'POST',
        body: {
          module, build_fn, initial_state_fn, query, model,
          intensity, judge, max_perturbations,
          coordination_roles, required_followups, policy_field, track_history,
        },
      });
      setActiveRun(data, { module, build_fn, initial_state_fn, query, model });
      renderAdapterResult(data, 'import');
      $('#import-result').classList.remove('hidden');
      $('#import-result').scrollIntoView({ behavior: 'smooth', block: 'start' });
      const parts = [`${data.perturbations.length} perturbations`];
      if (data.n_judge_findings) parts.push(`${data.n_judge_findings} judge finding(s)`);
      if (data.n_coordination_findings) parts.push(`${data.n_coordination_findings} coord finding(s)`);
      status.textContent = `done — ${parts.join(' · ')}`;
    } catch (e) {
      toast('Run failed: ' + e.message, 'error');
      status.textContent = '';
    } finally {
      btn.disabled = false;
    }
  }

  // ============================================================
  // BUILD tab
  // ============================================================

  const BUILD_TOPOLOGY_META = {
    supervisor: { min: 2, max: null,
      hint: 'First agent is the coordinator; the rest are specialists it delegates to in order.' },
    swarm: { min: 2, max: null,
      hint: 'Agents hand off in the order listed, peer to peer.' },
    pipeline: { min: 2, max: null,
      hint: 'Agents run as sequential stages in the order listed.' },
    loop: { min: 2, max: 2,
      hint: 'Exactly 2 agents: first is the generator, second is the critic.' },
    fanout: { min: 4, max: null,
      hint: 'First agent is the dispatcher, last is the merge step; everyone in between runs in parallel (at least 2 workers).' },
  };
  const BUILD_DEFAULT_AGENTS = {
    supervisor: [
      { name: 'coordinator', role: 'Break the task into subtasks and decide what each specialist should cover.' },
      { name: 'researcher', role: 'Research facts relevant to the task.' },
      { name: 'writer', role: 'Write the final answer using what the team found.' },
    ],
    swarm: [
      { name: 'intake', role: 'Understand the request and note what it actually needs.' },
      { name: 'solver', role: 'Solve the request based on what intake found.' },
    ],
    pipeline: [
      { name: 'draft', role: 'Draft a first-pass response to the task.' },
      { name: 'finalize', role: 'Polish the draft into a final response.' },
    ],
    loop: [
      { name: 'generator', role: 'Generate a solution. If revising, improve on the critic\'s feedback.' },
      { name: 'critic', role: 'Review the solution. Say APPROVED if it is good, else say REVISE with specific feedback.' },
    ],
    fanout: [
      { name: 'dispatcher', role: 'Split the task into independent angles for the workers to cover.' },
      { name: 'worker_a', role: 'Cover one angle of the task.' },
      { name: 'worker_b', role: 'Cover a different angle of the task.' },
      { name: 'merge', role: "Combine all workers' findings into one final answer." },
    ],
  };

  let _buildWired = false;
  let _buildAgents = [];  // [{name, role}, ...] — current form state
  let _buildSuggestedName = '';   // from the last /api/propose-target call, if any
  let _buildSuggestedQuery = '';  // ditto — surfaced for manual use in Import graph's Query field

  function initBuild() {
    if (_buildWired) return;
    _buildWired = true;
    $('#build-topology').addEventListener('change', onBuildTopologyChange);
    $('#build-add-agent').addEventListener('click', addBuildAgentRow);
    $('#build-generate').addEventListener('click', generateBuildTarget);
    $('#build-save').addEventListener('click', saveBuildTarget);
    $('#build-suggest').addEventListener('click', suggestBuildAgents);
    onBuildTopologyChange();  // seed default agents for the preselected topology
  }

  async function suggestBuildAgents() {
    const scenario = ($('#build-scenario').value || '').trim();
    if (!scenario) return toast('Describe a scenario first', 'error');

    const btn = $('#build-suggest');
    const status = $('#build-suggest-status');
    const rationaleEl = $('#build-suggest-rationale');
    btn.disabled = true;
    status.textContent = 'thinking…';
    rationaleEl.classList.add('hidden');
    try {
      const data = await api('/api/propose-target', {
        method: 'POST',
        body: { scenario },
      });
      // Setting .value directly does NOT fire the 'change' listener (that
      // only fires on user interaction), so onBuildTopologyChange — which
      // would reset _buildAgents to the topology's generic defaults — never
      // runs here. Set the agent roster straight from the proposal instead.
      $('#build-topology').value = data.topology;
      $('#build-topology-hint').textContent = BUILD_TOPOLOGY_META[data.topology].hint;
      _buildAgents = data.agents.map(a => ({ name: a.name, role: a.role }));
      _buildSuggestedName = data.suggested_name || '';
      _buildSuggestedQuery = data.suggested_query || '';
      renderBuildAgentRows();
      const notes = [];
      if (data.rationale) notes.push(`Why ${data.topology}: ${data.rationale}`);
      if (_buildSuggestedQuery) {
        notes.push(`Suggested test query (paste into Import graph's Query field once saved): "${_buildSuggestedQuery}"`);
      }
      if (notes.length) {
        rationaleEl.textContent = notes.join(' ');
        rationaleEl.classList.remove('hidden');
      }
      status.textContent = 'drafted — review/edit the roster below, then Generate';
    } catch (e) {
      toast('Suggest failed: ' + e.message, 'error');
      status.textContent = '';
    } finally {
      btn.disabled = false;
    }
  }

  function onBuildTopologyChange() {
    const topology = $('#build-topology').value;
    const meta = BUILD_TOPOLOGY_META[topology];
    $('#build-topology-hint').textContent = meta.hint;
    _buildAgents = BUILD_DEFAULT_AGENTS[topology].map(a => ({ ...a }));
    renderBuildAgentRows();
  }

  function renderBuildAgentRows() {
    const topology = $('#build-topology').value;
    const meta = BUILD_TOPOLOGY_META[topology];
    const host = $('#build-agent-rows');
    host.innerHTML = '';
    _buildAgents.forEach((agent, i) => {
      const nameInput = el('input', { type: 'text', value: agent.name, placeholder: 'agent name' });
      nameInput.addEventListener('input', () => { _buildAgents[i].name = nameInput.value; });
      const roleInput = el('textarea', { class: 'mono', rows: '2', text: agent.role,
        placeholder: 'what this agent should do' });
      roleInput.addEventListener('input', () => { _buildAgents[i].role = roleInput.value; });
      const removeBtn = el('button', {
        type: 'button', class: 'ghost-btn sm', text: '✕',
        onclick: () => removeBuildAgentRow(i),
      });
      if (_buildAgents.length <= meta.min) {
        removeBtn.disabled = true;
      }
      host.appendChild(el('div', {
        class: 'row', style: 'display: grid; grid-template-columns: 1fr 2fr auto; gap: 8px; align-items: start; margin-bottom: 8px;',
      }, [nameInput, roleInput, removeBtn]));
    });
    const addBtn = $('#build-add-agent');
    addBtn.disabled = meta.max != null && _buildAgents.length >= meta.max;
  }

  function addBuildAgentRow() {
    const topology = $('#build-topology').value;
    const meta = BUILD_TOPOLOGY_META[topology];
    if (meta.max != null && _buildAgents.length >= meta.max) return;
    _buildAgents.push({ name: '', role: '' });
    renderBuildAgentRows();
  }

  function removeBuildAgentRow(index) {
    const topology = $('#build-topology').value;
    const meta = BUILD_TOPOLOGY_META[topology];
    if (_buildAgents.length <= meta.min) return;
    _buildAgents.splice(index, 1);
    renderBuildAgentRows();
  }

  async function generateBuildTarget() {
    const topology = $('#build-topology').value;
    const model = ($('#build-model').value || 'gpt-4o-mini').trim();
    const agents = _buildAgents.map(a => ({ name: a.name.trim(), role: a.role.trim() }));
    if (agents.some(a => !a.name || !a.role)) {
      return toast('Every agent needs both a name and a role description', 'error');
    }

    const btn = $('#build-generate');
    const status = $('#build-status');
    btn.disabled = true;
    status.textContent = 'generating…';
    try {
      const data = await api('/api/build-target', {
        method: 'POST',
        body: { topology, agents, model },
      });
      $('#build-code').value = data.code;
      $('#build-preview').classList.remove('hidden');
      $('#build-save-result').classList.add('hidden');
      const suggested = _buildSuggestedName
        || `${topology}_${agents.map(a => a.name.toLowerCase().replace(/\s+/g, '_')).join('_')}`.slice(0, 50);
      $('#build-save-name').value = suggested;
      $('#build-preview').scrollIntoView({ behavior: 'smooth', block: 'start' });
      status.textContent = 'generated — review/edit the code below, then save it';
    } catch (e) {
      toast('Generate failed: ' + e.message, 'error');
      status.textContent = '';
    } finally {
      btn.disabled = false;
    }
  }

  async function saveBuildTarget() {
    const name = ($('#build-save-name').value || '').trim();
    const code = $('#build-code').value;
    if (!name) return toast('Name is required', 'error');

    const btn = $('#build-save');
    const resultHost = $('#build-save-result');
    btn.disabled = true;
    try {
      const data = await api('/api/save-target', {
        method: 'POST',
        body: { name, code },
      });
      resultHost.innerHTML = '';
      resultHost.classList.remove('hidden');
      resultHost.appendChild(el('div', { class: 'help',
        text: `Saved as ${data.module} (${data.path}).` }));
      resultHost.appendChild(el('button', {
        class: 'ghost-btn', style: 'margin-top: 6px;', text: 'Test it in Import graph →',
        onclick: () => {
          activateTab('import');
          $('#import-module').value = data.module;
          $('#import-build-fn').value = 'build';
          $('#import-state-fn').value = 'initial_state';
          toast('Module path filled in — set a query and run when ready', 'info');
        },
      }));
      toast('Saved', 'success');
    } catch (e) {
      toast('Save failed: ' + e.message, 'error');
    } finally {
      btn.disabled = false;
    }
  }

  // ============================================================
  // TEST SUITE tab
  // ============================================================

  let _suiteWired = false;
  let _suiteTargets = [];  // cached rows from GET /api/targets

  function initSuite() {
    if (_suiteWired) return;
    _suiteWired = true;
    $('#suite-run').addEventListener('click', runSuite);
    $('#suite-close-detail').addEventListener('click', () => {
      $('#suite-detail').classList.add('hidden');
    });
    loadSuiteTargets();
  }

  async function loadSuiteTargets() {
    const host = $('#suite-target-list');
    host.classList.remove('empty');
    host.innerHTML = 'Loading…';
    try {
      const data = await api('/api/targets');
      _suiteTargets = data.targets || [];
      renderSuiteTargetList();
    } catch (e) {
      host.classList.add('empty');
      host.textContent = 'Failed to load targets: ' + e.message;
    }
  }

  function renderSuiteTargetList() {
    const host = $('#suite-target-list');
    host.innerHTML = '';
    if (!_suiteTargets.length) {
      host.classList.add('empty');
      host.textContent = 'No targets registered under examples/targets/.';
      return;
    }
    host.classList.remove('empty');
    const list = el('div', { class: 'trace-list' });
    _suiteTargets.forEach(t => {
      const disabled = !t.importable;
      const badges = [el('span', { class: 'pill', text: t.topology })];
      if (disabled) {
        badges.push(el('span', { class: 'pill danger', text: 'missing: ' + t.missing_deps.join(', ') }));
      } else if (t.needs_api_key) {
        badges.push(el('span', { class: 'pill warning', text: 'needs API key' }));
      }
      const checkbox = el('input', {
        type: 'checkbox', class: 'suite-target-checkbox',
      });
      checkbox.dataset.name = t.name;
      if (disabled) checkbox.disabled = true;
      const row = el('label', {
        class: 'trace-step',
        style: `grid-template-columns: auto 1fr auto; align-items: center; gap: 10px; `
          + `cursor: ${disabled ? 'not-allowed' : 'pointer'}; opacity: ${disabled ? 0.55 : 1};`,
      }, [
        checkbox,
        el('div', {}, [
          el('div', { class: 'trace-step-node', text: t.name }),
          el('div', { class: 'trace-step-summary', text: t.description }),
        ]),
        el('div', { style: 'display: flex; gap: 6px; align-items: center;' }, badges),
      ]);
      list.appendChild(row);
    });
    host.appendChild(list);
  }

  function selectedSuiteTargets() {
    return $$('.suite-target-checkbox:checked').map(cb => cb.dataset.name);
  }

  async function runSuite() {
    const target_names = selectedSuiteTargets();
    if (!target_names.length) return toast('Select at least one target', 'error');

    const intensity = $('#suite-intensity').value;
    const seed = parseInt($('#suite-seed').value, 10) || 7;
    const max_perturbations = parseInt($('#suite-max-perturbations').value, 10) || 0;
    const judge = $('#suite-judge').value;
    const track_history = $('#suite-track-history').checked;

    const btn = $('#suite-run');
    const status = $('#suite-status');
    btn.disabled = true;
    status.textContent = `running ${target_names.length} target(s) sequentially — this can take a while…`;
    try {
      const data = await api('/api/suite-run', {
        method: 'POST',
        body: { target_names, intensity, seed, max_perturbations, judge, track_history },
      });
      const results = data.results || [];
      renderSuiteResults(results);
      const ok = results.filter(r => r.status === 'ok').length;
      const skipped = results.filter(r => r.status === 'skipped').length;
      const errored = results.filter(r => r.status === 'error').length;
      status.textContent = `done — ${ok} ok · ${skipped} skipped · ${errored} error(s)`;
    } catch (e) {
      toast('Suite run failed: ' + e.message, 'error');
      status.textContent = '';
    } finally {
      btn.disabled = false;
    }
  }

  const SUITE_STATUS_PILL = { ok: 'success', skipped: 'warning', error: 'danger' };

  function renderSuiteResults(results) {
    const card = $('#suite-results');
    const host = $('#suite-results-list');
    const totalsHost = $('#suite-summary-totals');
    host.innerHTML = '';
    totalsHost.innerHTML = '';
    card.classList.remove('hidden');

    const ok = results.filter(r => r.status === 'ok');
    const agg = ok.reduce((acc, r) => {
      acc.crashed += r.result?.n_crashed || 0;
      acc.diverged += r.result?.n_diverged || 0;
      acc.coord += r.result?.n_coordination_findings || 0;
      acc.judge += r.result?.n_judge_findings || 0;
      return acc;
    }, { crashed: 0, diverged: 0, coord: 0, judge: 0 });
    [
      ['targets run', `${ok.length} ok · ${results.length - ok.length} skipped/errored`],
      ['crashed (total)', agg.crashed],
      ['diverged (total)', agg.diverged],
      ['coord findings (total)', agg.coord],
      ['judge findings (total)', agg.judge],
    ].forEach(([k, v]) => {
      totalsHost.appendChild(el('span', { class: 'k', text: k }));
      totalsHost.appendChild(el('span', { class: 'v', text: String(v) }));
    });

    results.forEach(r => {
      const summaryParts = [];
      if (r.result) {
        summaryParts.push(`${r.result.n_crashed} crashed`, `${r.result.n_diverged} diverged`);
        if (r.result.n_coordination_findings) {
          summaryParts.push(`${r.result.n_coordination_findings} coord finding(s)`);
        }
        if (r.result.n_judge_findings) {
          summaryParts.push(`${r.result.n_judge_findings} judge finding(s)`);
        }
      } else if (r.error) {
        summaryParts.push(r.error);
      }

      const row = el('div', { class: 'card tight', style: 'margin-bottom: 8px;' }, [
        el('div', { style: 'display: flex; justify-content: space-between; align-items: center; gap: 10px;' }, [
          el('div', { style: 'font-weight: 600;', text: r.target_name }),
          el('span', { class: `pill ${SUITE_STATUS_PILL[r.status] || 'same'}`, text: r.status }),
        ]),
        el('div', { class: 'muted', style: 'font-size: 12.5px; margin-top: 4px;',
                    text: summaryParts.join(' · ') || '—' }),
      ]);

      if (r.result) {
        row.appendChild(el('button', {
          class: 'ghost-btn sm', text: 'View full result', style: 'margin-top: 8px;',
          onclick: () => showSuiteDetail(r),
        }));
      }
      host.appendChild(row);
    });
  }

  function showSuiteDetail(r) {
    $('#suite-detail-title').textContent = r.target_name;
    renderAdapterResult(r.result, 'suite');
    $('#suite-detail').classList.remove('hidden');
    $('#suite-detail').scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  // ============================================================
  // SIMULATE tab (historical grounding + calibration)
  // ============================================================

  let _simulateWired = false;
  let _lastSimulateResult = null;

  function initSimulate() {
    if (_simulateWired) return;
    _simulateWired = true;
    $('#simulate-run').addEventListener('click', runSimulate);
    $('#calibrate-run').addEventListener('click', runCalibration);
    $('#simulate-download-json').addEventListener('click', () => {
      if (_lastSimulateResult) downloadJson(_lastSimulateResult, `simulate_${Date.now()}.json`);
    });
    loadHistoricalEvents();
  }

  async function loadHistoricalEvents() {
    const host = $('#historical-events-list');
    host.innerHTML = 'Loading…';
    try {
      const data = await api('/api/historical-events');
      const events = data.events || [];
      $('#calibrate-event-count').textContent = events.length;
      host.innerHTML = '';
      const list = el('div', { class: 'trace-list' });
      events.forEach(e => {
        const pillClass = e.actual_direction === 'up' ? 'success'
          : e.actual_direction === 'down' ? 'danger' : 'same';
        list.appendChild(el('div', { class: 'trace-step', style: 'grid-template-columns: auto 1fr auto; gap: 10px; align-items: center;' }, [
          el('div', { class: 'trace-step-node', text: e.ticker }),
          el('div', { class: 'trace-step-summary', text: `${e.reaction_date}: ${e.description}` }),
          el('span', { class: `pill ${pillClass}`, text: `${e.actual_move_pct > 0 ? '+' : ''}${e.actual_move_pct.toFixed(1)}%` }),
        ]));
      });
      host.appendChild(list);
    } catch (e) {
      host.textContent = 'Failed to load historical events: ' + e.message;
    }
  }

  async function runSimulate() {
    const scenario = ($('#simulate-scenario').value || '').trim();
    if (!scenario) return toast('Describe a scenario first', 'error');
    const ground = $('#simulate-ground').checked;

    const btn = $('#simulate-run');
    const status = $('#simulate-status');
    btn.disabled = true;
    status.textContent = 'thinking — proposing a roster, grounding, then running the simulation (can take a minute)…';
    try {
      const data = await api('/api/simulate-scenario', {
        method: 'POST',
        body: { scenario, ground },
      });
      _lastSimulateResult = data;
      renderSimulateResult(data);
      status.textContent = `done — ${data.topology} topology`;
    } catch (e) {
      toast('Simulate failed: ' + e.message, 'error');
      status.textContent = '';
    } finally {
      btn.disabled = false;
    }
  }

  function renderSimulateResult(data) {
    const card = $('#simulate-result');
    card.classList.remove('hidden');

    const verdictHost = $('#simulate-verdict');
    verdictHost.innerHTML = '';
    const verdict = data.verdict || {};
    const directionPill = verdict.direction === 'up' ? 'success'
      : verdict.direction === 'down' ? 'danger' : 'same';
    const kvs = [
      ['Topology', data.topology],
      ['Why this topology', data.rationale],
    ];
    kvs.forEach(([k, v]) => {
      verdictHost.appendChild(el('span', { class: 'k', text: k }));
      verdictHost.appendChild(el('span', { class: 'v', text: v || '—' }));
    });
    verdictHost.appendChild(el('span', { class: 'k', text: 'Direction' }));
    verdictHost.appendChild(el('span', { class: 'v' }, [
      el('span', { class: `pill ${directionPill}`, text: verdict.direction || 'unknown' }),
    ]));
    if (verdict.magnitude_pct != null) {
      verdictHost.appendChild(el('span', { class: 'k', text: 'Magnitude' }));
      verdictHost.appendChild(el('span', { class: 'v', text: `${verdict.magnitude_pct}%` }));
    }
    if (verdict.confidence) {
      verdictHost.appendChild(el('span', { class: 'k', text: 'Confidence' }));
      verdictHost.appendChild(el('span', { class: 'v', text: verdict.confidence }));
    }
    if (verdict.reasoning) {
      verdictHost.appendChild(el('span', { class: 'k', text: 'Reasoning' }));
      verdictHost.appendChild(el('span', { class: 'v', text: verdict.reasoning }));
    }

    const groundingHost = $('#simulate-grounding');
    const groundingList = $('#simulate-grounding-list');
    groundingList.innerHTML = '';
    const analogues = data.grounded_on || [];
    if (analogues.length) {
      groundingHost.classList.remove('hidden');
      analogues.forEach(e => {
        groundingList.appendChild(el('div', { class: 'card tight', style: 'margin-bottom: 6px;' }, [
          el('div', { style: 'font-weight: 600;', text: `${e.ticker} — ${e.reaction_date}` }),
          el('div', { class: 'muted', style: 'font-size: 12.5px;', text: e.description }),
          el('div', { class: 'mono', style: 'font-size: 12px; margin-top: 4px;',
                      text: `actual result: ${e.actual_move_pct > 0 ? '+' : ''}${e.actual_move_pct.toFixed(1)}% (${e.actual_direction})` }),
        ]));
      });
    } else {
      groundingHost.classList.add('hidden');
    }

    $('#simulate-notes').textContent = JSON.stringify(data.notes || {}, null, 2);
    card.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  async function runCalibration() {
    const ground = $('#calibrate-ground').checked;
    const btn = $('#calibrate-run');
    const status = $('#calibrate-status');
    btn.disabled = true;
    status.textContent = 'running against every historical event — this makes ~30-50 real LLM calls and can take a few minutes…';
    try {
      const data = await api('/api/calibrate', {
        method: 'POST',
        body: { ground },
      });
      renderCalibrationResult(data);
      const pct = data.accuracy != null ? `${(data.accuracy * 100).toFixed(0)}%` : 'n/a';
      status.textContent = `done — accuracy ${pct} (${data.n_matched}/${data.n_scored} scored, ${data.n_total} total)`;
    } catch (e) {
      toast('Calibration failed: ' + e.message, 'error');
      status.textContent = '';
    } finally {
      btn.disabled = false;
    }
  }

  function renderCalibrationResult(data) {
    const host = $('#calibrate-result');
    host.innerHTML = '';
    host.classList.remove('hidden');

    const pct = data.accuracy != null ? `${(data.accuracy * 100).toFixed(0)}%` : 'n/a';
    host.appendChild(el('div', { style: 'font-weight: 600; margin-bottom: 8px;',
      text: `Accuracy: ${pct} (${data.n_matched}/${data.n_scored} scored correctly, ${data.n_total} events total)` }));

    (data.results || []).forEach(r => {
      const statusPill = r.error ? 'warning' : (r.matched ? 'success' : (r.predicted_direction ? 'danger' : 'same'));
      const statusText = r.error ? 'error' : (r.predicted_direction ? (r.matched ? 'matched' : 'missed') : 'unscored');
      host.appendChild(el('div', { class: 'card tight', style: 'margin-bottom: 6px;' }, [
        el('div', { style: 'display: flex; justify-content: space-between; align-items: center; gap: 10px;' }, [
          el('div', { style: 'font-weight: 600;', text: `${r.event.ticker} — actual: ${r.event.actual_direction}` }),
          el('span', { class: `pill ${statusPill}`, text: statusText }),
        ]),
        el('div', { class: 'muted', style: 'font-size: 12.5px; margin-top: 4px;',
                    text: r.error || `predicted: ${r.predicted_direction || 'unparseable'}` }),
      ]));
    });
  }

  // ============================================================
  // OVERVIEW tab
  // ============================================================

  let _overviewWired = false;
  function initOverview() {
    if (_overviewWired) return;
    _overviewWired = true;
    refreshOverview();
  }

  function _openFromOverview(path) {
    activateTab('results');
    openResultsFile(path);
  }

  async function refreshOverview() {
    const totalsHost = $('#overview-totals');
    const systemsCard = $('#overview-systems-card');
    const systemsHost = $('#overview-systems');
    const chartCard = $('#overview-chart-card');
    const chartHost = $('#overview-chart');
    const findingsCard = $('#overview-findings-card');
    const findingsHost = $('#overview-findings');
    totalsHost.classList.remove('empty');
    totalsHost.innerHTML = 'Loading…';
    try {
      const data = await api('/api/overview');
      const t = data.totals;
      if (!t.runs) {
        totalsHost.classList.add('empty');
        totalsHost.textContent = 'No saved results yet. Run something from the Test tab, or an example script with --save-json.';
        systemsCard.classList.add('hidden');
        chartCard.classList.add('hidden');
        findingsCard.classList.add('hidden');
        return;
      }
      totalsHost.innerHTML = '';
      const kv = el('div', { class: 'kv-grid' });
      [
        ['systems tested', t.systems],
        ['agents seen', t.agents],
        ['total runs', t.runs],
        ['chaos perturbations', t.perturbations],
        ['diverged', t.diverged],
        ['crashed', t.crashed],
        ['findings', t.findings],
      ].forEach(([k, v]) => {
        kv.appendChild(el('span', { class: 'k', text: k }));
        kv.appendChild(el('span', { class: 'v', text: String(v) }));
      });
      totalsHost.appendChild(kv);

      const findingTypes = data.finding_types || [];
      chartCard.classList.toggle('hidden', findingTypes.length === 0);
      chartHost.innerHTML = '';
      if (findingTypes.length) {
        const chart = el('div', { class: 'bar-chart' });
        const maxCount = Math.max(...findingTypes.map(ft => ft.count));
        const totalCount = findingTypes.reduce((sum, ft) => sum + ft.count, 0);
        findingTypes.forEach(ft => {
          const pct = totalCount ? Math.round((ft.count / totalCount) * 100) : 0;
          const widthPct = maxCount ? (ft.count / maxCount) * 100 : 0;
          chart.appendChild(el('div', { class: 'bar-chart-row' }, [
            el('div', { class: 'bar-chart-label' }, [
              el('span', { class: 'name', text: ft.type }),
              el('span', { class: 'value', text: String(ft.count) }),
            ]),
            el('div', { class: 'bar-chart-track' }, [
              el('div', { class: 'bar-chart-fill', style: `width: ${widthPct}%;` }),
            ]),
            el('div', { class: 'bar-chart-tooltip', text:
              `${ft.count} of ${totalCount} finding(s) (${pct}%)` }),
          ]));
        });
        chartHost.appendChild(chart);
      }

      systemsCard.classList.toggle('hidden', data.systems.length === 0);
      systemsHost.innerHTML = '';
      const sList = el('div', { class: 'trace-list' });
      data.systems.forEach(s => {
        sList.appendChild(el('div', { class: 'trace-step', style: 'grid-template-columns: 1fr auto; cursor: default;' }, [
          el('div', { class: 'trace-step-body' }, [
            el('div', { class: 'trace-step-node', text: s.group }),
            el('div', { class: 'trace-step-summary', text: s.agents.join(', ') || '—' }),
          ]),
          el('div', { style: 'display: flex; align-items: center; gap: 10px;' }, [
            el('span', { class: 'muted', style: 'font-size: 11px;', text:
              `${s.n_files} file(s) · ${s.n_runs} runs · ${s.n_perturbations} perturbations · ${s.n_findings} findings` }),
          ]),
        ]));
      });
      systemsHost.appendChild(sList);

      findingsCard.classList.toggle('hidden', data.findings.length === 0);
      findingsHost.innerHTML = '';
      const fList = el('div', { class: 'trace-list' });
      data.findings.slice(0, 50).forEach(f => {
        fList.appendChild(el('div', {
          class: 'trace-step', style: 'grid-template-columns: 1fr auto; cursor: pointer;',
          onclick: () => _openFromOverview(f.source_path),
        }, [
          el('div', { class: 'trace-step-body' }, [
            el('div', { class: 'trace-step-node' }, [
              el('span', { class: 'pill warning', style: 'margin-right: 6px;', text: f.failure_type }),
              f.where || '',
            ]),
            el('div', { class: 'trace-step-summary', text: f.summary || '(no summary)' }),
          ]),
          el('div', { class: 'muted', style: 'font-size: 11px; align-self: center;', text:
            `${f.source_group} · ${fmt.short(new Date(f.modified_ts * 1000).toISOString())}` }),
        ]));
      });
      findingsHost.appendChild(fList);
    } catch (e) {
      totalsHost.classList.add('empty');
      totalsHost.textContent = 'Failed to load overview: ' + e.message;
    }
  }

  // ============================================================
  // RESULTS BROWSER tab
  // ============================================================

  let _resultsWired = false;
  async function initResults() {
    if (_resultsWired) return;
    _resultsWired = true;
    $('#results-close').addEventListener('click', () => {
      $('#results-viewer').classList.add('hidden');
    });
    $('#results-filter').addEventListener('input', applyResultsFilter);
    await refreshResultsIndex();
  }

  // Client-side filter over the already-rendered index — cheap (a few
  // hundred DOM nodes at most) and avoids a round-trip for something this
  // local. Matches group name or file path, case-insensitive substring.
  // Hides a whole group section when every row in it is filtered out, and
  // keeps each visible group's header count honest (visible, not total).
  function applyResultsFilter() {
    const q = ($('#results-filter').value || '').trim().toLowerCase();
    const sections = $$('#results-index > [data-group]');
    let anyVisible = false;
    sections.forEach(section => {
      const group = section.dataset.group;
      const rows = $$('.trace-step', section);
      let visibleInGroup = 0;
      rows.forEach(row => {
        const match = !q || group.toLowerCase().includes(q) || (row.dataset.path || '').toLowerCase().includes(q);
        row.classList.toggle('hidden', !match);
        if (match) visibleInGroup++;
      });
      section.classList.toggle('hidden', visibleInGroup === 0);
      if (visibleInGroup > 0) anyVisible = true;
      const heading = $('h3', section);
      if (heading) {
        heading.textContent = `${group} (${visibleInGroup}${visibleInGroup !== rows.length ? ` of ${rows.length}` : ''})`;
      }
    });
    const noMatches = $('#results-no-matches');
    if (noMatches) noMatches.classList.toggle('hidden', anyVisible || sections.length === 0);
  }

  // Two-click delete, no native confirm() dialog: first click arms the
  // button (relabels it, turns it danger-colored); a second click within
  // ARM_TIMEOUT_MS actually deletes. Clicking elsewhere / letting it time
  // out disarms back to the normal label — cheap, dependency-free undo
  // window for a destructive action with no server-side undo.
  const ARM_TIMEOUT_MS = 4000;
  function _makeDeleteButton(path, onDeleted) {
    let armed = false;
    let timer = null;
    const btn = el('button', { class: 'ghost-btn sm', text: 'Delete' });
    btn.addEventListener('click', async () => {
      if (!armed) {
        armed = true;
        btn.textContent = 'Confirm delete?';
        btn.classList.add('danger');
        timer = setTimeout(() => {
          armed = false;
          btn.textContent = 'Delete';
          btn.classList.remove('danger');
        }, ARM_TIMEOUT_MS);
        return;
      }
      clearTimeout(timer);
      btn.disabled = true;
      btn.textContent = 'Deleting…';
      try {
        await api('/api/results/' + path, { method: 'DELETE' });
        toast('Deleted ' + path);
        onDeleted();
      } catch (e) {
        toast('Delete failed: ' + e.message, 'error');
        btn.disabled = false;
        armed = false;
        btn.textContent = 'Delete';
        btn.classList.remove('danger');
      }
    });
    return btn;
  }

  async function refreshResultsIndex() {
    const host = $('#results-index');
    const filterField = $('#results-filter-field');
    host.classList.remove('empty');
    host.innerHTML = 'Loading…';
    try {
      const groups = await fetchResultsIndex(true);
      host.innerHTML = '';
      const groupNames = Object.keys(groups).sort();
      if (!groupNames.length) {
        host.classList.add('empty');
        host.textContent = 'No saved results yet. Run something from the Test tab, or an example script with --save-json.';
        filterField.classList.add('hidden');
        return;
      }
      filterField.classList.remove('hidden');
      groupNames.forEach(group => {
        const entries = groups[group];
        const section = el('div', { style: 'margin-bottom: 16px;', 'data-group': group }, [
          el('h3', { style: 'margin-bottom: 6px; font-size: 14px;', text: group + ` (${entries.length})` }),
        ]);
        const list = el('div', { class: 'trace-list' });
        entries.forEach(entry => {
          const row = el('div', {
            class: 'trace-step',
            style: 'grid-template-columns: 1fr auto; cursor: default;',
            'data-path': entry.path,
          }, [
            el('div', {
              class: 'trace-step-body', style: 'cursor: pointer;',
              onclick: () => openResultsFile(entry.path),
            }, [
              el('div', { class: 'trace-step-node', text: entry.name }),
              el('div', { class: 'trace-step-summary', text: entry.path }),
            ]),
            el('div', { style: 'display: flex; align-items: center; gap: 10px;' }, [
              el('span', { class: 'muted', style: 'font-size: 11px;', text:
                `${(entry.size_bytes / 1024).toFixed(1)} KB · ${fmt.short(new Date(entry.modified_ts * 1000).toISOString())}` }),
              el('button', { class: 'ghost-btn sm', text: 'Evidence pack', onclick: () => openReportModal(entry.path) }),
              _makeDeleteButton(entry.path, () => refreshResultsIndex()),
            ]),
          ]);
          list.appendChild(row);
        });
        section.appendChild(list);
        host.appendChild(section);
      });
      host.appendChild(el('div', { id: 'results-no-matches', class: 'empty hidden', text: 'No saved runs match that filter.' }));
      applyResultsFilter();
    } catch (e) {
      host.classList.add('empty');
      host.textContent = 'Failed to load results: ' + e.message;
    }
  }

  async function openResultsFile(relpath) {
    const viewer = $('#results-viewer');
    const title = $('#results-viewer-title');
    const body = $('#results-viewer-body');
    title.textContent = relpath;
    body.textContent = 'Loading…';
    viewer.classList.remove('hidden');
    viewer.scrollIntoView({ behavior: 'smooth', block: 'start' });
    try {
      const data = await api('/api/results/' + relpath);
      $('#results-download').onclick = () => downloadJson(data, relpath.replace(/[\\/]/g, '_'));
      body.innerHTML = '';
      body.appendChild(renderResultsViewerBody(data, relpath));
    } catch (e) {
      body.textContent = 'Failed to load: ' + e.message;
    }
  }

  function renderResultsViewerBody(data, relpath) {
    if (data && data.baseline && data.perturbations) {
      return _renderAdapterShaped(data);
    }
    if (data && (data.per_question || data.per_shape || data.per_fixture)) {
      return _renderSweepShaped(data);
    }
    return el('pre', {
      class: 'mono',
      style: 'max-height: 600px; overflow: auto; font-size: 11px;',
      text: JSON.stringify(data, null, 2),
    });
  }

  function _renderAdapterShaped(data) {
    const host = el('div');
    const kvs = [
      ['question / topic', data.question || data.graph_name || '(unknown)'],
      ['intensity', data.intensity || '?'],
      ['perturbations', (data.perturbations || []).length],
      ['crashed', data.n_crashed ?? 0],
      ['diverged', data.n_diverged ?? 0],
      ['unchanged', data.n_unchanged ?? 0],
      ['judge findings', data.n_judge_findings ?? 0],
      ['coord findings', data.n_coordination_findings ?? 0],
    ];
    const kv = el('div', { class: 'kv-grid' });
    kvs.forEach(([k, v]) => {
      kv.appendChild(el('span', { class: 'k', text: k }));
      kv.appendChild(el('span', { class: 'v', text: String(v) }));
    });
    host.appendChild(kv);
    host.appendChild(el('details', { style: 'margin-top: 12px;' }, [
      el('summary', { class: 'help', text: 'Full JSON' }),
      el('pre', { class: 'mono', style: 'max-height: 500px; overflow: auto; font-size: 11px;',
                 text: JSON.stringify(data, null, 2) }),
    ]));
    return host;
  }

  function _renderSweepShaped(data) {
    const host = el('div');
    const agg = data.aggregate || {};
    if (Object.keys(agg).length) {
      const kv = el('div', { class: 'kv-grid' });
      Object.entries(agg).forEach(([k, v]) => {
        if (typeof v === 'object' && v !== null) return;
        kv.appendChild(el('span', { class: 'k', text: k }));
        kv.appendChild(el('span', { class: 'v', text: String(v) }));
      });
      host.appendChild(kv);
    }
    const arr = data.per_question || data.per_shape || data.per_fixture || [];
    if (arr.length) {
      host.appendChild(el('h4', { style: 'margin-top: 14px;', text: `Per-case (${arr.length}):` }));
      arr.forEach((row, i) => {
        const label = row.question || row.shape || row.fixture || row.category || `case ${i+1}`;
        host.appendChild(el('details', { style: 'margin-top: 4px;' }, [
          el('summary', { text: label.length > 80 ? label.slice(0, 77) + '…' : label }),
          el('pre', { class: 'mono', style: 'max-height: 240px; overflow: auto; font-size: 11px;',
                     text: JSON.stringify(row, null, 2) }),
        ]));
      });
    }
    host.appendChild(el('details', { style: 'margin-top: 12px;' }, [
      el('summary', { class: 'help', text: 'Full JSON' }),
      el('pre', { class: 'mono', style: 'max-height: 500px; overflow: auto; font-size: 11px;',
                 text: JSON.stringify(data, null, 2) }),
    ]));
    return host;
  }

  // ---------- start -------------------------------------------------------

  initAdapter();
})();
