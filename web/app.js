// drift — frontend logic. No build step, no dependencies.
//
// Tabs after the 2026-06-29 cleanup: Adapter (run drift_test against a
// bundled graph), Results (browse saved experiment JSON), Custom (paused
// while we re-wire @drift.agent to the adapter). Native sim, Runs, Detail,
// and Compare were removed along with the simulator runtime.

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
    if (name === 'adapter') initAdapter();
    if (name === 'results') initResults();
  }
  $$('.tab').forEach(t => t.addEventListener('click', () => activateTab(t.dataset.tab)));

  // ---------- adapter tab ------------------------------------------------

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
      intensity: 'light',
      max_perturbations: 4,
      judge: 'off',
      divergence_mode: 'exact',
      baseline_rollouts: 1,
      max_judge_calls: 0,
    },
    balanced: {
      intensity: 'aggressive',
      max_perturbations: 6,
      judge: 'openai',
      divergence_mode: 'tiered',
      baseline_rollouts: 3,
      max_judge_calls: 10,
    },
    thorough: {
      intensity: 'aggressive',
      max_perturbations: 12,
      judge: 'openai',
      divergence_mode: 'tiered',
      baseline_rollouts: 5,
      max_judge_calls: 25,
    },
    exhaustive: {
      intensity: 'exhaustive',
      max_perturbations: 100,
      judge: 'openai',
      divergence_mode: 'tiered',
      baseline_rollouts: 5,
      max_judge_calls: 50,
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
  let _lastAdapterRun = null;

  async function initAdapter() {
    if (_adapterWired) return;
    _adapterWired = true;
    $('#adapter-run').addEventListener('click', runAdapterDemo);
    $('#adapter-graph').addEventListener('change', onAdapterGraphChange);
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

    btn.disabled = true;
    const judgeOn = judge !== 'off' || divergence_mode === 'tiered';
    status.textContent = judgeOn ? 'running (judge on, may take a minute)…' : 'running…';
    try {
      const data = await api('/api/adapter-demo', {
        method: 'POST',
        body: {
          graph_name, state_overrides,
          intensity, seed, max_perturbations,
          auto_chaos_exclude, judge, judge_model, user_guidelines,
          divergence_mode, baseline_rollouts,
          max_judge_calls: Number.isFinite(max_judge_calls) ? max_judge_calls : 10,
          similarity_threshold: Number.isFinite(similarity_threshold) ? similarity_threshold : 0.85,
        },
      });
      renderAdapterResult(data);
      $('#adapter-result').classList.remove('hidden');
      $('#adapter-result').scrollIntoView({ behavior: 'smooth', block: 'start' });
      const parts = [`${data.perturbations.length} perturbations`];
      if (data.n_judge_findings) parts.push(`${data.n_judge_findings} judge finding(s)`);
      if (data.n_coordination_findings) parts.push(`${data.n_coordination_findings} coord finding(s)`);
      if (data.judge_calls_used) parts.push(`${data.judge_calls_used}/${data.judge_calls_budget} tier-3 calls`);
      status.textContent = `done — ${parts.join(' · ')}`;
    } catch (e) {
      toast('Adapter run failed: ' + e.message, 'error');
      status.textContent = '';
    } finally {
      btn.disabled = false;
    }
  }

  // ---- trace rendering helpers ------------------------------------------

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

  function renderAdapterResult(data) {
    _lastAdapterRun = data;
    const dlBtn = $('#adapter-download-json');
    if (dlBtn) {
      dlBtn.onclick = () => downloadJson(
        data,
        `adapter_${data.graph_name}_${data.seed}_${Date.now()}.json`,
      );
    }
    // --- summary kv ---
    const summary = $('#adapter-summary');
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

    // --- baseline trace ---
    const base = $('#adapter-baseline-trace');
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
                                   text: `Trace — ${(b.trace || []).length} super-step(s):` }));
      base.appendChild(renderTraceList(b.trace));

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

    // --- perturbations ---
    const host = $('#adapter-perturbations');
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
      el('span', { class: 'muted', style: 'margin-left: auto; font-size: 11px;', text: '▾ click to expand' }),
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
      what: 'Orchestrator spawned more subagents than the task warranted — either too many distinct subagents (hard count) or too many subagents per measurable output (ratio rule).',
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

  // ---------- results browser tab ----------------------------------------

  let _resultsWired = false;
  async function initResults() {
    if (_resultsWired) return;
    _resultsWired = true;
    $('#results-close').addEventListener('click', () => {
      $('#results-viewer').classList.add('hidden');
    });
    await refreshResultsIndex();
  }

  async function refreshResultsIndex() {
    const host = $('#results-index');
    host.classList.remove('empty');
    host.innerHTML = 'Loading…';
    try {
      const data = await api('/api/results');
      const groups = data.groups || {};
      host.innerHTML = '';
      const groupNames = Object.keys(groups).sort();
      if (!groupNames.length) {
        host.classList.add('empty');
        host.textContent = 'No saved results yet. Run an example script with --save-json to populate this list.';
        return;
      }
      groupNames.forEach(group => {
        const entries = groups[group];
        const section = el('div', { style: 'margin-bottom: 16px;' }, [
          el('h3', { style: 'margin-bottom: 6px; font-size: 14px;', text: group + ` (${entries.length})` }),
        ]);
        const list = el('div', { class: 'trace-list' });
        entries.forEach(entry => {
          const row = el('div', {
            class: 'trace-step',
            style: 'cursor: pointer; grid-template-columns: 1fr auto;',
            onclick: () => openResultsFile(entry.path),
          }, [
            el('div', { class: 'trace-step-body' }, [
              el('div', { class: 'trace-step-node', text: entry.name }),
              el('div', { class: 'trace-step-summary', text: entry.path }),
            ]),
            el('div', { class: 'muted', style: 'text-align: right; font-size: 11px;', text:
              `${(entry.size_bytes / 1024).toFixed(1)} KB · ${fmt.short(new Date(entry.modified_ts * 1000).toISOString())}` }),
          ]);
          list.appendChild(row);
        });
        section.appendChild(list);
        host.appendChild(section);
      });
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
