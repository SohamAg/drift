// drift — frontend logic. No build step, no dependencies.
//
// Layout: tabs (New Run, Runs, Compare, About) and a hidden "detail" panel
// that takes over when a run row is clicked. State is kept in the DOM and
// in a tiny module-level cache; we re-fetch from /api/* whenever it matters.

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
    pct(x) { return (x * 100).toFixed(1) + '%'; },
    num(x) { return Number(x).toLocaleString(); },
    truncNum(n, p = 3) {
      if (typeof n !== 'number') return n;
      return n.toFixed(p).replace(/\.?0+$/, '');
    },
  };

  // Severity classification — drives color coding for cells, ticker pills, etc.
  // 'critical' = the system-broke kind. 'warning' = drift / silent issues.
  // 'info' = artifacts of intra-step coordination, technically defects but lower stakes.
  const FAILURE_SEVERITY = {
    contradictory_refund:     'critical',
    contradictory_review:     'critical',
    contradictory_diagnosis:  'critical',
    security_bypass:          'critical',
    merge_without_approval:   'critical',
    sentiment_collapse:       'critical',
    escalation_loop:          'warning',
    queue_explosion:          'warning',
    silent_remediation:       'warning',
    comms_lag:                'warning',
    hallucinated_reference:   'warning',
    policy_inconsistency:     'info',
    stale_snapshot_reference: 'info',
  };
  const sev = (t) => FAILURE_SEVERITY[t] || 'info';

  // Icon-free dot prefix for topology in pills.
  const topoDot = (name) =>
    el('span', { class: `topo-dot ${name || ''}`, title: name || '' });

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
    if (name === 'detect') initDetect();
    if (name === 'adapter') initAdapter();
    if (name === 'runs') refreshRuns();
    if (name === 'compare') populateComparePickers();
    if (name === 'analyze') initAnalyze();
    if (name === 'custom') initCustom();
  }
  $$('.tab').forEach(t => t.addEventListener('click', () => activateTab(t.dataset.tab)));

  // The detail view is a 5th panel that's not a tab; switching to it hides
  // the active tab's panel.
  function showDetail() {
    $$('.tab-panel').forEach(p => p.classList.remove('active'));
    $('#tab-detail').classList.add('active');
  }

  // ---------- bootstrap ---------------------------------------------------

  let TOPOLOGIES = [];
  let SCENARIOS = [];
  let LAST_CONFIG = null;

  async function bootstrap() {
    try {
      [TOPOLOGIES, SCENARIOS] = await Promise.all([
        api('/api/topologies'),
        api('/api/scenarios'),
      ]);
      populateTopologyDropdown();
      populateAboutTopologies();
    } catch (e) {
      toast('Could not load topologies/scenarios: ' + e.message, 'error');
    }
    await refreshRuns();
    // Detect tab is the default landing — load its cards immediately so
    // visitors see content without having to click anything.
    initDetect();
  }

  function populateTopologyDropdown() {
    const sel = $('#topology');
    sel.innerHTML = '';
    TOPOLOGIES.forEach(t => sel.appendChild(el('option', { value: t.name, text: t.name })));
    sel.addEventListener('change', onTopologyChange);
    onTopologyChange();
  }

  function onTopologyChange() {
    const t = TOPOLOGIES.find(x => x.name === $('#topology').value);
    if (!t) return;
    $('#topology-help').textContent = t.description;
    populateScenarioDropdown(t);
  }

  function populateScenarioDropdown(topology) {
    const sel = $('#scenario');
    sel.innerHTML = '';
    sel.appendChild(el('option', { value: '', text: '(empty — stochastic only)' }));
    // Filter scenarios by whether their referenced events are in the topology's registry.
    const supported = SCENARIOS.filter(s =>
      s.events_used.length === 0 || s.events_used.every(ev => topology.events.includes(ev))
    );
    supported.forEach(s => sel.appendChild(el('option', {
      value: s.filename,
      text: `${s.name}  —  ${s.scripted_count} scripted, ${s.stochastic_count} stochastic`,
    })));
    sel.addEventListener('change', onScenarioChange);
    onScenarioChange();
  }

  function onScenarioChange() {
    const filename = $('#scenario').value;
    if (!filename) {
      $('#scenario-help').textContent = 'No scripted events; only stochastic injection (none if topology has no defaults).';
      return;
    }
    const s = SCENARIOS.find(x => x.filename === filename);
    if (s) {
      $('#scenario-help').textContent =
        `${s.scripted_count} scripted events at fixed timesteps; ${s.stochastic_count} stochastic entries.`;
    }
  }

  function populateAboutTopologies() {
    const host = $('#about-topologies');
    host.innerHTML = '';
    TOPOLOGIES.forEach(t => {
      const card = el('div', { class: `topo-card ${t.name}` }, [
        el('h4', {}, [topoDot(t.name), t.name]),
        el('p',  { text: t.description }),
        el('div', { style: 'margin-bottom: 10px;' }, t.roles.map(r =>
          el('span', { class: 'pill', text: r })
        )),
        el('div', { class: 'muted mono',
                    text: `${t.detectors.length} detectors · ${t.events.length} events` }),
      ]);
      host.appendChild(card);
    });
  }

  // ---------- new run form ------------------------------------------------

  $('#new-run-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const form = e.target;
    const variant = form.querySelector('input[name="prompt_variant"]:checked').value;
    const body = {
      topology:        $('#topology').value,
      scenario:        $('#scenario').value || null,
      steps:           parseInt($('#steps').value, 10),
      seed:            parseInt($('#seed').value, 10),
      llm:             $('#llm').value,
      model:           $('#model').value || null,
      prompt_variant:  variant,
      run_id:          $('#run-id').value || null,
    };
    LAST_CONFIG = body;

    const submitBtn = form.querySelector('button[type=submit]');
    submitBtn.disabled = true;
    submitBtn.textContent = 'Starting…';

    try {
      const res = await api('/api/runs', { method: 'POST', body });
      toast(`Run started: ${res.run_id}`, 'success');
      pollRunStatus(res.run_id);
    } catch (e) {
      toast('Failed to start: ' + e.message, 'error');
    } finally {
      submitBtn.disabled = false;
      submitBtn.textContent = 'Start run';
    }
  });

  $('#clone-last').addEventListener('click', () => {
    if (!LAST_CONFIG) { toast('No previous run config in this session', 'info'); return; }
    $('#topology').value = LAST_CONFIG.topology;
    onTopologyChange();
    setTimeout(() => {
      $('#scenario').value = LAST_CONFIG.scenario || '';
      onScenarioChange();
    }, 0);
    $('#steps').value = LAST_CONFIG.steps;
    $('#seed').value  = LAST_CONFIG.seed;
    $('#llm').value   = LAST_CONFIG.llm;
    $('#model').value = LAST_CONFIG.model || '';
    document.querySelector(`input[name="prompt_variant"][value="${LAST_CONFIG.prompt_variant}"]`).checked = true;
    $('#run-id').value = '';
    toast('Cloned config — adjust and submit', 'info');
  });

  // ---------- live status polling -----------------------------------------

  let activePoll = null;

  // Persistent state across polls so we can animate transitions.
  const livePrev = { ftypeCounts: {}, eventIds: new Set() };

  function pollRunStatus(runId) {
    if (activePoll) clearInterval(activePoll);
    const host = $('#live-status');
    host.classList.remove('empty');

    // Reset transition state for the new run.
    livePrev.ftypeCounts = {};
    livePrev.eventIds = new Set();

    // Render scaffold once, then update fields by id on each poll. Avoids
    // tearing down DOM (and animations) every tick.
    host.innerHTML = '';
    host.appendChild(el('div', { class: 'live-panel' }, [
      el('div', { class: 'live-header' }, [
        el('div', { class: 'left' }, [
          el('span', { id: 'live-status-pill', class: 'pill status-queued', text: 'queued' }),
          el('span', { id: 'live-topo' }),
          el('span', { id: 'live-runid', class: 'run-id' }),
        ]),
        el('div', { id: 'live-step', class: 'step-counter', text: '0 / 0' }),
      ]),
      el('div', { id: 'live-progress', class: 'progress' }, el('div', { class: 'progress-bar' })),
      el('div', { class: 'card tight' }, [
        el('h3', { text: 'Live world state' }),
        el('div', { id: 'live-world', class: 'world-grid' }),
      ]),
      el('div', { class: 'card tight' }, [
        el('h3', { text: 'Failures detected' }),
        el('div', { id: 'live-failures', class: 'failure-ticker empty', text: 'none yet' }),
      ]),
      el('div', { class: 'card tight' }, [
        el('h3', { text: 'Recent events (most recent first)' }),
        el('div', { id: 'live-events', class: 'event-tape empty', text: 'no events yet' }),
      ]),
      el('div', { class: 'card tight' }, [
        el('h3', { text: 'Latest agent actions' }),
        el('div', { id: 'live-actions', class: 'action-chips empty', text: 'no actions yet' }),
      ]),
      el('div', { id: 'live-finish' }),
    ]));

    paint({
      run_id: runId, status: 'queued', completed_steps: 0, total_steps: 0,
      failure_count: 0, started_at: new Date().toISOString(),
      world_state: {}, failures_by_type: {}, recent_events: [], recent_failures: [], recent_actions: [],
    });

    activePoll = setInterval(async () => {
      try {
        const s = await api(`/api/runs/${encodeURIComponent(runId)}/status`);
        paint(s);
        if (s.status === 'done' || s.status === 'failed') {
          clearInterval(activePoll); activePoll = null;
          refreshRuns();
        }
      } catch (e) {
        clearInterval(activePoll); activePoll = null;
        toast('Polling failed: ' + e.message, 'error');
      }
    }, 700);

    function paint(s) {
      const isDone = s.status === 'done' || s.status === 'failed';
      const pct = s.total_steps > 0 ? Math.min(100, (s.completed_steps / s.total_steps) * 100) : 0;

      // header pieces
      const pill = $('#live-status-pill');
      pill.className = `pill status-${s.status}`;
      pill.textContent = s.status;
      const runIdEl = $('#live-runid');
      runIdEl.textContent = s.run_id;
      const topoEl = $('#live-topo');
      topoEl.innerHTML = '';
      if (s.topology) {
        topoEl.appendChild(el('span', { class: `pill topo-${s.topology}` }, [
          topoDot(s.topology),
          s.topology,
        ]));
      }
      $('#live-step').textContent = `${s.completed_steps} / ${s.total_steps}`;

      const progressEl = $('#live-progress');
      progressEl.classList.toggle('idle', isDone || s.status === 'queued');
      progressEl.firstElementChild.style.width = pct + '%';

      paintLiveWorld(s.world_state, s.topology);
      paintLiveFailures(s.failures_by_type || {});
      paintLiveEvents(s.recent_events || []);
      paintLiveActions(s.recent_actions || []);

      const finish = $('#live-finish');
      finish.innerHTML = '';
      if (s.error) {
        finish.appendChild(el('div', { class: 'pill danger', text: s.error }));
      }
      if (isDone) {
        finish.appendChild(el('div', { class: 'actions' }, [
          el('button', {
            class: 'primary',
            text: 'View full result →',
            onclick: () => openRunDetail(s.run_id),
          }),
          el('span', { class: 'muted', text: `Finished ${fmt.short(s.finished_at)}` }),
        ]));
      }
    }
  }

  // Live world bars. Picks the metrics most relevant per topology
  // and color-codes the bar based on whether the value is "danger" range.
  function paintLiveWorld(world, topology) {
    const host = $('#live-world');
    host.innerHTML = '';
    if (!world || !Object.keys(world).length) {
      host.appendChild(el('div', { class: 'empty', text: 'waiting for first step…' }));
      return;
    }

    const cell = (label, value, fill, severity) => {
      const c = el('div', { class: `world-cell${severity ? ' ' + severity : ''}` });
      c.appendChild(el('div', { class: 'label', text: label }));
      c.appendChild(el('div', { class: 'value', text: value }));
      if (fill !== undefined) {
        const bar = el('div', { class: 'bar' });
        bar.appendChild(el('span', { style: `width:${Math.max(0, Math.min(100, fill * 100))}%` }));
        c.appendChild(bar);
      }
      return c;
    };

    const sentiment = world.customer_sentiment;
    if (typeof sentiment === 'number') {
      const sevCls = sentiment < 0.25 ? 'crit' : sentiment < 0.5 ? 'warn' : 'ok';
      const label = topology === 'ops' ? 'public trust' : topology === 'code_review' ? 'team morale' : 'customer sentiment';
      host.appendChild(cell(label, sentiment.toFixed(2), sentiment, sevCls));
    }
    if (typeof world.system_load === 'number') {
      const sevCls = world.system_load > 0.85 ? 'crit' : world.system_load > 0.6 ? 'warn' : 'ok';
      host.appendChild(cell('system load', world.system_load.toFixed(2), world.system_load, sevCls));
    }
    if (typeof world.refund_policy_version === 'number' && topology === 'support') {
      host.appendChild(cell('policy version', `v${world.refund_policy_version}`));
    }
    if (typeof world.deadline_pressure === 'number') {
      const sevCls = world.deadline_pressure > 0.7 ? 'crit' : world.deadline_pressure > 0.4 ? 'warn' : 'ok';
      host.appendChild(cell('deadline pressure', world.deadline_pressure.toFixed(2), world.deadline_pressure, sevCls));
    }
    if (typeof world.inventory_delay_minutes === 'number' && world.inventory_delay_minutes > 0) {
      host.appendChild(cell('inventory delay', `${world.inventory_delay_minutes}m`));
    }
    const openCases = world.open_cases ? Object.keys(world.open_cases).length : 0;
    const openLabel = topology === 'code_review' ? 'open PRs' : topology === 'ops' ? 'open incidents' : 'open cases';
    host.appendChild(cell(openLabel, fmt.num(openCases)));
    const queued = (world.escalation_queue || []).length;
    if (queued > 0 || topology === 'support') {
      host.appendChild(cell('queue depth', fmt.num(queued)));
    }
    host.appendChild(cell('timestep', `t=${world.timestep ?? 0}`));
  }

  function paintLiveFailures(byType) {
    const host = $('#live-failures');
    host.classList.toggle('empty', !Object.keys(byType).length);
    if (!Object.keys(byType).length) {
      host.textContent = 'none yet';
      return;
    }
    host.textContent = '';
    Object.entries(byType).sort((a, b) => b[1] - a[1]).forEach(([t, c]) => {
      const prev = livePrev.ftypeCounts[t] || 0;
      const bumped = c > prev;
      const pill = el('span', { class: `ticker-pill sev-${sev(t)}${bumped ? ' bump' : ''}` }, [
        t,
        el('span', { class: 'count', text: c }),
      ]);
      host.appendChild(pill);
    });
    livePrev.ftypeCounts = { ...byType };
  }

  function paintLiveEvents(events) {
    const host = $('#live-events');
    host.classList.toggle('empty', !events.length);
    if (!events.length) { host.textContent = 'no events yet'; return; }
    host.textContent = '';
    // Newest first, top of list.
    [...events].reverse().forEach(e => {
      const isNew = !livePrev.eventIds.has(e.event_id);
      const row = el('div', { class: 'tape-row', style: isNew ? '' : 'animation: none' }, [
        el('div', { class: 'step', text: `t=${e.timestep}` }),
        el('div', {}, [
          el('div', { class: 'name', text: e.name }),
          el('div', { class: 'summary', text: e.summary }),
        ]),
      ]);
      host.appendChild(row);
      livePrev.eventIds.add(e.event_id);
    });
  }

  function paintLiveActions(actions) {
    const host = $('#live-actions');
    host.classList.toggle('empty', !actions.length);
    if (!actions.length) { host.textContent = 'no actions yet'; return; }
    host.textContent = '';
    actions.forEach(a => {
      const target = a.target_case_id ? ` ${a.target_case_id}` : '';
      host.appendChild(el('span', { class: 'action-chip' }, [
        el('strong', { text: a.agent_name }),
        ` · ${a.kind}${target}`,
      ]));
    });
  }

  // ---------- runs list ---------------------------------------------------

  let RUNS = [];

  async function refreshRuns() {
    try {
      RUNS = await api('/api/runs');
    } catch (e) {
      toast('Could not load runs: ' + e.message, 'error');
      RUNS = [];
    }
    $('#runs-count').textContent = RUNS.length || '';
    paintRuns();
  }

  function paintRuns() {
    const tbody = $('#runs-tbody');
    tbody.innerHTML = '';
    const q = $('#runs-search').value.trim().toLowerCase();
    const filtered = !q ? RUNS : RUNS.filter(r =>
      [r.run_id, r.topology, r.scenario, r.llm, r.prompt_variant].some(x =>
        (x || '').toString().toLowerCase().includes(q)));
    $('#runs-empty').classList.toggle('hidden', filtered.length > 0);

    // Re-order so each child appears immediately after its parent. Children
    // whose parent isn't in the visible filtered set stay where they were.
    const byId = new Map(filtered.map(r => [r.run_id, r]));
    const placed = new Set();
    const ordered = [];
    filtered.forEach(r => {
      if (placed.has(r.run_id)) return;
      // Walk up to the top-most visible parent so we anchor the whole chain.
      let root = r;
      while (root.parent_run_id && byId.has(root.parent_run_id)) {
        root = byId.get(root.parent_run_id);
      }
      // Emit root + a DFS over its descendants.
      const stack = [root];
      while (stack.length) {
        const n = stack.shift();
        if (placed.has(n.run_id)) continue;
        ordered.push(n);
        placed.add(n.run_id);
        const children = filtered.filter(x => x.parent_run_id === n.run_id);
        // Process children right after this node, preserving their listing order.
        stack.unshift(...children);
      }
    });

    ordered.forEach(r => {
      const failPillClass =
        r.n_failures === 0 ? 'success' :
        r.n_failures < 10 ? 'info' :
        r.n_failures < 50 ? 'warning' : 'critical';
      const isChild = !!r.parent_run_id && byId.has(r.parent_run_id);
      const tr = el('tr', {
        class: 'table-row-link' + (isChild ? ' is-child' : ''),
        onclick: () => openRunDetail(r.run_id),
      }, [
        el('td', { class: 'mono nowrap' }, [
          r.parent_run_id
            ? el('span', { class: 'fork-indicator', title: `forked from ${r.parent_run_id} at t=${r.branch_at_step}` }, [
                el('span', { class: 'branch-glyph', text: '⑂' }),
              ])
            : null,
          r.run_id,
        ]),
        el('td', {}, r.topology
          ? el('span', { class: `pill topo-${r.topology}` }, [topoDot(r.topology), r.topology])
          : '—'),
        el('td', { class: 'mono', text: r.scenario || '—' }),
        el('td', { class: 'num', text: r.final_step }),
        el('td', { class: 'num mono', text: (r.seed ?? '—') }),
        el('td', {}, r.llm ? el('span', { class: 'pill', text: r.llm }) : '—'),
        el('td', {}, r.prompt_variant ? el('span', { class: 'pill', text: r.prompt_variant }) : '—'),
        el('td', { class: 'num' }, [
          el('span', { class: `pill ${failPillClass}`, text: r.n_failures }),
        ]),
        el('td', { class: 'nowrap muted', text: fmt.short(r.started_at) }),
        el('td', {}, el('a', { href: '#', text: 'View →', onclick: (ev) => { ev.preventDefault(); ev.stopPropagation(); openRunDetail(r.run_id); } })),
      ]);
      tbody.appendChild(tr);
    });
  }
  $('#runs-search').addEventListener('input', paintRuns);
  $('#runs-refresh').addEventListener('click', refreshRuns);

  // ---------- run detail --------------------------------------------------

  $('#detail-back').addEventListener('click', () => activateTab('runs'));

  // Fork button + modal wiring is set up once. The modal pulls context from
  // the currently-open run detail when it opens.
  let CURRENT_DETAIL = null;  // last loaded run-detail data
  $('#detail-fork-btn').addEventListener('click', () => {
    if (!CURRENT_DETAIL) { toast('Open a run first', 'error'); return; }
    openForkModal(CURRENT_DETAIL);
  });
  $('#detail-compare-parent-btn').addEventListener('click', () => {
    const parent = CURRENT_DETAIL?.summary?.parent_run_id;
    const self = CURRENT_DETAIL?.summary?.run_id;
    if (!parent || !self) return;
    activateTab('compare');
    setTimeout(() => {
      $('#cmp-a').value = parent;
      $('#cmp-b').value = self;
      $('#cmp-go').click();
    }, 50);
  });
  $$('#fork-modal .modal-close').forEach(b => b.addEventListener('click', closeForkModal));
  $('#fork-modal').addEventListener('click', (ev) => {
    if (ev.target.id === 'fork-modal') closeForkModal();
  });
  $('#fork-at-slider').addEventListener('input', (ev) => {
    $('#fork-at').value = ev.target.value;
  });
  $('#fork-at').addEventListener('input', (ev) => {
    $('#fork-at-slider').value = ev.target.value;
  });
  $('#fork-submit').addEventListener('click', submitFork);

  async function openRunDetail(runId) {
    showDetail();
    $('#detail-title').textContent = runId;
    $('#detail-meta').innerHTML = '';
    $('#detail-lineage').classList.add('hidden');
    $('#detail-compare-parent-btn').classList.add('hidden');
    $('#detail-failure-summary').innerHTML = '<div class="empty">Loading…</div>';
    $('#detail-failure-list').innerHTML = '';
    $('#detail-events').innerHTML = '';
    $('#detail-agents').innerHTML = '';
    $('#detail-world').innerHTML = '';

    let data;
    try {
      data = await api(`/api/runs/${encodeURIComponent(runId)}`);
    } catch (e) {
      $('#detail-failure-summary').innerHTML = '';
      toast('Could not load run: ' + e.message, 'error');
      return;
    }

    CURRENT_DETAIL = data;
    paintDetailMeta(data.summary);
    paintLineage(data.summary);
    paintFailures(data.failures);
    paintTimeline(data.events);
    paintAgents(data.actions);
    paintWorld(data.snapshots);
  }

  function paintLineage(s) {
    const badge = $('#detail-lineage');
    const cmpBtn = $('#detail-compare-parent-btn');
    if (!s.parent_run_id) {
      badge.classList.add('hidden');
      cmpBtn.classList.add('hidden');
      return;
    }
    badge.innerHTML = '';
    badge.appendChild(el('span', { class: 'lineage-icon', text: '⑂' }));
    badge.appendChild(el('span', {}, [
      'Forked from ',
      el('a', {
        href: '#', class: 'mono',
        text: s.parent_run_id,
        onclick: (ev) => { ev.preventDefault(); openRunDetail(s.parent_run_id); },
      }),
      ' at ',
      el('span', { class: 'mono', text: `t=${s.branch_at_step}` }),
    ]));
    // Summarize the override knobs that were used.
    const o = s.fork_overrides || {};
    const overrideBits = [];
    if (o.seed != null) overrideBits.push(`seed=${o.seed}`);
    if (o.prompt_variants && Object.keys(o.prompt_variants).length) {
      overrideBits.push(Object.entries(o.prompt_variants).map(([r, v]) => `${r}:${v}`).join(', '));
    }
    if (o.disabled_agents && o.disabled_agents.length) {
      overrideBits.push('disabled ' + o.disabled_agents.join(','));
    }
    if (overrideBits.length) {
      badge.appendChild(el('span', { class: 'muted', text: ' · ' + overrideBits.join(' · ') }));
    }
    badge.classList.remove('hidden');
    cmpBtn.classList.remove('hidden');
  }

  function paintDetailMeta(s) {
    const meta = $('#detail-meta');
    meta.innerHTML = '';
    const kvs = [
      ['Topology',  s.topology],
      ['Scenario',  s.scenario],
      ['Seed',      s.seed],
      ['LLM',       s.llm],
      ['Variant',   s.prompt_variant],
      ['Steps',     `${s.final_step} / ${s.steps_requested ?? s.final_step}`],
      ['Failures',  s.n_failures],
      ['Actions',   s.n_actions],
      ['Started',   fmt.short(s.started_at)],
    ];
    kvs.forEach(([k, v]) => {
      meta.appendChild(el('span', { class: 'k', text: k }));
      meta.appendChild(el('span', { class: 'v', text: v == null || v === '' ? '—' : String(v) }));
    });
  }

  function paintFailures(failures) {
    const summary = $('#detail-failure-summary');
    const list = $('#detail-failure-list');
    summary.innerHTML = '';
    list.innerHTML = '';

    if (!failures.length) {
      summary.appendChild(el('div', { class: 'empty', text: 'No failures detected — clean run.' }));
      return;
    }

    const byType = {};
    failures.forEach(f => { byType[f.failure_type] = (byType[f.failure_type] || 0) + 1; });

    const sumGrid = el('div', { class: 'failure-summary' });
    Object.entries(byType).sort((a, b) => b[1] - a[1]).forEach(([type, count]) => {
      sumGrid.appendChild(el('div', { class: `failure-cell sev-${sev(type)}` }, [
        el('div', { class: 'ftype', text: type }),
        el('div', { class: 'fcount', text: fmt.num(count) }),
      ]));
    });
    summary.appendChild(sumGrid);

    failures.slice(0, 200).forEach(f => {
      list.appendChild(el('div', { class: 'failure-row' }, [
        el('div', { class: 'step', text: `t=${f.timestep}` }),
        el('div', {}, [
          el('div', {}, [
            el('span', { class: `pill ${sev(f.failure_type) === 'critical' ? 'critical' : sev(f.failure_type)}`, text: f.failure_type }),
            ' ',
            el('span', { class: 'muted', text: (f.agents_involved || []).join(', ') || '—' }),
          ]),
          el('div', { class: 'mono muted', html: escapeHtml(f.summary || '') }),
        ]),
      ]));
    });
    if (failures.length > 200) {
      list.appendChild(el('div', { class: 'muted', text: `… and ${failures.length - 200} more.` }));
    }
  }

  function paintTimeline(events) {
    const host = $('#detail-events');
    host.innerHTML = '';
    if (!events.length) {
      host.appendChild(el('div', { class: 'empty', text: 'No events fired in this run.' }));
      return;
    }
    events.forEach(e => {
      host.appendChild(el('div', { class: 'timeline-row' }, [
        el('div', { class: 'step', text: `t=${e.timestep}` }),
        el('div', { class: 'name', text: e.name }),
        el('div', { class: 'summary', text: e.summary }),
      ]));
    });
  }

  // Color palette for agent action segments — derived from accent + neutrals.
  const SEG_COLORS = [
    '#6366f1', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6',
    '#06b6d4', '#84cc16', '#ec4899', '#64748b', '#14b8a6',
  ];

  function paintAgents(actions) {
    const host = $('#detail-agents');
    host.innerHTML = '';
    if (!actions.length) {
      host.appendChild(el('div', { class: 'empty', text: 'No actions emitted.' }));
      return;
    }

    // tally per-agent x kind
    const byAgent = {};
    actions.forEach(a => {
      byAgent[a.agent_name] = byAgent[a.agent_name] || {};
      byAgent[a.agent_name][a.kind] = (byAgent[a.agent_name][a.kind] || 0) + 1;
    });

    // stable color per kind across all rows so the legend means something
    const kindColor = {};
    let nextColor = 0;
    const allKinds = new Set();
    Object.values(byAgent).forEach(kinds => Object.keys(kinds).forEach(k => allKinds.add(k)));
    [...allKinds].sort().forEach(k => { kindColor[k] = SEG_COLORS[nextColor++ % SEG_COLORS.length]; });

    Object.entries(byAgent).sort().forEach(([agent, kinds]) => {
      const total = Object.values(kinds).reduce((a, b) => a + b, 0);
      const stack = el('div', { class: 'bar-stack' });
      Object.entries(kinds).sort((a, b) => b[1] - a[1]).forEach(([k, c]) => {
        const w = (c / total) * 100;
        if (w < 1) return;
        stack.appendChild(el('div', {
          class: 'bar-seg',
          style: `flex-basis:${w}%; background:${kindColor[k]}`,
          title: `${k}: ${c} (${fmt.pct(c / total)})`,
          text: w >= 8 ? `${k} ${c}` : '',
        }));
      });
      host.appendChild(el('div', { class: 'agent-row' }, [
        el('div', {}, [
          el('div', { class: 'agent-name', text: agent }),
          el('div', { class: 'muted mono', text: `${total} actions` }),
        ]),
        el('div', {}, [stack, agentLegend(kinds, kindColor)]),
      ]));
    });
  }

  function agentLegend(kinds, kindColor) {
    const wrap = el('div', { class: 'bar-legend' });
    Object.entries(kinds).sort((a, b) => b[1] - a[1]).forEach(([k, c]) => {
      wrap.appendChild(el('span', {}, [
        el('span', { class: 'swatch', style: `background:${kindColor[k]}` }),
        `${k} · ${c}`,
      ]));
    });
    return wrap;
  }

  function paintWorld(snapshots) {
    const host = $('#detail-world');
    host.innerHTML = '';
    if (!snapshots.length) {
      host.appendChild(el('div', { class: 'empty', text: 'No snapshots.' }));
      return;
    }
    const final = snapshots[snapshots.length - 1];
    const known = ['timestep', 'customer_sentiment', 'refund_policy_version',
                   'inventory_delay_minutes', 'system_load', 'deadline_pressure'];
    known.forEach(k => {
      if (k in final) {
        host.appendChild(el('span', { class: 'k', text: k }));
        const v = typeof final[k] === 'number' ? final[k].toFixed(3).replace(/\.?0+$/, '') : final[k];
        host.appendChild(el('span', { class: 'v', text: v }));
      }
    });
    host.appendChild(el('span', { class: 'k', text: 'open_cases' }));
    host.appendChild(el('span', { class: 'v', text: Object.keys(final.open_cases || {}).length }));
    host.appendChild(el('span', { class: 'k', text: 'escalation_queue' }));
    host.appendChild(el('span', { class: 'v', text: (final.escalation_queue || []).length }));
  }

  // ---------- compare -----------------------------------------------------

  function populateComparePickers() {
    ['#cmp-a', '#cmp-b'].forEach(id => {
      const sel = $(id);
      const cur = sel.value;
      sel.innerHTML = '';
      sel.appendChild(el('option', { value: '', text: '— pick a run —' }));
      RUNS.forEach(r => {
        const label = `${r.run_id} (${r.topology || '?'} · ${r.n_failures} fail)`;
        sel.appendChild(el('option', { value: r.run_id, text: label }));
      });
      sel.value = cur;
    });
  }

  // Compare state — remembers the current pair and mode for the toggle.
  let CMP_STATE = { a: null, b: null, mode: 'auto', data: null };

  async function doCompare(mode) {
    const a = $('#cmp-a').value, b = $('#cmp-b').value;
    if (!a || !b) { toast('Pick two runs', 'error'); return; }
    if (a === b) { toast('Pick two different runs', 'error'); return; }
    let data;
    try {
      data = await api('/api/compare', { method: 'POST', body: { run_a: a, run_b: b, mode } });
    } catch (e) {
      toast('Compare failed: ' + e.message, 'error');
      return;
    }
    CMP_STATE = { a, b, mode, data };
    paintCompare(data);
  }

  $('#cmp-go').addEventListener('click', () => doCompare('auto'));

  function paintCompare(data) {
    $('#cmp-result').classList.remove('hidden');
    paintCompareRelationship(data);
    paintCompareFailures(data);
    paintCompareWorld(data);
    paintCompareTimeline(data);
    paintCompareAgents(data);
  }

  function paintCompareRelationship(data) {
    const host = $('#cmp-relationship');
    host.innerHTML = '';
    host.classList.remove('hidden');
    host.classList.add('relationship-card');

    let labelHtml;
    if (data.relationship === 'parent_child') {
      labelHtml = el('span', { class: 'label', html:
        `<strong>Parent–child relationship detected.</strong> ` +
        `Runs diverge at <code>t=${data.divergence_step}</code>. ` +
        `Comparing <em>only the divergent steps</em> by default — toggle to see whole-run totals.` });
    } else if (data.relationship === 'siblings') {
      labelHtml = el('span', { class: 'label', html:
        `<strong>Sibling forks detected.</strong> ` +
        `Both runs forked from the same parent at <code>t=${data.divergence_step}</code>. ` +
        `Comparing the divergent steps only.` });
    } else {
      labelHtml = el('span', { class: 'label', html:
        `<strong>Unrelated runs.</strong> No shared lineage; comparing totals across the whole runs.` });
    }

    const leftBlock = el('div', { class: 'left' }, [
      el('span', { class: 'icon', text: data.relationship === 'unrelated' ? '·' : '⑂' }),
      labelHtml,
    ]);

    const toggle = el('div', { class: 'mode-toggle' }, [
      el('div', {
        class: 'mode-opt' + (data.mode === 'post_branch' ? ' active' : ''),
        text: 'Post-branch only',
        onclick: () => { if (data.relationship !== 'unrelated') doCompare('post_branch'); },
      }),
      el('div', {
        class: 'mode-opt' + (data.mode === 'total' ? ' active' : ''),
        text: 'Whole runs',
        onclick: () => doCompare('total'),
      }),
    ]);

    host.appendChild(leftBlock);
    if (data.relationship !== 'unrelated') host.appendChild(toggle);
  }

  function paintCompareTimeline(data) {
    const host = $('#cmp-timeline');
    host.innerHTML = '';
    const tl = data.timeline;
    if (!tl || !tl.steps || !tl.steps.length) {
      host.appendChild(el('div', { class: 'empty', text: 'No timeline data.' }));
      return;
    }

    // Header
    host.appendChild(el('div', { class: 'dt-head' }, [
      el('div', { text: 't' }),
      el('div', { class: 'dt-side-a', text: `A · ${truncate(CMP_STATE.a, 28)}` }),
      el('div', { class: 'dt-side-b', text: `B · ${truncate(CMP_STATE.b, 28)}` }),
    ]));

    const div = tl.divergence_step;
    let markerInserted = false;

    tl.steps.forEach(step => {
      // Insert branch marker right before the first post-divergence row.
      if (div != null && !markerInserted && step.t > div) {
        host.appendChild(el('div', { class: 'dt-row branch-marker',
          text: `↓ DIVERGED FROM t=${div} ↓` }));
        markerInserted = true;
      }
      const isShared = (div != null && step.t <= div);
      host.appendChild(el('div', { class: 'dt-row ' + (isShared ? 'shared' : 'diverged') }, [
        el('div', { class: 'dt-step', text: `t=${step.t}` }),
        renderSide(step.a),
        renderSide(step.b),
      ]));
    });

    function renderSide(side) {
      const wrap = el('div', { class: 'dt-side' });
      if (!side) {
        wrap.appendChild(el('span', { class: 'dt-empty', text: '— no record —' }));
        return wrap;
      }
      const evRow = el('div', {});
      (side.events || []).forEach(name => {
        evRow.appendChild(el('span', { class: 'ev-pill', text: name }));
      });
      const fRow = el('div', {});
      (side.failures || []).forEach(ft => {
        fRow.appendChild(el('span', { class: `fail-pill sev-${sev(ft)}`, text: ft }));
      });
      if (!(side.events || []).length && !(side.failures || []).length) {
        wrap.appendChild(el('span', { class: 'dt-empty', text: '(quiet)' }));
      } else {
        if ((side.events || []).length) wrap.appendChild(evRow);
        if ((side.failures || []).length) wrap.appendChild(fRow);
      }
      if (side.sentiment != null || side.open != null) {
        const meta = [];
        if (side.sentiment != null) meta.push(`s=${side.sentiment.toFixed(2)}`);
        if (side.open != null) meta.push(`open=${side.open}`);
        wrap.appendChild(el('div', { class: 'dt-meta', text: meta.join(' · ') }));
      }
      return wrap;
    }
  }

  function truncate(s, n) { return (s && s.length > n) ? s.slice(0, n - 1) + '…' : (s || ''); }

  function paintCompareFailures(data) {
    const host = $('#cmp-failures');
    host.innerHTML = '';
    const types = new Set([...Object.keys(data.a.failures_by_type), ...Object.keys(data.b.failures_by_type)]);
    if (!types.size) {
      host.appendChild(el('div', { class: 'empty', text: 'No failures in either run.' }));
      return;
    }
    host.appendChild(el('div', { class: 'diff-row head' }, [
      el('div', { text: 'failure_type' }),
      el('div', { class: 'num', text: 'A' }),
      el('div', { class: 'num', text: 'B' }),
      el('div', { class: 'num', text: 'Δ' }),
    ]));
    [...types].sort().forEach(t => {
      const ca = data.a.failures_by_type[t] || 0;
      const cb = data.b.failures_by_type[t] || 0;
      const delta = cb - ca;
      const cls = delta === 0 ? 'same' : delta > 0 ? 'up' : 'down';
      const pillCls = sev(t) === 'critical' ? 'critical' : sev(t);
      host.appendChild(el('div', { class: 'diff-row' }, [
        el('div', {}, [el('span', { class: `pill ${pillCls}`, text: t })]),
        el('div', { class: 'num', text: ca }),
        el('div', { class: 'num', text: cb }),
        el('div', { class: 'num' }, [el('span', { class: `pill ${cls}`, text: (delta > 0 ? '+' : '') + delta })]),
      ]));
    });
  }

  function paintCompareWorld(data) {
    const host = $('#cmp-world');
    host.innerHTML = '';
    const fa = data.a.final || {}, fb = data.b.final || {};
    const keys = ['customer_sentiment', 'refund_policy_version', 'inventory_delay_minutes', 'system_load', 'deadline_pressure'];
    host.appendChild(el('div', { class: 'diff-row head' }, [
      el('div', { text: 'field' }),
      el('div', { class: 'num', text: 'A' }),
      el('div', { class: 'num', text: 'B' }),
      el('div', { class: 'num', text: 'Δ' }),
    ]));
    keys.forEach(k => {
      if (!(k in fa) && !(k in fb)) return;
      const va = fa[k], vb = fb[k];
      let delta = null;
      if (typeof va === 'number' && typeof vb === 'number') delta = vb - va;
      let cls = 'same';
      if (delta !== null) cls = delta === 0 ? 'same' : delta > 0 ? 'up' : 'down';
      const fmtNum = (n) => typeof n === 'number' ? n.toFixed(3).replace(/\.?0+$/, '') : (n ?? '—');
      host.appendChild(el('div', { class: 'diff-row' }, [
        el('div', { text: k }),
        el('div', { class: 'num mono', text: fmtNum(va) }),
        el('div', { class: 'num mono', text: fmtNum(vb) }),
        el('div', { class: 'num' }, [
          delta == null ? '—' : el('span', { class: `pill ${cls}`, text: (delta > 0 ? '+' : '') + fmtNum(delta) }),
        ]),
      ]));
    });
  }

  // ---------- fork modal -------------------------------------------------

  function openForkModal(detail) {
    const summary = detail.summary;
    const topoName = summary.topology;
    if (!topoName) { toast('This run has no topology metadata; cannot fork', 'error'); return; }
    const topology = TOPOLOGIES.find(t => t.name === topoName);
    if (!topology) { toast('Unknown topology in this run', 'error'); return; }

    const finalStep = summary.final_step || (detail.snapshots?.length ?? 0);
    const ctx = $('#fork-modal-context');
    ctx.innerHTML = '';
    ctx.appendChild(el('span', { text: 'Forking from ' }));
    ctx.appendChild(el('span', { class: 'mono', text: summary.run_id }));
    ctx.appendChild(el('span', { text: ` · ${finalStep} steps completed · topology ` }));
    ctx.appendChild(el('span', { class: `pill topo-${topoName}` }, [topoDot(topoName), topoName]));

    // Branch-at-step controls
    const at = Math.max(0, Math.floor(finalStep / 2));  // default to halfway
    $('#fork-at-slider').min = 0;
    $('#fork-at-slider').max = finalStep;
    $('#fork-at-slider').value = at;
    $('#fork-at').min = 0;
    $('#fork-at').max = finalStep;
    $('#fork-at').value = at;
    $('#fork-at-help').textContent = `0 = re-run from the beginning. Higher = branch later. Max ${finalStep}.`;

    $('#fork-seed').value = '';
    $('#fork-run-id').value = '';

    // Prompt-variant rows per role.
    const variantHost = $('#fork-variants');
    variantHost.innerHTML = '';
    const parentVariant = summary.prompt_variant || 'naive';
    topology.roles.forEach(role => {
      const row = el('div', { class: 'fork-variant-row' }, [
        el('label', {}, [topoDot(topoName), role]),
        el('select', { 'data-role': role }, [
          el('option', { value: '', text: `inherit (${parentVariant})` }),
          el('option', { value: 'naive', text: 'naive' }),
          el('option', { value: 'hardened', text: 'hardened' }),
        ]),
      ]);
      variantHost.appendChild(row);
    });

    // Disable-agent checkboxes per role.
    const disableHost = $('#fork-disable');
    disableHost.innerHTML = '';
    topology.roles.forEach(role => {
      const id = `fork-dis-${role}`;
      const row = el('div', { class: 'fork-disable-row' }, [
        el('label', { for: id, text: role }),
        el('input', { type: 'checkbox', id, 'data-role': role }),
      ]);
      disableHost.appendChild(row);
    });

    const modal = $('#fork-modal');
    modal.classList.remove('hidden');
    modal.setAttribute('aria-hidden', 'false');
    // Focus the timestep input for quick keyboard editing.
    setTimeout(() => $('#fork-at').focus(), 50);
  }

  function closeForkModal() {
    const modal = $('#fork-modal');
    modal.classList.add('hidden');
    modal.setAttribute('aria-hidden', 'true');
  }

  async function submitFork() {
    if (!CURRENT_DETAIL) return;
    const parentId = CURRENT_DETAIL.summary.run_id;

    const branchAt = parseInt($('#fork-at').value, 10);
    if (Number.isNaN(branchAt) || branchAt < 0) {
      toast('Branch step must be a non-negative integer', 'error');
      return;
    }
    const seedRaw = $('#fork-seed').value;
    const seed = seedRaw === '' ? null : parseInt(seedRaw, 10);

    // Collect variant overrides — only include roles where the user picked something.
    const variants = {};
    $$('#fork-variants select').forEach(sel => {
      if (sel.value) variants[sel.dataset.role] = sel.value;
    });

    // Collect disabled agents.
    const disabled = [];
    $$('#fork-disable input[type=checkbox]').forEach(cb => {
      if (cb.checked) disabled.push(cb.dataset.role);
    });

    const runIdRaw = $('#fork-run-id').value.trim();
    const body = {
      branch_at_step: branchAt,
      seed,
      prompt_variants: variants,
      disabled_agents: disabled,
      new_run_id: runIdRaw || null,
    };

    const submitBtn = $('#fork-submit');
    submitBtn.disabled = true;
    submitBtn.textContent = 'Launching…';
    try {
      const res = await api(`/api/runs/${encodeURIComponent(parentId)}/fork`, {
        method: 'POST', body,
      });
      closeForkModal();
      toast(`Fork started: ${res.run_id}`, 'success');
      // Switch to the New Run tab so the live status panel is visible.
      activateTab('new');
      pollRunStatus(res.run_id);
      refreshRuns();  // updates the table in the background
    } catch (e) {
      toast('Fork failed: ' + e.message, 'error');
    } finally {
      submitBtn.disabled = false;
      submitBtn.textContent = 'Launch fork';
    }
  }

  // ---------- detect (MAST demo) -----------------------------------------

  let DETECT_INITED = false;
  let DETECT_TRACES = [];

  async function initDetect() {
    if (DETECT_INITED) return;
    DETECT_INITED = true;
    try {
      const data = await api('/api/mast-demos');
      DETECT_TRACES = data.traces || [];
      renderDetectCards();
    } catch (e) {
      $('#mast-cards').innerHTML = '';
      $('#mast-cards').appendChild(el('div', { class: 'empty', text: 'Could not load MAST demos: ' + e.message }));
    }
  }

  function renderDetectCards() {
    const host = $('#mast-cards');
    host.innerHTML = '';
    if (!DETECT_TRACES.length) {
      host.appendChild(el('div', { class: 'empty', text: 'No MAST demo traces available.' }));
      return;
    }
    DETECT_TRACES.forEach(t => {
      const storyBadge = el('span', {
        class: 'pill ' + (t.story === 'WIN' ? '' : t.story === 'MIXED' ? 'warn' : 'critical'),
        text: t.story,
      });
      const card = el('div', { class: 'card mast-card' }, [
        el('div', { class: 'mast-card-head' }, [
          el('h3', { style: 'margin:0; flex:1;', text: t.title }),
          storyBadge,
        ]),
        el('p', { class: 'help', text: t.task_brief }),
        el('p', { class: 'help', text: t.story_blurb }),
        el('div', { class: 'kv-grid', style: 'margin: 8px 0;' }, [
          el('span', { class: 'k', text: 'Trace size' }),
          el('span', { class: 'v', text: `${fmt.num(t.trace_chars)} chars${t.trace_truncated ? ' (truncated to 100k)' : ''}` }),
          el('span', { class: 'k', text: 'Human-flagged modes' }),
          el('span', { class: 'v', text: `${t.n_ground_truth_positives}` }),
        ]),
        t.ground_truth_modes && t.ground_truth_modes.length
          ? el('div', { class: 'mono muted', style: 'font-size: 11px; margin-bottom: 8px;', text: t.ground_truth_modes.join(' • ') })
          : null,
        el('div', { class: 'actions' }, [
          el('button', {
            class: 'primary',
            text: 'Run drift (cached)',
            onclick: () => runMastDemo(t.id, 'cached'),
          }),
          el('button', {
            class: 'ghost-btn',
            text: 'Run live (≈ 5 s, costs tokens)',
            onclick: () => runMastDemo(t.id, 'live'),
          }),
        ]),
      ].filter(Boolean));
      host.appendChild(card);
    });
  }

  async function runMastDemo(traceId, mode) {
    // Disable all card buttons while a request is in flight.
    const buttons = Array.from(document.querySelectorAll('#mast-cards button'));
    buttons.forEach(b => b.disabled = true);
    // Pull guidelines (only meaningful for live mode; server ignores on cached).
    const guidelinesRaw = (($('#mast-user-guidelines') || {}).value || '').split('\n');
    const userGuidelines = guidelinesRaw.map(s => s.trim()).filter(Boolean);
    try {
      const data = await api('/api/mast-analyze', {
        method: 'POST',
        body: { trace_id: traceId, mode, user_guidelines: userGuidelines },
      });
      renderMastResult(data);
      $('#mast-result').classList.remove('hidden');
      $('#mast-result').scrollIntoView({ behavior: 'smooth', block: 'start' });
    } catch (e) {
      toast('Run failed: ' + e.message, 'error');
    } finally {
      buttons.forEach(b => b.disabled = false);
    }
  }

  function renderMastResult(data) {
    const title = `${data.demo_meta.title} — ${data.mode === 'live' ? 'live' : 'cached'} drift analysis`;
    $('#mast-result-title').textContent = title;

    const meta = $('#mast-result-meta');
    meta.innerHTML = '';
    const s = data.summary || {};
    const kvs = [
      ['MAS framework',    data.mas_name],
      ['Benchmark',        data.benchmark_name],
      ['Trace size',       `${fmt.num(data.n_chars)} chars${data.truncated ? ' (truncated)' : ''}`],
      ['Mode',             data.mode + (data.latency_s ? ` (${data.latency_s}s)` : '')],
      ['Human-flagged',    s.n_ground_truth_positives],
      ['drift predictions', s.n_predicted_positives],
    ];
    kvs.forEach(([k, v]) => {
      meta.appendChild(el('span', { class: 'k', text: k }));
      meta.appendChild(el('span', { class: 'v', text: v == null || v === '' ? '—' : String(v) }));
    });

    // Precision / recall headline
    const summaryHost = $('#mast-result-summary');
    summaryHost.innerHTML = '';
    const precStr = s.precision != null ? s.precision.toFixed(2) : '—';
    const recStr  = s.recall    != null ? s.recall.toFixed(2)    : '—';
    const f1Str   = s.f1        != null ? s.f1.toFixed(2)        : '—';
    const grid = el('div', { class: 'failure-summary' }, [
      el('div', { class: 'failure-cell' }, [
        el('div', { class: 'ftype', text: 'TP' }),
        el('div', { class: 'fcount', text: String(s.n_tp) }),
      ]),
      el('div', { class: 'failure-cell sev-warn' }, [
        el('div', { class: 'ftype', text: 'FP' }),
        el('div', { class: 'fcount', text: String(s.n_fp) }),
      ]),
      el('div', { class: 'failure-cell sev-critical' }, [
        el('div', { class: 'ftype', text: 'FN' }),
        el('div', { class: 'fcount', text: String(s.n_fn) }),
      ]),
      el('div', { class: 'failure-cell' }, [
        el('div', { class: 'ftype', text: 'TN' }),
        el('div', { class: 'fcount', text: String(s.n_tn) }),
      ]),
      el('div', { class: 'failure-cell' }, [
        el('div', { class: 'ftype', text: 'Precision' }),
        el('div', { class: 'fcount', text: precStr }),
      ]),
      el('div', { class: 'failure-cell' }, [
        el('div', { class: 'ftype', text: 'Recall' }),
        el('div', { class: 'fcount', text: recStr }),
      ]),
      el('div', { class: 'failure-cell' }, [
        el('div', { class: 'ftype', text: 'F1' }),
        el('div', { class: 'fcount', text: f1Str }),
      ]),
    ]);
    summaryHost.appendChild(grid);

    // Per-mode side-by-side
    const modesHost = $('#mast-result-modes');
    modesHost.innerHTML = '';
    // Sort: TP first (wins), FN next (misses), FP, then TN
    const order = { TP: 0, FN: 1, FP: 2, TN: 3 };
    const sorted = (data.per_mode || []).slice().sort((a, b) =>
      (order[a.outcome] ?? 9) - (order[b.outcome] ?? 9) || (a.mode_id || '').localeCompare(b.mode_id || '')
    );
    sorted.forEach(m => {
      // Hide pure TN rows by default to keep the list focused on signal
      if (m.outcome === 'TN') return;
      const pillClass =
        m.outcome === 'TP' ? '' :
        m.outcome === 'FN' ? 'critical' :
        m.outcome === 'FP' ? 'warn' : '';
      const agree = m.annotator_agreement || [0, 0];
      modesHost.appendChild(el('div', { class: 'failure-row' }, [
        el('div', { class: 'step', text: m.outcome }),
        el('div', {}, [
          el('div', {}, [
            el('span', { class: `pill ${pillClass}`, text: m.name }),
            ' ',
            el('span', { class: 'muted', text: `human raters: ${agree[0]}/${agree[1]}` }),
          ]),
          m.evidence
            ? el('div', { class: 'mono muted', text: `drift evidence: ${m.evidence}` })
            : (m.outcome === 'FN'
                ? el('div', { class: 'mono muted', text: 'drift did not flag this — human raters did.' })
                : null),
        ].filter(Boolean)),
      ]));
    });
    if (!modesHost.children.length) {
      modesHost.appendChild(el('div', { class: 'empty', text: 'No signal vs ground truth — all modes were TN (both agreed no failure).' }));
    }
  }

  // ---------- analyze (external trace) -----------------------------------

  let ANALYZE_INITED = false;

  function initAnalyze() {
    if (ANALYZE_INITED) return;
    if (!TOPOLOGIES.length) {
      // Topologies not loaded yet — bail out; activateTab('analyze') will
      // re-call us, or bootstrap will once TOPOLOGIES populates.
      return;
    }
    const sel = $('#analyze-topology');
    sel.innerHTML = '';
    TOPOLOGIES.forEach(t => sel.appendChild(el('option', { value: t.name, text: t.name })));
    sel.addEventListener('change', onAnalyzeTopologyChange);
    onAnalyzeTopologyChange();

    $('#analyze-load-sample').addEventListener('click', loadSampleTrace);
    $('#analyze-clear').addEventListener('click', clearAnalyze);
    $('#analyze-submit').addEventListener('click', submitAnalyze);
    ANALYZE_INITED = true;
  }

  function onAnalyzeTopologyChange() {
    const t = TOPOLOGIES.find(x => x.name === $('#analyze-topology').value);
    if (!t) return;
    const det = (t.detectors || []).join(', ');
    $('#analyze-topology-help').textContent =
      `${t.description || ''}${det ? '  •  Detectors: ' + det : ''}`;
  }

  async function loadSampleTrace() {
    try {
      const data = await api('/api/sample-trace');
      $('#analyze-trace').value = data.trace;
      if (data.topology) {
        $('#analyze-topology').value = data.topology;
        onAnalyzeTopologyChange();
      }
      toast('Sample trace loaded.', 'info');
    } catch (e) {
      toast('Could not load sample: ' + e.message, 'error');
    }
  }

  function clearAnalyze() {
    $('#analyze-trace').value = '';
    $('#analyze-result').classList.add('hidden');
    $('#analyze-summary').innerHTML = '';
    $('#analyze-failure-summary').innerHTML = '';
    $('#analyze-failure-list').innerHTML = '';
  }

  async function submitAnalyze() {
    const trace = $('#analyze-trace').value.trim();
    const topology = $('#analyze-topology').value;
    if (!trace) {
      toast('Paste a trace first (or click "Load sample trace").', 'error');
      return;
    }
    const btn = $('#analyze-submit');
    btn.disabled = true;
    btn.textContent = 'Running detectors…';
    try {
      const data = await api('/api/analyze', { method: 'POST', body: { topology, trace } });
      renderAnalyzeResult(data);
      $('#analyze-result').classList.remove('hidden');
      // Scroll the results into view so the user sees them immediately.
      $('#analyze-result').scrollIntoView({ behavior: 'smooth', block: 'start' });
    } catch (e) {
      toast('Analyze failed: ' + e.message, 'error');
    } finally {
      btn.disabled = false;
      btn.textContent = 'Run detectors';
    }
  }

  function renderAnalyzeResult(data) {
    const summary = data.summary || {};
    const meta = $('#analyze-summary');
    meta.innerHTML = '';
    const kvs = [
      ['Topology',   summary.topology],
      ['Steps',      `${summary.first_step ?? '–'} .. ${summary.last_step ?? '–'}`],
      ['Snapshots',  summary.n_snapshots],
      ['Actions',    summary.n_actions],
      ['Events',     summary.n_events],
      ['Failures',   (data.failures || []).length],
    ];
    kvs.forEach(([k, v]) => {
      meta.appendChild(el('span', { class: 'k', text: k }));
      meta.appendChild(el('span', { class: 'v', text: v == null || v === '' ? '—' : String(v) }));
    });

    const summaryHost = $('#analyze-failure-summary');
    const listHost = $('#analyze-failure-list');
    summaryHost.innerHTML = '';
    listHost.innerHTML = '';

    const failures = data.failures || [];
    if (!failures.length) {
      summaryHost.appendChild(el('div', { class: 'empty', text: 'No failures detected in this trace.' }));
      return;
    }

    const byType = {};
    failures.forEach(f => { byType[f.failure_type] = (byType[f.failure_type] || 0) + 1; });
    const sumGrid = el('div', { class: 'failure-summary' });
    Object.entries(byType).sort((a, b) => b[1] - a[1]).forEach(([type, count]) => {
      sumGrid.appendChild(el('div', { class: `failure-cell sev-${sev(type)}` }, [
        el('div', { class: 'ftype', text: type }),
        el('div', { class: 'fcount', text: fmt.num(count) }),
      ]));
    });
    summaryHost.appendChild(sumGrid);

    failures.slice(0, 200).forEach(f => {
      listHost.appendChild(el('div', { class: 'failure-row' }, [
        el('div', { class: 'step', text: `t=${f.timestep}` }),
        el('div', {}, [
          el('div', {}, [
            el('span', { class: `pill ${sev(f.failure_type) === 'critical' ? 'critical' : sev(f.failure_type)}`, text: f.failure_type }),
            ' ',
            el('span', { class: 'muted', text: (f.agents_involved || []).join(', ') || '—' }),
          ]),
          el('div', { class: 'mono muted', html: escapeHtml(f.summary || '') }),
        ]),
      ]));
    });
    if (failures.length > 200) {
      listHost.appendChild(el('div', { class: 'muted', text: `… and ${failures.length - 200} more.` }));
    }
  }

  // ---------- custom / BYOA -----------------------------------------------

  let CUSTOM_INITED = false;

  function initCustom() {
    if (CUSTOM_INITED) return;
    if (!TOPOLOGIES.length) return;  // bootstrap hasn't loaded yet; re-call will happen
    const sel = $('#byoa-topology');
    sel.innerHTML = '';
    TOPOLOGIES.forEach(t => sel.appendChild(el('option', { value: t.name, text: t.name })));
    sel.addEventListener('change', onCustomTopologyChange);
    onCustomTopologyChange();

    $('#byoa-load-example').addEventListener('click', loadCustomExample);
    $('#byoa-clear').addEventListener('click', clearCustom);
    $('#byoa-submit').addEventListener('click', submitCustom);
    CUSTOM_INITED = true;
  }

  function onCustomTopologyChange() {
    const t = TOPOLOGIES.find(x => x.name === $('#byoa-topology').value);
    if (!t) return;
    const det = (t.detectors || []).join(', ');
    $('#byoa-topology-help').textContent =
      `Layers ${t.name}-specific detectors on top of the cross-topology ones${det ? '  •  ' + det : ''}`;
  }

  async function loadCustomExample() {
    try {
      const ex = await api('/api/byoa-example');
      $('#byoa-code').value = ex.code;
      if (ex.detector_topology) {
        $('#byoa-topology').value = ex.detector_topology;
        onCustomTopologyChange();
      }
      toast('Example loaded — hit Run on my agents.', 'info');
    } catch (e) {
      toast('Could not load example: ' + e.message, 'error');
    }
  }

  function clearCustom() {
    $('#byoa-code').value = '';
    $('#byoa-result').classList.add('hidden');
    $('#byoa-summary').innerHTML = '';
    $('#byoa-failure-summary').innerHTML = '';
    $('#byoa-failure-list').innerHTML = '';
    const acl = $('#byoa-auto-chaos-list');
    if (acl) acl.innerHTML = '';
    const acc = $('#byoa-auto-chaos-card');
    if (acc) acc.hidden = true;
  }

  async function submitCustom() {
    const code = $('#byoa-code').value;
    if (!code.trim()) {
      toast('Paste agent code first (or click "Load example").', 'error');
      return;
    }
    const guidelinesRaw = (($('#byoa-user-guidelines') || {}).value || '').split('\n');
    const userGuidelines = guidelinesRaw.map(s => s.trim()).filter(Boolean);
    const body = {
      code,
      detector_topology: $('#byoa-topology').value,
      steps: parseInt($('#byoa-steps').value, 10) || 10,
      seed: parseInt($('#byoa-seed').value, 10) || 0,
      auto_chaos: ($('#byoa-auto-chaos') || {}).value || 'off',
      judge: ($('#byoa-judge') || {}).value || 'off',
      judge_model: ($('#byoa-judge-model') || {}).value.trim() || null,
      judge_every: parseInt(($('#byoa-judge-every') || {}).value, 10) || 5,
      user_guidelines: userGuidelines,
    };
    const btn = $('#byoa-submit');
    btn.disabled = true;
    btn.textContent = 'Running…';
    try {
      const data = await api('/api/byoa', { method: 'POST', body });
      renderCustomResult(data);
      $('#byoa-result').classList.remove('hidden');
      $('#byoa-result').scrollIntoView({ behavior: 'smooth', block: 'start' });
    } catch (e) {
      toast('Run failed: ' + e.message, 'error');
    } finally {
      btn.disabled = false;
      btn.textContent = 'Run on my agents';
    }
  }

  function renderCustomResult(data) {
    const summary = data.summary || {};
    const meta = $('#byoa-summary');
    meta.innerHTML = '';
    const agentList = (summary.agents || []).map(a => `${a.name} (${a.role})`).join(', ') || '—';
    const acIntensity = summary.auto_chaos && summary.auto_chaos !== 'off' ? summary.auto_chaos : null;
    const judgeSpec = summary.judge && summary.judge !== 'off' ? summary.judge : null;
    const judgeDesc = judgeSpec
      ? `${judgeSpec}${summary.judge_model ? ` (${summary.judge_model})` : ''} — ${summary.n_failures_llm || 0} fired`
      : 'off';
    const failuresDesc = (summary.n_failures_llm || 0) > 0
      ? `${summary.n_failures} (${summary.n_failures_deterministic} deterministic + ${summary.n_failures_llm} llm-judged)`
      : String(summary.n_failures);
    const kvs = [
      ['Detectors',      summary.detector_topology],
      ['Agents',         agentList],
      ['Steps completed', `${summary.steps_completed} / ${summary.steps_requested}`],
      ['Actions',        summary.n_actions],
      ['Events',         summary.n_events],
      ['Auto-chaos',     acIntensity ? `${acIntensity} — ${summary.n_auto_chaos_injected} injected` : 'off'],
      ['LLM judge',      judgeDesc],
      ['Failures',       failuresDesc],
    ];
    kvs.forEach(([k, v]) => {
      meta.appendChild(el('span', { class: 'k', text: k }));
      meta.appendChild(el('span', { class: 'v', text: v == null || v === '' ? '—' : String(v) }));
    });

    // Auto-chaos card: list every injected event so the user can see what drift fired.
    const acCard = $('#byoa-auto-chaos-card');
    const acList = $('#byoa-auto-chaos-list');
    if (acCard && acList) {
      acList.innerHTML = '';
      const injected = data.auto_chaos_injected || [];
      if (injected.length) {
        acCard.hidden = false;
        injected.forEach(ev => {
          acList.appendChild(el('div', { class: 'failure-row' }, [
            el('div', { class: 'step', text: `t=${ev.timestep}` }),
            el('div', {}, [
              el('div', {}, [
                el('span', { class: 'pill', text: ev.name }),
              ]),
              el('div', { class: 'mono muted', text: ev.summary || '' }),
            ]),
          ]));
        });
      } else {
        acCard.hidden = true;
      }
    }

    const summaryHost = $('#byoa-failure-summary');
    const listHost = $('#byoa-failure-list');
    summaryHost.innerHTML = '';
    listHost.innerHTML = '';

    const failures = data.failures || [];
    if (!failures.length) {
      summaryHost.appendChild(el('div', { class: 'empty', text: 'No failures detected — clean run.' }));
      return;
    }

    const byType = {};
    failures.forEach(f => { byType[f.failure_type] = (byType[f.failure_type] || 0) + 1; });
    const sumGrid = el('div', { class: 'failure-summary' });
    Object.entries(byType).sort((a, b) => b[1] - a[1]).forEach(([type, count]) => {
      sumGrid.appendChild(el('div', { class: `failure-cell sev-${sev(type)}` }, [
        el('div', { class: 'ftype', text: type }),
        el('div', { class: 'fcount', text: fmt.num(count) }),
      ]));
    });
    summaryHost.appendChild(sumGrid);

    failures.slice(0, 200).forEach(f => {
      listHost.appendChild(el('div', { class: 'failure-row' }, [
        el('div', { class: 'step', text: `t=${f.timestep}` }),
        el('div', {}, [
          el('div', {}, [
            el('span', { class: `pill ${sev(f.failure_type) === 'critical' ? 'critical' : sev(f.failure_type)}`, text: f.failure_type }),
            ' ',
            el('span', { class: 'muted', text: (f.agents_involved || []).join(', ') || '—' }),
          ]),
          el('div', { class: 'mono muted', html: escapeHtml(f.summary || '') }),
        ]),
      ]));
    });
    if (failures.length > 200) {
      listHost.appendChild(el('div', { class: 'muted', text: `… and ${failures.length - 200} more.` }));
    }
  }

  function paintCompareAgents(data) {
    const host = $('#cmp-agents');
    host.innerHTML = '';
    const allAgents = new Set([
      ...Object.keys(data.a.actions_by_agent_kind || {}),
      ...Object.keys(data.b.actions_by_agent_kind || {}),
    ]);
    if (!allAgents.size) {
      host.appendChild(el('div', { class: 'empty', text: 'No actions to compare.' }));
      return;
    }
    [...allAgents].sort().forEach(agent => {
      const ka = data.a.actions_by_agent_kind[agent] || {};
      const kb = data.b.actions_by_agent_kind[agent] || {};
      const kinds = new Set([...Object.keys(ka), ...Object.keys(kb)]);
      const block = el('div', { class: 'card', style: 'margin-bottom: 12px;' }, [
        el('h4', { style: 'margin: 0 0 8px; font-size: 13px;', text: agent }),
        el('div', { class: 'diff-row head' }, [
          el('div', { text: 'kind' }),
          el('div', { class: 'num', text: 'A' }),
          el('div', { class: 'num', text: 'B' }),
          el('div', { class: 'num', text: 'Δ' }),
        ]),
      ]);
      [...kinds].sort().forEach(k => {
        const ca = ka[k] || 0, cb = kb[k] || 0, d = cb - ca;
        const cls = d === 0 ? 'same' : d > 0 ? 'up' : 'down';
        block.appendChild(el('div', { class: 'diff-row' }, [
          el('div', { class: 'mono', text: k }),
          el('div', { class: 'num', text: ca }),
          el('div', { class: 'num', text: cb }),
          el('div', { class: 'num' }, [el('span', { class: `pill ${cls}`, text: (d > 0 ? '+' : '') + d })]),
        ]));
      });
      host.appendChild(block);
    });
  }

  // ---------- adapter tab (langgraph) -------------------------------------

  let _adapterWired = false;
  function initAdapter() {
    if (_adapterWired) return;
    _adapterWired = true;
    $('#adapter-run').addEventListener('click', runAdapterDemo);
  }

  async function runAdapterDemo() {
    const btn = $('#adapter-run');
    const status = $('#adapter-status');
    const intensity = $('#adapter-intensity').value;
    const seed = parseInt($('#adapter-seed').value, 10) || 0;
    const excludeRaw = ($('#adapter-exclude').value || '').trim();
    const auto_chaos_exclude = excludeRaw
      ? excludeRaw.split(',').map(s => s.trim()).filter(Boolean)
      : [];

    btn.disabled = true;
    status.textContent = 'running…';
    try {
      const data = await api('/api/adapter-demo', {
        method: 'POST',
        body: { intensity, seed, auto_chaos_exclude },
      });
      renderAdapterResult(data);
      $('#adapter-result').classList.remove('hidden');
      $('#adapter-result').scrollIntoView({ behavior: 'smooth', block: 'start' });
      status.textContent = `done — ${data.perturbations.length} perturbations`;
    } catch (e) {
      toast('Adapter run failed: ' + e.message, 'error');
      status.textContent = '';
    } finally {
      btn.disabled = false;
    }
  }

  function renderAdapterResult(data) {
    // --- summary kv ---
    const summary = $('#adapter-summary');
    summary.innerHTML = '';
    const kvs = [
      ['Graph',          data.graph_name],
      ['Intensity',      data.intensity],
      ['Seed',           data.seed],
      ['Patterns total', data.patterns_total + ' (schema-derived)'],
      ['Perturbations',  data.perturbations.length],
      ['Crashed',        data.n_crashed],
      ['Diverged',       data.n_diverged],
      ['Unchanged',      data.n_unchanged],
    ];
    kvs.forEach(([k, v]) => {
      summary.appendChild(el('span', { class: 'k', text: k }));
      summary.appendChild(el('span', { class: 'v', text: v == null || v === '' ? '—' : String(v) }));
    });

    const headline = $('#adapter-headline');
    headline.innerHTML = '';
    headline.appendChild(el('p', {
      class: 'help',
      text: data.graph_description,
    }));

    // --- baseline ---
    const base = $('#adapter-baseline');
    base.innerHTML = '';
    const b = data.baseline;
    if (b.crashed) {
      base.appendChild(el('div', {}, [
        el('span', { class: 'pill danger', text: 'BASELINE CRASHED' }),
        el('div', { class: 'mono', style: 'margin-top: 8px;', text: `${b.error_type}: ${b.error}` }),
      ]));
    } else {
      const fs = b.final_state || {};
      base.appendChild(el('pre', { class: 'mono', text: JSON.stringify(fs, null, 2) }));
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
    // Sort: crashes -> diverges -> unchanged. Highest-signal first.
    const sorted = data.perturbations.slice().sort((p, q) => {
      const rank = (x) => x.crashed ? 0 : x.diverged ? 1 : 2;
      const r = rank(p) - rank(q);
      return r !== 0 ? r : p.event_name.localeCompare(q.event_name);
    });
    sorted.forEach(p => host.appendChild(renderPerturbation(p)));
  }

  function renderPerturbation(p) {
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
      detail = 'Graph absorbed the perturbation; no observable change in final state.';
    }

    const head = el('div', { class: 'row', style: 'align-items: center; gap: 10px; margin-bottom: 4px;' }, [
      el('span', { class: pillClass, text: pillText }),
      el('code', { text: p.event_name }),
      el('span', { class: 'muted', text: `(${p.duration_s}s)` }),
    ]);
    const sum = el('div', { class: 'muted', style: 'margin-bottom: 6px;', text: p.event_summary });
    const det = el('div', { class: 'mono', style: 'white-space: pre-wrap; word-break: break-word;', text: detail });
    return el('div', { class: 'failure-cell', style: 'padding: 10px 12px;' }, [head, sum, det]);
  }

  // ---------- utils -------------------------------------------------------

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }

  // ---------- start -------------------------------------------------------

  bootstrap();
})();
