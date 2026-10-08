// FairFix NT workspace. Display only: every score, level and position shown here comes
// from the server, which gets them from the triage pipeline. Roles: the officer records
// reports; the admin (maintenance coordinator) works the queue and assigns tradies.

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const page = () => $('#page');
const state = { me: null, ref: null, filter: '', tab: 'all' };

const NAV = {
  admin: [
    ['dashboard', 'Dashboard', 'layout-dashboard'],
    ['new', 'Upload report', 'upload'],
    ['requests', 'Repair Requests', 'file-text'],
    ['queue', 'Priority Queue', 'list-ordered'],
    ['communities', 'Communities', 'map'],
    ['fairness', 'Fairness Monitor', 'scale'],
    ['tradies', 'Tradies', 'hard-hat'],
    ['reports', 'Reports', 'chart-column'],
  ],
  officer: [
    ['new', 'Upload report', 'upload'],
    ['mine', 'My submissions', 'clipboard-list'],
  ],
};

// ---- helpers -------------------------------------------------------------------

function esc(v = '') {
  return String(v ?? '').replace(/[&<>'"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[c]));
}
const icon = (name, cls = 'size-4') => `<i data-lucide="${name}" class="${cls}"></i>`;
const refreshIcons = () => window.lucide && lucide.createIcons({ attrs: { 'stroke-width': 1.75 } });
const tip = (inner, text) => `<span class="tip inline-flex" tabindex="0">${inner}<span class="tip-text">${esc(text)}</span></span>`;

function toast(msg) {
  const t = $('#toast');
  t.textContent = msg;
  t.classList.remove('opacity-0', 'translate-y-2');
  clearTimeout(window.__toast);
  window.__toast = setTimeout(() => t.classList.add('opacity-0', 'translate-y-2'), 3000);
}

async function api(url, opts = {}) {
  const o = { ...opts, headers: { ...(opts.headers || {}) } };
  if (o.body && !(o.body instanceof FormData)) {
    o.headers['Content-Type'] = 'application/json';
    o.body = JSON.stringify(o.body);
  }
  const r = await fetch(url, o);
  const data = (r.headers.get('content-type') || '').includes('json') ? await r.json() : await r.text();
  if (r.status === 401 && url !== '/api/login') { showLogin(); throw new Error('Please sign in'); }
  if (!r.ok) {
    const d = data?.detail;
    throw new Error(Array.isArray(d) ? d.map(x => x.msg).join('; ') : d || data || 'Request failed');
  }
  return data;
}

function setHead(title, sub = '', actions = '') {
  $('#pageTitle').textContent = title;
  $('#pageSubtitle').textContent = sub;
  $('#pageActions').innerHTML = actions;
  document.title = `${title} · FairFix NT`;
}

async function reference() {
  if (!state.ref) {
    state.ref = await api('/api/reference');
    const m = state.ref.mode;
    $('#modeBadge').innerHTML = `<span class="badge b-outline">${icon('database', 'size-3.5')}Saved model answers</span>
      <span class="tip-text">Model reading: ${esc(m.recorded.replace('recorded:', ''))}. New text: ${esc(m.new_text)}.</span>`;
    $('#modeBadge').classList.remove('hidden');
    refreshIcons();
  }
  return state.ref;
}

function card(title, body, { desc = '', action = '', pad = true } = {}) {
  return `<section class="card">
    ${title ? `<div class="card-head"><div><h2 class="card-title">${title}</h2>${desc ? `<p class="card-desc">${desc}</p>` : ''}</div>${action}</div>` : ''}
    <div class="${pad ? 'card-body' : 'pt-3'}">${body}</div></section>`;
}

function stat(label, value, sub, iconName, tone = 'text-slate-400') {
  return `<div class="card card-body">
    <div class="flex items-center justify-between"><span class="text-sm font-medium text-slate-600">${label}</span>${icon(iconName, `size-4 ${tone}`)}</div>
    <div class="mt-2 text-3xl font-semibold tracking-tight">${value}</div>
    ${sub ? `<p class="mt-1 text-xs text-slate-500">${sub}</p>` : ''}</div>`;
}

const tableWrap = (head, rows, empty, cols) => `<div class="overflow-x-auto"><table class="tbl"><thead><tr>${head}</tr></thead>
  <tbody>${rows || `<tr><td colspan="${cols}" class="py-8 text-center text-slate-500">${esc(empty)}</td></tr>`}</tbody></table></div>`;

// ---- labels (display only: the order always comes from the server) -----------------

function priorityOf(j) {
  if (j.in_review_band) return 'Needs review';
  if (j.safety_level === 2) return 'Critical';
  if (j.safety_level === 1 || j.urgency_tally === 4) return 'High';
  if (j.urgency_tally === 3) return 'Medium';
  if (j.urgency_tally === 2) return 'Low';
  return 'Needs review';
}
const PRIORITY_CLASS = { Critical: 'b-critical', High: 'b-high', Medium: 'b-medium', Low: 'b-low', 'Needs review': 'b-review' };
const PRIORITY_TIP = {
  Critical: 'Active safety risk described in the report',
  High: 'Possible safety risk, or an emergency repair with no working alternative',
  Medium: 'Listed repair, one point lower (e.g. another working one named)',
  Low: 'General repair',
  'Needs review': 'Not on the fault list: waiting for a coordinator tier call',
};
const priorityBadge = j => tip(`<span class="badge ${PRIORITY_CLASS[priorityOf(j)]}">${priorityOf(j)}</span>`, PRIORITY_TIP[priorityOf(j)]);

const categoryBadge = j => j.tier === 'dangerous'
  ? tip('<span class="badge b-outline">Emergency</span>', 'Emergency repair (NT Residential Tenancies Act s63)')
  : j.tier === 'standard' ? '<span class="badge b-outline">General</span>'
  : tip('<span class="badge b-outline text-slate-400">Unlisted</span>', 'Not on the fault list');

const statusBadge = j => j.status === 'Assigned' ? `<span class="badge b-accent">${icon('user-check', 'size-3')}${esc(j.tradie)}</span>`
  : j.status === 'Completed' ? '<span class="badge b-ok">Completed</span>' : '<span class="badge b-outline">Open</span>';

const safetyText = j => ['No safety risk', 'Possible safety risk', 'Active safety risk'][j.safety_level];
const waiting = d => `${d} day${d === 1 ? '' : 's'}`;
// Stage 5: straight-line km to the nearest NT housing office (display only, never ordering).
const dist = d => !d || d.km == null ? '<span class="text-slate-400">unknown</span>'
  : `<span title="Straight line to the ${esc(d.office)}">${Math.round(d.km)} km</span>`;
const source = j => j.source_tag === 'officer' ? 'Officer (phone call)' : 'Tenant (form or message)';
const ref = (id, link = true) => link ? `<a href="#/job/${esc(id)}" class="whitespace-nowrap font-medium text-teal-700 hover:underline">${esc(id)}</a>` : `<span class="whitespace-nowrap font-medium">${esc(id)}</span>`;

function matches(j) {
  const q = state.filter.toLowerCase();
  return !q || `${j.job_id || ''} ${j.report_id} ${j.community} ${j.fault || j.raw_text || ''} ${j.tradie || ''}`.toLowerCase().includes(q);
}

function searchBox(placeholder) {
  return `<div class="relative w-full sm:w-64">${icon('search', 'pointer-events-none absolute left-2.5 top-2.5 size-4 text-slate-400')}
    <input id="search" type="search" class="input pl-8" placeholder="${placeholder}" value="${esc(state.filter)}" aria-label="Search"></div>`;
}

let searchTimer;
function wireSearch(render) {
  const el = $('#search');
  if (!el) return;
  el.oninput = e => {
    state.filter = e.target.value;
    clearTimeout(searchTimer);
    searchTimer = setTimeout(async () => {
      const pos = $('#search')?.selectionStart;
      await render();
      const s = $('#search');
      if (s) { s.focus(); if (pos != null) s.setSelectionRange(pos, pos); }
    }, 200);
  };
}

// ---- shell -------------------------------------------------------------------------

function navHtml() {
  const items = NAV[state.me.role].map(([r, label, ic]) =>
    `<a href="#/${r}" data-route="${r}" class="nav-link">${icon(ic)}${label}</a>`).join('');
  return `<div class="flex h-14 items-center gap-2.5 border-b border-slate-200 px-4">
      <div class="grid size-7 place-items-center rounded-md bg-teal-700 text-[11px] font-semibold text-white">FF</div>
      <span class="text-sm font-semibold">FairFix NT</span></div>
    <nav class="flex-1 space-y-0.5 p-3" aria-label="Main navigation">${items}</nav>
    <div class="space-y-2 border-t border-slate-200 p-3 text-xs text-slate-500">
      <div class="rounded-md bg-slate-50 p-2.5"><div class="font-medium text-slate-700">Queue order</div>Safety → urgency → oldest report. Distance is never used.</div>
      <div class="flex items-center gap-2 px-1">${icon('phone', 'size-3.5')}Repairs hotline <span class="font-medium text-slate-700">1800 104 076</span></div>
    </div>`;
}

function showLogin() {
  state.me = null;
  state.ref = null;
  $('#appView').classList.add('hidden');
  $('#loginView').classList.remove('hidden');
  $('#loginView').classList.add('grid');
  $('#loginUser').focus();
}

function showApp() {
  $('#loginView').classList.add('hidden');
  $('#loginView').classList.remove('grid');
  $('#appView').classList.remove('hidden');
  $('#sidebar').innerHTML = navHtml();
  $('#sheetPanel').innerHTML = navHtml();
  const initials = state.me.name.split(' ').map(w => w[0]).join('').slice(0, 2);
  $('#userInitials').textContent = initials;
  $('#userName').textContent = state.me.name;
  $('#menuName').textContent = state.me.name;
  $('#menuRole').textContent = state.me.role === 'admin' ? 'Admin · works the queue' : 'Officer · records reports';
  refreshIcons();
  reference().catch(() => {});
  const [, name] = location.hash.split('/');
  if (!name || !allowed(name)) location.hash = `#/${NAV[state.me.role][0][0]}`;  // hashchange renders it
  else route();
}

const allowed = name => name === 'how' || (name === 'job' ? state.me.role === 'admin' : NAV[state.me.role].some(([r]) => r === name));

function closeSheet() { $('#sheet').classList.add('hidden'); }
$('#menuBtn').onclick = () => $('#sheet').classList.remove('hidden');
$('#sheet').onclick = e => { if (e.target.closest('[data-close-sheet]') || e.target.closest('a')) closeSheet(); };
$('#userBtn').onclick = e => {
  e.stopPropagation();
  const open = $('#userMenu').classList.toggle('hidden') === false;
  $('#userBtn').setAttribute('aria-expanded', open);
};
document.addEventListener('click', e => { if (!e.target.closest('#userMenu')) $('#userMenu').classList.add('hidden'); });
document.addEventListener('keydown', e => { if (e.key === 'Escape') { closeSheet(); $('#userMenu').classList.add('hidden'); } });

$('#loginForm').onsubmit = async e => {
  e.preventDefault();
  try {
    state.me = await api('/api/login', { method: 'POST', body: Object.fromEntries(new FormData(e.target)) });
    e.target.reset();
    location.hash = '';
    showApp();
  } catch (err) { toast(err.message); }
};
$$('.demo-fill').forEach(b => b.onclick = () => { $('#loginUser').value = b.dataset.user; $('#loginPass').value = b.dataset.pass; });
$('#logoutBtn').onclick = async () => { await api('/api/logout', { method: 'POST' }).catch(() => {}); location.hash = ''; showLogin(); };

// ---- Admin: dashboard -------------------------------------------------------------

function tradieMatches(jobs, tradies) {
  // Unassigned jobs with at least one available tradie listed for the fault's trade.
  return jobs.filter(j => j.status === 'Open' && j.needed_trades.length &&
    tradies.some(t => t.available && t.trades.some(x => j.needed_trades.includes(x)))).length;
}

async function renderDashboard() {
  setHead('Dashboard', 'Overview of housing maintenance requests across the NT.',
    `<a href="#/queue" class="btn btn-outline">${icon('list-ordered')}Priority queue</a>`);
  const [q, tradies] = await Promise.all([api('/api/queue'), api('/api/tradies')]);
  const open = [...q.ranked, ...q.review_band];
  const count = p => q.ranked.filter(j => priorityOf(j) === p).length;
  const avgWait = open.length ? Math.round(open.reduce((s, j) => s + j.days_waiting, 0) / open.length) : 0;
  const top = q.ranked.slice(0, 8);
  if (!open.length && !q.needs_human.length) {
    page().innerHTML = `<div class="card flex flex-col items-center gap-3 px-6 py-16 text-center">
      ${icon('inbox', 'size-8 text-slate-300')}<div><h2 class="text-base font-semibold">No reports yet</h2>
      <p class="mt-1 text-sm text-slate-500">Upload a housing document to run it through the six stages. Nothing is pre-loaded.</p></div>
      <a href="#/new" class="btn btn-primary">${icon('upload')}Upload report</a></div>`;
    return;
  }

  page().innerHTML = `
    <div class="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
      ${stat('Open requests', open.length + q.needs_human.length, `${q.ranked.length} ranked · ${q.review_band.length} in review · ${q.needs_human.length} need a read`, 'inbox')}
      ${stat('Critical', count('Critical'), 'Active safety risk described', 'shield-alert', 'text-red-500')}
      ${stat('High priority', count('High'), 'Possible safety risk or emergency repair', 'triangle-alert', 'text-amber-500')}
      ${stat('Tradie matches', tradieMatches(open, tradies), 'Unassigned jobs with an available, qualified tradie', 'user-check', 'text-teal-600')}
    </div>
    <div class="mt-4 grid gap-4 lg:grid-cols-3">
      <div class="min-w-0 lg:col-span-2">${card('Priority queue', tableWrap(
        '<th class="w-12">Rank</th><th>Reference</th><th>Community</th><th>Issue</th><th>Priority</th><th>NT category</th><th>Status</th>',
        top.map(j => `<tr class="row-link" data-job="${esc(j.job_id)}"><td class="font-medium text-slate-500">${j.position}</td><td>${ref(j.job_id)}</td>
          <td>${esc(j.community)}</td><td class="max-w-[200px] truncate" title="${esc(j.fault)}">${esc(j.fault)}</td>
          <td>${priorityBadge(j)}</td><td>${categoryBadge(j)}</td><td>${statusBadge(j)}</td></tr>`).join(''),
        'No ranked jobs.', 7),
        { desc: 'Next in line: safety first, then urgency, then oldest report.', pad: false,
          action: `<a href="#/queue" class="btn btn-ghost btn-sm">View all ${icon('arrow-up-right', 'size-3.5')}</a>` })}</div>
      <div class="space-y-4">
        ${card('Workload', `
          <div class="kv"><span class="flex items-center gap-2 text-slate-600">${icon('clock', 'size-4 text-slate-400')}Average waiting time</span><span class="font-semibold">${waiting(avgWait)}</span></div>
          <div class="h-px bg-slate-100"></div>
          <div class="kv"><span class="flex items-center gap-2 text-slate-600">${icon('users', 'size-4 text-slate-400')}Tradie profiles</span><span class="font-semibold">${tradies.length}</span></div>
          <div class="h-px bg-slate-100"></div>
          <div class="kv"><span class="flex items-center gap-2 text-slate-600">${icon('check', 'size-4 text-slate-400')}Available now</span><span class="font-semibold">${tradies.filter(t => t.available).length}</span></div>`,
          { action: `<a href="#/tradies" class="btn btn-ghost btn-sm">Tradies</a>` })}
        ${card('Needs attention', `
          <a href="#/requests" data-tab="read" class="kv rounded-md hover:text-teal-700"><span class="text-slate-600">Need a human read</span><span class="badge ${q.needs_human.length ? 'b-critical' : 'b-outline'}">${q.needs_human.length}</span></a>
          <div class="h-px bg-slate-100"></div>
          <a href="#/requests" data-tab="review" class="kv rounded-md hover:text-teal-700"><span class="text-slate-600">Waiting for a tier call</span><span class="badge ${q.review_band.length ? 'b-review' : 'b-outline'}">${q.review_band.length}</span></a>`)}
      </div>
    </div>`;
  $$('[data-tab]').forEach(a => a.onclick = () => { state.tab = a.dataset.tab; });
  linkRows();
}

function linkRows() {
  $$('tr[data-job]').forEach(tr => tr.onclick = e => { if (!e.target.closest('a')) location.hash = `#/job/${tr.dataset.job}`; });
}

// ---- Admin: repair requests ---------------------------------------------------------

async function renderRequests() {
  setHead('Repair Requests', 'Every report received, whatever its state.', searchBox('Search reference, community, issue'));
  const q = await api('/api/queue');
  const rows = [
    ...q.needs_human.map(r => ({ ...r, kind: 'read' })),
    ...q.review_band.map(j => ({ ...j, kind: 'review' })),
    ...q.ranked.map(j => ({ ...j, kind: j.status === 'Assigned' ? 'assigned' : 'ranked' })),
    ...q.completed.map(j => ({ ...j, kind: 'completed' })),
  ];
  const tabs = [['all', 'All'], ['read', 'Needs a read'], ['review', 'Review band'], ['ranked', 'Open'], ['assigned', 'Assigned'], ['completed', 'Completed']];
  const n = k => k === 'all' ? rows.length : rows.filter(r => r.kind === k).length;
  const shown = rows.filter(r => (state.tab === 'all' || r.kind === state.tab) && matches(r));

  page().innerHTML = `
    <div class="mb-4 flex flex-wrap gap-1 rounded-lg bg-slate-100 p-1 text-sm" role="tablist">
      ${tabs.map(([k, l]) => `<button role="tab" data-tabbtn="${k}" class="rounded-md px-3 py-1.5 font-medium ${state.tab === k ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-600 hover:text-slate-900'}">${l} <span class="text-slate-400">${n(k)}</span></button>`).join('')}
    </div>
    ${card('', tableWrap('<th>Reference</th><th>Community</th><th>Issue</th><th>Priority</th><th>Status</th><th>Waiting</th>',
      shown.map(r => r.kind === 'read'
        ? `<tr><td>${ref(r.report_id, false)}<div class="text-xs text-slate-400">${esc(r.source_file || '')}${r.source_item ? ' · item ' + r.source_item : ''}</div></td><td>${esc(r.community)}</td>
            <td class="max-w-[360px]"><div class="truncate" title="${esc(r.raw_text)}">${esc(r.raw_text)}</div><div class="text-xs text-red-600">${esc(r.reason)}</div></td>
            <td><span class="badge b-critical">Needs a read</span></td><td><span class="badge b-outline">With coordinator</span></td><td>${waiting(r.days_waiting)}</td></tr>`
        : `<tr class="row-link" data-job="${esc(r.job_id)}"><td>${ref(r.job_id)}<div class="text-xs text-slate-400">${esc(source(r))}</div></td><td>${esc(r.community)}</td>
            <td class="max-w-[360px]"><div class="truncate" title="${esc(r.fault)}">${esc(r.fault)}</div></td>
            <td>${priorityBadge(r)}</td><td>${statusBadge(r)}</td><td>${waiting(r.days_waiting)}</td></tr>`).join(''),
      'No requests match.', 6), { pad: false })}`;
  $$('[data-tabbtn]').forEach(b => b.onclick = () => { state.tab = b.dataset.tabbtn; renderRequests().then(refreshIcons); });
  wireSearch(() => renderRequests().then(refreshIcons));
  linkRows();
}

// ---- Admin: priority queue ------------------------------------------------------------

async function renderQueue() {
  setHead('Priority Queue', 'Ranked by safety, then urgency, then who reported first. Distance is shown but never used for order.',
    searchBox('Search reference, community, issue'));
  const q = await api('/api/queue');
  const ranked = q.ranked.filter(matches), band = q.review_band.filter(matches);
  page().innerHTML = `
    ${band.length ? `<div class="mb-4">${card('Review band', tableWrap('<th>Reference</th><th>Community</th><th>Issue</th><th>Waiting</th><th></th>',
      band.map(j => `<tr class="row-link" data-job="${esc(j.job_id)}"><td>${ref(j.job_id)}</td><td>${esc(j.community)}</td>
        <td class="max-w-[420px] truncate" title="${esc(j.fault)}">${esc(j.fault)}</td><td>${waiting(j.days_waiting)}</td>
        <td class="text-right"><span class="badge b-review">Tier call needed</span></td></tr>`).join(''), '', 5),
      { desc: 'Not on the fault list and no safety risk: held above the queue until a coordinator makes a tier call.', pad: false })}</div>` : ''}
    ${card('Ranked queue', tableWrap(
      '<th class="w-12">Rank</th><th>Reference</th><th>Community</th><th>Issue</th><th>Priority</th><th>NT category</th><th class="text-right">Score</th><th>Waiting</th><th>Distance</th><th>Status</th>',
      ranked.map(j => `<tr class="row-link" data-job="${esc(j.job_id)}"><td class="font-medium text-slate-500">${j.position}</td><td>${ref(j.job_id)}</td><td>${esc(j.community)}</td>
        <td class="max-w-[300px]"><div class="truncate" title="${esc(j.fault)}">${esc(j.fault)}</div><div class="truncate text-xs text-slate-400">${esc(j.decided_by)}${j.flags.length ? ` · ${j.flags.length} flag${j.flags.length > 1 ? 's' : ''}` : ''}</div></td>
        <td>${priorityBadge(j)}</td><td>${categoryBadge(j)}</td>
        <td class="text-right tabular-nums">${j.urgency_tally == null ? '<span class="text-slate-400">—</span>' : `${j.urgency_tally}<span class="text-xs text-slate-400"> (${j.base_points}+${j.severity_bump})</span>`}</td>
        <td class="whitespace-nowrap">${waiting(j.days_waiting)}</td><td class="whitespace-nowrap text-slate-500">${dist(j.distance)}</td><td>${statusBadge(j)}</td></tr>`).join(''),
      'No jobs match.', 10),
      { desc: 'Click a job to see why it sits where it does and to assign a tradie.', pad: false })}`;
  wireSearch(() => renderQueue().then(refreshIcons));
  linkRows();
}

// ---- Admin: communities and fairness ------------------------------------------------------

async function renderCommunities() {
  setHead('Communities', 'Who is waiting where. Use this to plan trips, not to reorder the queue.');
  const f = await api('/api/fairness');
  page().innerHTML = card('', tableWrap(
    '<th>Community</th><th>To nearest office</th><th class="text-right">Open jobs</th><th class="text-right">Safety jobs</th><th class="text-right">In review</th><th>Oldest waiting</th>',
    f.communities.map(c => `<tr><td class="font-medium">${esc(c.community)}</td><td class="text-slate-500">${dist(c.distance)}</td>
      <td class="text-right tabular-nums">${c.open}</td><td class="text-right">${c.safety ? `<span class="badge b-critical">${c.safety}</span>` : '<span class="text-slate-400">0</span>'}</td>
      <td class="text-right tabular-nums">${c.review_band}</td>
      <td>${c.oldest_days >= 14 ? `<span class="badge b-high">${waiting(c.oldest_days)}</span>` : waiting(c.oldest_days)}</td></tr>`).join(''),
    'No open jobs.', 6), { pad: false });
}

async function renderFairness() {
  setHead('Fairness Monitor', 'What an "efficient" nearest-first queue would do, compared with the real one.');
  const f = await api('/api/fairness');
  const s = f.summary;
  page().innerHTML = `
    <div class="mb-4 flex items-start gap-2 rounded-lg border border-teal-200 bg-teal-50 p-3 text-sm text-teal-900">${icon('scale', 'mt-0.5 size-4 shrink-0')}
      <p>The real queue is ordered by <b>safety → urgency → oldest report</b>. Distance is not in the sort key, so a remote tenant can't be pushed back for being far away. The table shows what would happen if it were.</p></div>
    <div class="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
      ${stat('Jobs pushed back', s.jobs_pushed_back, 'if sorted nearest-first', 'arrow-down', 'text-red-500')}
      ${stat('Places lost', s.places_lost, 'in total across those jobs', 'list-ordered')}
      ${stat('Safety jobs pushed back', s.safety_jobs_pushed_back, 'jobs with a safety risk', 'shield-alert', 'text-red-500')}
      ${stat('Worst case', s.worst ? `#${s.worst.position} → #${s.worst.nearest_first_position}` : '—', s.worst ? `${esc(s.worst.community)}: ${esc(s.worst.fault).slice(0, 48)}` : '', 'map-pin')}
    </div>
    <div class="mt-4">${card('Real queue vs nearest-first', tableWrap(
      '<th>Reference</th><th>Community</th><th>Issue</th><th>Priority</th><th>To nearest office</th><th class="text-right">Real</th><th class="text-right">Nearest-first</th><th class="text-right">Change</th>',
      f.what_if.map(r => `<tr class="row-link" data-job="${esc(r.job_id)}"><td>${ref(r.job_id)}</td><td>${esc(r.community)}</td>
        <td class="max-w-[280px] truncate" title="${esc(r.fault)}">${esc(r.fault)}</td><td>${priorityBadge(r)}</td><td class="text-slate-500">${dist(r.distance)}</td>
        <td class="text-right font-medium tabular-nums">${r.position}</td><td class="text-right tabular-nums text-slate-500">${r.nearest_first_position}</td>
        <td class="text-right tabular-nums">${r.change > 0 ? `<span class="font-medium text-red-600">↓ ${r.change}</span>` : r.change < 0 ? `<span class="text-emerald-600">↑ ${-r.change}</span>` : '<span class="text-slate-400">—</span>'}</td></tr>`).join(''),
      'No ranked jobs.', 8), { desc: 'A comparison only. The real queue never does this.', pad: false })}</div>`;
  linkRows();
}

// ---- Admin: reports ----------------------------------------------------------------------

async function renderReports() {
  setHead('Reports', 'Counts from the current queue, and a CSV export.',
    `<button id="clearBtn" class="btn btn-ghost text-red-600 hover:bg-red-50 hover:text-red-700">${icon('trash-2')}Clear all data</button>
     <button id="exportBtn" class="btn btn-outline">${icon('download')}Export CSV</button>`);
  const q = await api('/api/queue');
  const jobs = [...q.ranked, ...q.review_band];
  const tally = (list, key) => list.reduce((m, x) => (m[key(x)] = (m[key(x)] || 0) + 1, m), {});
  const block = (title, counts, order) => card(title, (order || Object.keys(counts).sort()).filter(k => counts[k]).map((k, i) =>
    `${i ? '<div class="h-px bg-slate-100"></div>' : ''}<div class="kv"><span class="text-slate-600">${esc(k)}</span><span class="font-semibold tabular-nums">${counts[k]}</span></div>`).join('')
    || '<p class="text-sm text-slate-500">Nothing yet.</p>');
  page().innerHTML = `<div class="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
      ${block('By priority', { ...tally(jobs, priorityOf), 'Needs a read': q.needs_human.length }, ['Critical', 'High', 'Medium', 'Low', 'Needs review', 'Needs a read'])}
      ${block('By NT category', tally(jobs, j => j.tier === 'dangerous' ? 'Emergency' : j.tier === 'standard' ? 'General' : 'Unlisted'))}
      ${block('By status', { ...tally(jobs, j => j.status), Completed: q.completed.length }, ['Open', 'Assigned', 'Completed'])}
      ${block('By community', tally(jobs, j => j.community))}
    </div>
    ${q.skipped.length ? `<div class="mt-4">${card('Files skipped at intake', q.skipped.map(s => `<p class="text-sm"><b>${esc(s.file)}</b> <span class="text-slate-500">${esc(s.reason)}</span></p>`).join(''))}</div>` : ''}`;
  $('#clearBtn').onclick = async () => {
    if (!confirm('Delete every report, job and decision? This cannot be undone.')) return;
    try { await api('/api/reset', { method: 'POST' }); toast('All data cleared.'); location.hash = '#/dashboard'; }
    catch (err) { toast(err.message); }
  };
  $('#exportBtn').onclick = () => {
    const header = ['Rank', 'Reference', 'Community', 'Issue', 'Priority', 'NT category', 'Score', 'Safety', 'Days waiting', 'Status', 'Tradie', 'Why here'];
    const lines = [...q.ranked, ...q.review_band].map(j => [j.position ?? 'review', j.job_id, j.community, j.fault, priorityOf(j),
      j.tier || 'unlisted', j.urgency_tally ?? '', safetyText(j), j.days_waiting, j.status, j.tradie || '', j.decided_by || 'awaiting tier call']);
    const csv = [header, ...lines].map(r => r.map(v => `"${String(v).replaceAll('"', '""')}"`).join(',')).join('\n');
    const a = Object.assign(document.createElement('a'), { href: URL.createObjectURL(new Blob([csv], { type: 'text/csv' })), download: 'fairfix_queue.csv' });
    a.click();
    URL.revokeObjectURL(a.href);
  };
}

// ---- Admin: job ---------------------------------------------------------------------------

function tradieOption(t, assignedId) {
  const checked = (assignedId ? t.assigned_here : t.recommended) ? 'checked' : '';
  return `<label class="flex cursor-pointer gap-3 rounded-md border p-3 text-sm ${t.recommended ? 'border-teal-300 bg-teal-50/60' : 'border-slate-200 hover:bg-slate-50'} ${!t.available || !t.qualified ? 'opacity-60' : ''}">
    <input type="radio" name="tradie_id" value="${t.id}" class="mt-0.5 accent-teal-700" ${checked} ${t.available ? '' : 'disabled'}>
    <div class="min-w-0 flex-1">
      <div class="flex flex-wrap items-center gap-2"><span class="font-medium">${esc(t.name)}</span>
        ${t.recommended ? '<span class="badge b-accent">Recommended</span>' : ''}${t.assigned_here ? '<span class="badge b-ok">Assigned</span>' : ''}</div>
      <div class="text-xs text-slate-500">${esc(t.trades.join(', '))} · ${esc(t.base)} · <span class="whitespace-nowrap">${esc(t.phone)}</span></div>
      <div class="mt-1 text-xs text-slate-500">${t.reasons.map(esc).join(' · ')}</div>
    </div></label>`;
}

async function renderJob(id, trade) {
  const j = await api(`/api/jobs/${encodeURIComponent(id)}` + (trade ? `?trade=${encodeURIComponent(trade)}` : ''));
  const refData = await reference();
  setHead(`Job ${j.job_id}`, `${j.community}${j.region ? ' · ' + j.region : ''} · ${source(j)}`,
    `<button class="btn btn-outline" onclick="history.length > 1 ? history.back() : (location.hash = '#/queue')">${icon('chevron-left')}Back</button>`);
  const untiered = j.urgency_tally == null;
  const d = j.logistics.distance;
  const needed = j.logistics.needed_trades;
  const tradeSource = j.logistics.trade_source;
  // Unlisted fault: the fault list gives no trade, so the coordinator chooses one.
  const tradePicker = tradeSource === 'pipeline' ? '' : `<div><label class="lbl" for="tradePick">Trade needed <span class="font-normal text-slate-400">(your call: this fault isn't on the list)</span></label>
      <select id="tradePick" name="trade" class="input"><option value="">Choose a trade to see a recommendation…</option>
        ${refData.trades.map(t => `<option ${t === needed[0] ? 'selected' : ''}>${esc(t)}</option>`).join('')}</select></div>`;
  const assignDesc = tradeSource === 'pipeline' ? `Needs: <b class="text-slate-700">${esc(needed.join(' or '))}</b> (from the fault list). `
    : tradeSource === 'coordinator' ? `Trade: <b class="text-slate-700">${esc(needed[0])}</b> (your call). `
    : 'Not on the fault list, so choose the trade first. ';

  page().innerHTML = `
    <div class="mb-4 flex flex-wrap items-center gap-2">
      ${j.position ? `<span class="badge b-outline">Rank ${j.position}</span>` : '<span class="badge b-review">Review band</span>'}
      ${priorityBadge(j)} ${categoryBadge(j)} ${statusBadge(j)}
      <span class="badge b-outline">${safetyText(j)}</span>
    </div>
    <div class="grid gap-4 lg:grid-cols-3">
      <div class="min-w-0 space-y-4 lg:col-span-2">
        ${card('Report', `<pre class="whitespace-pre-wrap break-words rounded-md bg-slate-50 p-3 font-mono text-[13px] text-slate-800">${esc(j.raw_text)}</pre>
          <dl class="mt-3 grid gap-x-4 gap-y-1 text-sm sm:grid-cols-[140px_1fr]">
            <dt class="text-slate-500">Source</dt><dd>${esc(source(j))}${j.source_file ? ` · ${esc(j.source_file)}${j.source_item ? ' item ' + j.source_item : ''}` : ''}</dd>
            <dt class="text-slate-500">Reported</dt><dd>${new Date(j.original_timestamp).toLocaleString()} <span class="text-slate-400">(${esc(j.timestamp_source || 'from report')})</span></dd>
            <dt class="text-slate-500">Read by</dt><dd>${esc(j.extractor)}</dd></dl>`,
          { desc: 'The exact text the model read. Name, phone, email and address are never sent to it.' })}

        ${card('What the model read', `
          <dl class="grid gap-x-4 gap-y-1 text-sm sm:grid-cols-[200px_1fr]">${j.facts.map(f => `<dt class="text-slate-500">${esc(f.field)}</dt><dd>${esc(f.value)}</dd>`).join('')}</dl>
          <div class="my-3 h-px bg-slate-100"></div>
          <div class="space-y-2">${j.spans.length ? j.spans.map(s => `<div class="flex items-start gap-3 text-sm">
            <span class="w-32 shrink-0 text-xs text-slate-500">${esc(s.field.replaceAll('_', ' '))}</span><span class="flex-1">"${esc(s.text)}"</span>
            ${s.verified ? `<span class="badge b-ok">${icon('check', 'size-3')}In report</span>` : `<span class="badge b-critical">${icon('x', 'size-3')}Not in report</span>`}</div>`).join('')
            : '<p class="text-sm text-slate-500">No quotes.</p>'}</div>`,
          { desc: 'Facts only, no scores. Code checks every quote word for word.' })}

        ${card(j.trace ? 'Why this position' : 'Why it is in the review band',
          (j.trace ? `<table class="tbl"><tbody>${j.trace.map((r, i) => `<tr><td class="w-36 text-slate-500 ${i ? '' : 'border-t-0'}">${esc(r.label)}</td>
            <td class="font-medium ${i ? '' : 'border-t-0'}">${esc(r.value)}</td><td class="text-slate-500 ${i ? '' : 'border-t-0'}">${esc(r.note)}</td></tr>`).join('')}</tbody></table>`
            : `<p class="text-sm">${esc(j.review_reason)}</p>`)
          + j.flags.map(f => `<div class="mt-2 flex gap-2 rounded-md border border-amber-200 bg-amber-50 p-2 text-sm text-amber-900">${icon('triangle-alert', 'mt-0.5 size-4 shrink-0')}${esc(f)}</div>`).join(''),
          { desc: 'Every number has a named source. The tier comes from the fault table, never the model.' })}

        ${card('Tenant messages', `
          <div class="rounded-md border-l-2 border-teal-600 bg-slate-50 p-3 text-sm" id="sms">${esc(j.sms)}</div>
          <button class="btn btn-ghost btn-sm mt-1" data-copy="sms">${icon('copy', 'size-3.5')}Copy SMS</button>
          <h3 class="mb-2 mt-3 text-sm font-medium">"Why is my repair here?"</h3>
          <div class="rounded-md border-l-2 border-teal-600 bg-slate-50 p-3 text-sm" id="why">${esc(j.why)}</div>
          <button class="btn btn-ghost btn-sm mt-1" data-copy="why">${icon('copy', 'size-3.5')}Copy answer</button>`,
          { desc: 'Templates only, so they can\'t invent a reason. Never shows other tenants or distance.' })}
      </div>

      <div class="space-y-4">
        ${untiered ? card('Tier call', `<form id="tierForm" class="space-y-3">
            <div><label class="lbl">Counts as</label><select name="fault_name" class="input" required><option value="">Choose a listed fault…</option>${refData.fault_names.map(n => `<option>${esc(n)}</option>`).join('')}</select></div>
            <div><label class="lbl">Reason</label><textarea name="reason" class="input" required placeholder="e.g. Hard-wired smoke alarm fault: treat as an electrical safety repair"></textarea></div>
            <button class="btn btn-primary w-full" type="submit">Save tier call</button></form>`,
          { desc: 'Not on the fault list. Choose what it counts as; it then enters the queue by the normal rules.' }) : ''}

        ${card('Assign a tradie', `<form id="assignForm" class="space-y-3">
            ${tradePicker}
            <div class="space-y-2">${j.recommendations.map(t => tradieOption(t, j.tradie_id)).join('')}</div>
            <div><label class="lbl">Note</label><textarea name="note" class="input" required placeholder="e.g. Mia is in Katherine Thursday, can reach Ngukurr Friday">${esc(j.status === 'Assigned' ? j.note || '' : '')}</textarea></div>
            <button class="btn btn-primary w-full" type="submit">${j.status === 'Assigned' ? 'Change assignment' : 'Assign tradie'}</button></form>`,
          { desc: `${assignDesc}Recommendation is advice for this job only: it never changes the queue order.` })}

        ${card('Close or reopen', `<form id="decisionForm" class="space-y-3">
            <textarea name="note" class="input" required placeholder="e.g. Repaired and tested; tenant confirmed"></textarea>
            <div class="flex gap-2">${j.status !== 'Completed' ? '<button class="btn btn-primary flex-1" type="submit" value="Completed">Mark completed</button>' : ''}
              ${j.status !== 'Open' ? '<button class="btn btn-outline flex-1" type="submit" value="Open">Reopen</button>' : ''}</div></form>`,
          { desc: 'Completed jobs leave the queue. Reopening clears the tradie.' })}

        ${card('Tenant follow-up', `<form id="followForm" class="space-y-3">
            <textarea name="text" class="input" required placeholder="e.g. water is getting worse, now dripping near the light switch"></textarea>
            <button class="btn btn-outline w-full" type="submit">${icon('message-square-reply')}Add follow-up and re-read</button></form>`,
          { desc: 'Re-read and re-ranked; the original report date is kept.' })}

        ${card('Logistics', `
          <div class="kv"><span class="text-slate-500">Nearest housing office</span><span>${d.km == null ? '<span class="text-slate-400">unknown (community not listed)</span>' : `${esc(d.office)} · ${dist(d)}`}</span></div>
          <div class="h-px bg-slate-100"></div>
          <div class="kv"><span class="text-slate-500">Same trip possible</span><span class="text-right">${j.logistics.shared_trip.length ? j.logistics.shared_trip.map(x => ref(x)).join(', ') : 'none'}</span></div>
          <div class="h-px bg-slate-100"></div>
          <p class="pt-2 text-sm text-slate-600">${esc(j.logistics.community_line)}</p>`,
          { desc: 'Display only. None of this changes the order.' })}

        ${card('Audit trail', j.audit.length ? `<ol class="space-y-3">${j.audit.map(a => `<li class="text-sm">
            <div class="text-xs text-slate-500">${new Date(a.at).toLocaleString()} · ${esc(a.actor)}</div>
            <div><span class="font-medium">${esc(a.action)}</span>${a.note ? ` <span class="text-slate-600">— ${esc(a.note)}</span>` : ''}</div></li>`).join('')}</ol>`
          : '<p class="text-sm text-slate-500">Nothing yet.</p>')}
      </div>
    </div>`;

  $$('[data-copy]').forEach(b => b.onclick = async () => {
    try { await navigator.clipboard.writeText($('#' + b.dataset.copy).textContent); toast('Copied.'); }
    catch { toast('Copy failed: select the text instead.'); }
  });
  submitForm('#tierForm', f => api(`/api/jobs/${id}/tier`, { method: 'POST', body: f }), 'Tier call saved. The job is now in the queue.');
  submitForm('#assignForm', f => {
    if (!f.tradie_id) throw new Error('Choose a tradie');
    return api(`/api/jobs/${id}/assign`, { method: 'POST', body: { tradie_id: Number(f.tradie_id), note: f.note, trade: f.trade || null } });
  }, 'Tradie assigned.');
  const pick = $('#tradePick');
  if (pick) pick.onchange = async () => {
    const note = $('#assignForm textarea[name=note]').value;
    await renderJob(id, pick.value);
    refreshIcons();
    $('#assignForm textarea[name=note]').value = note;  // keep what was typed
    $('#tradePick').focus();
  };
  submitForm('#decisionForm', (f, btn) => api(`/api/jobs/${id}/decision`, { method: 'POST', body: { status: btn.value, note: f.note } }), 'Saved.');
  submitForm('#followForm', async f => {
    const r = await api(`/api/jobs/${id}/followup`, { method: 'POST', body: f });
    toast(r.status === 'ok' ? 'Follow-up read and the job re-ranked.' : 'Follow-up saved, but it needs a human read (flagged on this job).');
  });
}

function submitForm(sel, action, done) {
  const form = $(sel);
  if (!form) return;
  form.onsubmit = async e => {
    e.preventDefault();
    const btn = e.submitter || $('button[type=submit]', form);
    btn.disabled = true;
    try {
      await action(Object.fromEntries(new FormData(form)), btn);
      if (done) toast(done);
      await route();
    } catch (err) { toast(err.message); }
    finally { btn.disabled = false; }
  };
}

// ---- Admin: tradies ------------------------------------------------------------------------

async function renderTradies() {
  setHead('Tradies', 'Who can be sent, where they are based, and what they are already doing.');
  const [tradies, refData] = await Promise.all([api('/api/tradies'), reference()]);
  page().innerHTML = `
    ${card('', tableWrap('<th>Name</th><th>Trades</th><th>Home base</th><th>Phone</th><th>Assigned jobs</th><th class="text-right">Availability</th>',
      tradies.map(t => `<tr><td class="font-medium">${esc(t.name)}</td><td>${t.trades.map(x => `<span class="badge b-outline mr-1">${esc(x)}</span>`).join('')}</td>
        <td>${esc(t.base)}</td><td class="whitespace-nowrap text-slate-500">${esc(t.phone)}</td>
        <td>${t.jobs.length ? t.jobs.map(x => `${ref(x.job_id)} <span class="text-xs text-slate-400">${esc(x.community)}</span>`).join('<br>') : '<span class="text-slate-400">none</span>'}</td>
        <td class="text-right"><button class="btn btn-outline btn-sm" data-toggle="${t.id}">
          <span class="size-2 rounded-full ${t.available ? 'bg-emerald-500' : 'bg-slate-300'}"></span>${t.available ? 'Available' : 'Away'}</button></td></tr>`).join(''),
      'No tradies.', 6), { pad: false })}
    <div class="mt-4 grid gap-4 lg:grid-cols-2">
      ${card('Add a tradie', `<form id="tradieForm" class="space-y-3">
          <div><label class="lbl">Name</label><input name="name" class="input" required></div>
          <div><span class="lbl">Trades</span><div class="flex flex-wrap gap-x-4 gap-y-2 text-sm">${refData.trades.map(t => `<label class="flex items-center gap-2"><input type="checkbox" name="trades" value="${esc(t)}" class="accent-teal-700">${esc(t)}</label>`).join('')}</div></div>
          <div class="grid gap-3 sm:grid-cols-2">
            <div><label class="lbl">Home base</label><input name="base" list="bases" class="input" required placeholder="e.g. Katherine"></div>
            <div><label class="lbl">Phone</label><input name="phone" class="input"></div>
          </div>
          <datalist id="bases">${refData.communities.map(c => `<option>${esc(c)}</option>`).join('')}</datalist>
          <button class="btn btn-primary" type="submit">${icon('plus')}Add tradie</button></form>`)}
      ${card('How recommendations work', `<ul class="list-disc space-y-1.5 pl-5 text-sm text-slate-600">
          <li>Only tradies listed for the fault's trade are recommended (a blocked toilet needs a plumber).</li>
          <li>Tradies marked away can't be assigned.</li>
          <li>Someone already assigned in the same community comes first: one trip, two jobs.</li>
          <li>Then the nearest home base, then the lightest workload.</li>
          <li>It only picks <b>who</b> goes, never <b>which job</b> goes first. The admin decides.</li></ul>`)}
    </div>`;
  $$('[data-toggle]').forEach(b => b.onclick = async () => {
    try { await api(`/api/tradies/${b.dataset.toggle}/availability`, { method: 'POST' }); await renderTradies(); refreshIcons(); }
    catch (err) { toast(err.message); }
  });
  $('#tradieForm').onsubmit = async e => {
    e.preventDefault();
    const fd = new FormData(e.target);
    const body = { name: fd.get('name'), base: fd.get('base'), phone: fd.get('phone'), trades: fd.getAll('trades') };
    if (!body.trades.length) return toast('Tick at least one trade');
    try { await api('/api/tradies', { method: 'POST', body }); toast('Tradie added.'); await renderTradies(); refreshIcons(); }
    catch (err) { toast(err.message); }
  };
}

// ---- Officer: new report -------------------------------------------------------------------

async function renderIntake() {
  setHead('Upload report', 'Upload a housing document or record a call. Each report runs through all six stages and the result appears below.');
  const refData = await reference();
  const now = new Date(Date.now() - new Date().getTimezoneOffset() * 60000).toISOString().slice(0, 16);
  page().innerHTML = `
    <div class="mb-4 flex items-start gap-2 rounded-lg border border-slate-200 bg-white p-3 text-sm text-slate-600">${icon('shield-alert', 'mt-0.5 size-4 shrink-0 text-teal-700')}
      Only the fault text goes to the model. The tenant's own urgency rating, name, phone, email and address are kept out of it.</div>
    <div class="grid gap-4 lg:grid-cols-3">
      <div class="min-w-0 lg:col-span-2">${card('Phone call or message', `<form id="textForm" class="space-y-3">
          <div class="grid gap-3 sm:grid-cols-2">
            <div><label class="lbl">Source</label><select name="source_tag" class="input"><option value="officer">Officer (phone call)</option><option value="tenant_direct">Tenant (SMS, email, web)</option></select></div>
            <div><label class="lbl">Community</label><input name="community" list="communities" class="input" required placeholder="e.g. Wadeye"></div>
            <div><label class="lbl">When the tenant reported it</label><input type="datetime-local" name="reported_at" value="${now}" class="input"></div>
            <div><label class="lbl">Tenant reference <span class="font-normal text-slate-400">(optional)</span></label><input name="tenant_ref" class="input" placeholder="e.g. tenancy number"></div>
          </div>
          <datalist id="communities">${refData.communities.map(c => `<option>${esc(c)}</option>`).join('')}</datalist>
          <div><label class="lbl">Tenant's words</label><textarea name="message" class="input" required placeholder="e.g. dunny won't go down, going to my sister's place"></textarea></div>
          <button class="btn btn-primary" type="submit">Submit report</button></form>`, { desc: 'Type the tenant\'s words as they said them.' })}</div>
      ${card('Upload a form', `<form id="uploadForm" class="space-y-3">
          <input type="file" name="file" accept=".pdf,.txt" required class="block w-full text-sm text-slate-600 file:mr-3 file:rounded-md file:border file:border-slate-200 file:bg-white file:px-3 file:py-1.5 file:text-sm file:font-medium hover:file:bg-slate-50">
          <button class="btn btn-outline w-full" type="submit">${icon('upload')}Upload and read</button></form>`,
        { desc: 'A GEHSF03 PDF (each table row becomes its own job) or a .txt report.' })}
    </div>
    <div id="intakeResult" class="mt-4"></div>`;

  const show = res => { $('#intakeResult').innerHTML = res.reports.map(stageResult).join(''); refreshIcons(); };
  $('#textForm').onsubmit = async e => {
    e.preventDefault();
    const f = Object.fromEntries(new FormData(e.target));
    if (!f.reported_at) delete f.reported_at;
    try { show(await api('/api/reports/text', { method: 'POST', body: f })); e.target.message.value = ''; toast('Report submitted.'); }
    catch (err) { toast(err.message); }
  };
  $('#uploadForm').onsubmit = async e => {
    e.preventDefault();
    try { show(await api('/api/reports/upload', { method: 'POST', body: new FormData(e.target) })); e.target.reset(); toast('Form read and submitted.'); }
    catch (err) { toast(err.message); }
  };
}

function step(n, title, body, tone = 'b-accent') {
  return `<li class="relative flex gap-3 pb-5 last:pb-0">
    <span class="absolute left-3 top-7 h-[calc(100%-1.75rem)] w-px bg-slate-200"></span>
    <span class="badge ${tone} relative z-10 grid size-6 shrink-0 place-items-center rounded-full p-0">${n}</span>
    <div class="min-w-0 flex-1"><div class="text-sm font-semibold">${title}</div><div class="mt-1 space-y-1.5 text-sm text-slate-600">${body}</div></div></li>`;
}

function stageResult(r) {
  const s1 = r.stage1, s2 = r.stage2, admin = state.me?.role === 'admin';
  const ok = s2.status === 'ok';
  const head = `<div class="flex flex-wrap items-center justify-between gap-2 border-b border-slate-100 px-4 py-3">
      <div><div class="text-sm font-semibold">Ticket ${esc(s1.request_id)}</div>
        <div class="text-xs text-slate-500">${esc(s1.community)}${s1.region ? ' · ' + esc(s1.region) : ''}${s1.source_file ? ` · ${esc(s1.source_file)}${s1.source_item != null ? ' item ' + s1.source_item : ''}` : ''}</div></div>
      ${ok ? '<span class="badge b-ok">Ranked</span>' : '<span class="badge b-critical">Needs a human read</span>'}</div>`;
  const stage1 = step(1, 'Intake', `
      <div>Ticket <b class="text-slate-900">${esc(s1.request_id)}</b> · tenant ${esc(s1.tenant_id)} · ${s1.source_tag === 'officer' ? 'officer (phone call)' : 'tenant (form or message)'}</div>
      <div>Reported ${new Date(s1.original_report_timestamp).toLocaleString()} <span class="text-slate-400">(${esc(s1.timestamp_source || 'from report')})</span></div>
      <div class="flex gap-1.5 text-xs text-slate-500">${icon('shield-alert', 'mt-0.5 size-3.5 shrink-0 text-teal-700')}${esc(s1.privacy)}</div>
      <pre class="whitespace-pre-wrap break-words rounded-md bg-slate-50 p-2.5 font-mono text-xs text-slate-800">${esc(s1.raw_text)}</pre>`);
  const stage2 = step(2, 'Extraction (model reads)', ok
      ? `<div><span class="text-slate-500">Read by</span> ${esc(s2.extractor)}</div>` + s2.faults.map(f => `<div class="rounded-md border border-slate-200 p-2.5">
          <div class="font-medium text-slate-900">"${esc(f.fault_description)}"</div>
          <div class="mt-1 grid gap-x-3 text-xs sm:grid-cols-2">${f.facts.map(x => `<div><span class="text-slate-500">${esc(x.field)}:</span> ${esc(x.value)}</div>`).join('')}</div></div>`).join('')
      : `<div class="text-red-700">${esc(s2.problem)}</div><div class="text-xs text-slate-500">The report is kept and listed under "Needs a human read". Nothing is guessed.</div>`,
    ok ? 'b-accent' : 'b-critical');
  const jobs = r.jobs.map(j => {
    const s6 = j.stage6, s5 = j.stage5, s4 = j.stage4;
    return `<div class="mt-4 rounded-lg border border-slate-200">
      <div class="flex flex-wrap items-center justify-between gap-2 border-b border-slate-100 px-3 py-2">
        <div class="text-sm"><span class="font-semibold">Job ${esc(j.job_id)}</span> <span class="text-slate-500">· ${esc(j.fault)}</span></div>
        ${admin ? `<a href="#/job/${esc(j.job_id)}" class="btn btn-ghost btn-sm">Open job ${icon('arrow-up-right', 'size-3.5')}</a>` : ''}</div>
      <ol class="p-3">
        ${step(3, 'Verification', j.stage3.length ? j.stage3.map(sp => `<div class="flex items-start gap-2"><span class="w-28 shrink-0 text-xs text-slate-500">${esc(sp.field.replaceAll('_', ' '))}</span>
            <span class="flex-1">"${esc(sp.text)}"</span>${sp.verified ? '<span class="badge b-ok">In report</span>' : '<span class="badge b-critical">Not in report</span>'}</div>`).join('') : 'No quotes to check.')}
        ${step(4, 'Evaluation (urgency and safety)', `
            <div>${s4.tier ? `<b class="text-slate-900">${esc(s4.tier_entry)}</b> → ${esc(s4.tier)} <span class="text-slate-400">(${esc(s4.tier_sources)})</span>` : 'Not on the fault list: no tier'}</div>
            ${s4.urgency_tally != null ? `<div>Urgency score <b class="text-slate-900">${s4.urgency_tally}</b> = ${s4.base_points} + ${s4.severity_bump} <span class="text-slate-400">(${esc(s4.tally_reasons.join('; '))})</span></div>` : ''}
            <div>Safety level <b class="text-slate-900">${s4.safety_level}</b> <span class="text-slate-400">(${esc(s4.safety_reason)})</span></div>`)}
        ${step(5, 'Logistics', `
            <div>Required trade: <b class="text-slate-900">${s5.required_trades.length ? esc(s5.required_trades.join(' or ')) : 'not confirmed yet'}</b></div>
            <div>Nearest housing office: ${s5.distance.km == null ? '<span class="text-slate-400">unknown (community not listed)</span>' : `${esc(s5.distance.office)} · ${dist(s5.distance)} straight line`} <span class="text-xs text-slate-400">· not used for order</span></div>
            <div>Suggested tradie: ${esc(s5.recommended_tradie || (s5.required_trades.length ? 'none available' : 'after a tier call sets the trade'))}</div>`)}
        ${step(6, 'Ranking and explanation', `
            <div class="flex flex-wrap items-center gap-2">${s6.in_review_band ? '<span class="badge b-review">Review band</span>' : `<span class="badge b-outline">Rank ${s6.position} of ${s6.queue_length}</span>`}
              ${priorityBadge(j)} ${categoryBadge(j)}</div>
            <div class="text-slate-500">${esc(s6.decided_by)}</div>
            <div class="rounded-md border-l-2 border-teal-600 bg-slate-50 p-2.5 text-slate-800"><div class="mb-1 text-xs font-medium text-slate-500">Why is my repair here? (tenant answer)</div>${esc(s6.why)}</div>
            <div class="rounded-md border-l-2 border-slate-300 bg-slate-50 p-2.5 text-slate-800"><div class="mb-1 text-xs font-medium text-slate-500">Tenant SMS</div>${esc(s6.sms)}</div>`)}
      </ol></div>`;
  }).join('');
  return `<section class="card mt-4">${head}<div class="p-4"><ol>${stage1}${stage2}</ol>${jobs}</div></section>`;
}

// ---- Officer: my submissions -------------------------------------------------------------------

async function renderMine() {
  setHead('My submissions', 'Reports you recorded, and their progress. Priorities are set by the coordinator.',
    `<a href="#/new" class="btn btn-primary">${icon('upload')}Upload report</a>`);
  const rows = await api('/api/my-reports');
  page().innerHTML = card('', tableWrap('<th>Reference</th><th>Community</th><th>Submitted</th><th>Report</th><th>Progress</th>',
    rows.map(r => `<tr><td class="font-medium">${r.jobs.length ? r.jobs.map(x => esc(x.job_id)).join('<br>') : esc(r.report_id)}</td><td>${esc(r.community)}</td>
      <td class="whitespace-nowrap text-slate-500">${new Date(r.submitted_at).toLocaleString()}</td><td class="max-w-[360px]"><div class="truncate" title="${esc(r.raw_text)}">${esc(r.raw_text)}</div></td>
      <td>${!r.read ? '<span class="badge b-critical">Needs a human read</span>' : r.jobs.map(x => x.status === 'Assigned' ? `<span class="badge b-accent">Assigned · ${esc(x.tradie)}</span>`
        : x.status === 'Completed' ? '<span class="badge b-ok">Completed</span>' : '<span class="badge b-outline">With the coordinator</span>').join('<br>')}</td></tr>`).join(''),
    'Nothing submitted yet. Use "Upload report".', 5), { pad: false });
}

// ---- How it works --------------------------------------------------------------------------------

function renderHow() {
  setHead('How it works', 'Six stages. One reads with a language model; every stage that decides is plain code.');
  const stages = [
    ['Intake', 'Code', 'Python reads the form or message, removes personal details and the tenant\'s own urgency label, and stamps the report time.'],
    ['Extraction', 'AI', 'The model reads the tenant\'s words and returns facts with exact quotes: which listed fault, any alternative, any hazard. No numbers.'],
    ['Verification', 'Code', 'Every quote is checked word for word against the report. A made-up quote is caught and can never lower a score.'],
    ['Evaluation', 'Code', 'The tier is looked up in the NT fault table (Residential Tenancies Act s63). Score = 3 or 2, +1 unless another working one is named. Safety level 0–2 from the hazard.'],
    ['Logistics', 'Code', 'Distance, shared trips, community waiting times and tradie suggestions for the coordinator. They never change the order.'],
    ['Ranking', 'Code', 'Sorted by safety, then score, then oldest report. Each job gets a written reason, a coordinator view and a tenant message.'],
  ];
  page().innerHTML = `
    <div class="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">${stages.map(([t, k, p], i) => `<div class="card card-body">
      <div class="flex items-center justify-between"><span class="text-sm font-semibold">${i + 1}. ${t}</span>
        <span class="badge ${k === 'AI' ? 'b-high' : 'b-outline'}">${k}</span></div><p class="mt-2 text-sm text-slate-600">${p}</p></div>`).join('')}</div>
    <div class="mt-4 grid gap-4 lg:grid-cols-3">
      ${card('What the model never sees', '<ul class="list-disc space-y-1 pl-5 text-sm text-slate-600"><li>Tier labels or point values</li><li>Scoring rules or rank positions</li><li>Other tenants\' jobs</li><li>The tenant\'s own urgency rating</li><li>Name, phone, email or address</li></ul>')}
      ${card('Guarantees checked by tests', '<ul class="list-disc space-y-1 pl-5 text-sm text-slate-600"><li>No unflagged job ranks above a flagged one</li><li>Equal safety and score: oldest report first</li><li>Changing any distance never changes any position</li><li>How a tenant writes doesn\'t lower their score</li><li>A follow-up never resets the report date</li></ul>')}
      ${card('Who does what', '<ul class="list-disc space-y-1 pl-5 text-sm text-slate-600"><li><b>Officer</b>: records calls and messages, uploads forms, gives the tenant a reference.</li><li><b>Admin</b>: tier calls, assigns tradies, closes jobs. Every decision has a note in the audit trail.</li><li><b>The system</b>: ranks and explains. It never assigns a tradie or overrides the admin.</li></ul>')}
    </div>`;
}

// ---- Router ------------------------------------------------------------------------------

const PAGES = {
  dashboard: renderDashboard, requests: renderRequests, queue: renderQueue, communities: renderCommunities,
  fairness: renderFairness, tradies: renderTradies, reports: renderReports, new: renderIntake, mine: renderMine, how: renderHow,
};

async function route() {
  if (!state.me) return;
  const [, name = '', id] = location.hash.split('/');
  if (!allowed(name)) { location.hash = `#/${NAV[state.me.role][0][0]}`; return; }
  const active = name === 'job' ? 'queue' : name;
  $$('.nav-link[data-route]').forEach(a => a.classList.toggle('active', a.dataset.route === active));
  const label = (NAV[state.me.role].find(([r]) => r === active) || [, name === 'how' ? 'How it works' : ''])[1];
  $('#crumb').innerHTML = name === 'job' ? `<a href="#/queue" class="hover:text-slate-900">Priority Queue</a> / <span class="text-slate-900">${esc(decodeURIComponent(id || ''))}</span>`
    : `<span class="text-slate-900">${esc(label)}</span>`;
  try {
    if (name === 'job' && id) await renderJob(decodeURIComponent(id));
    else await PAGES[name]();
  } catch (err) {
    if (state.me) page().innerHTML = `<div class="rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm text-amber-900">Could not load this page: ${esc(err.message)}</div>`;
  }
  refreshIcons();
}

window.addEventListener('hashchange', route);
api('/api/me').then(me => { state.me = me; showApp(); }).catch(() => showLogin());
