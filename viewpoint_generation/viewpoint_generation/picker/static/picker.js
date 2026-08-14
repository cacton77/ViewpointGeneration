/* Part picker front end.
 *
 * Renders the catalog as a card grid, filters it locally (the catalog is small
 * enough that round-tripping every keystroke to the server buys nothing), and
 * drives selection and sync through the JSON API.
 */

const state = {
  parts: [],
  selectedId: null,
  openId: null,
  busy: new Set(),
  // Plan chosen in the strip for the open part. '' means "model only";
  // null means "whichever plan the catalog considers current".
  chosenPlanId: null,
  plans: [],
};

const el = {
  grid: document.getElementById('grid'),
  empty: document.getElementById('empty'),
  search: document.getElementById('search'),
  maturity: document.getElementById('maturity'),
  stateFilter: document.getElementById('state-filter'),
  statusLine: document.getElementById('status-line'),
  syncBtn: document.getElementById('sync-btn'),
  rosBadge: document.getElementById('ros-badge'),
  selection: document.getElementById('selection'),
  toast: document.getElementById('toast'),
  detail: document.getElementById('detail'),
  detailBody: document.getElementById('detail-body'),
  detailClose: document.getElementById('detail-close'),
};

function toast(message, kind) {
  el.toast.textContent = message || '';
  el.toast.className = 'toast' + (kind ? ' ' + kind : '');
  if (message) {
    clearTimeout(toast._timer);
    toast._timer = setTimeout(() => {
      el.toast.textContent = '';
      el.toast.className = 'toast';
    }, 8000);
  }
}

async function api(path, options) {
  const response = await fetch(path, options);
  let payload = {};
  try {
    payload = await response.json();
  } catch (e) {
    payload = { message: `${response.status} ${response.statusText}` };
  }
  if (!response.ok) {
    throw new Error(payload.message || `Request failed (${response.status})`);
  }
  return payload;
}

function visibleParts() {
  const needle = el.search.value.trim().toLowerCase();
  const maturity = el.maturity.value;
  const cardState = el.stateFilter.value;
  return state.parts.filter((part) => {
    if (maturity && (part.maturity || '').toUpperCase() !== maturity) return false;
    if (cardState && part.state !== cardState) return false;
    if (!needle) return true;
    return [part.title, part.part_number, part.description, part.revision]
      .some((field) => (field || '').toLowerCase().includes(needle));
  });
}

function card(part) {
  const node = document.createElement('article');
  node.className = 'card';
  if (part.eng_item_id === state.selectedId) node.classList.add('selected');
  if (state.busy.has(part.eng_item_id)) node.classList.add('busy');
  node.dataset.id = part.eng_item_id;

  const thumb = part.has_thumbnail
    ? `<img src="/thumb/${encodeURIComponent(part.eng_item_id)}" alt="" loading="lazy">`
    : '<span class="placeholder">▣</span>';

  const rev = part.revision ? `${part.revision}` : '';
  const maturity = (part.maturity || '').replace('_', ' ');

  node.innerHTML = `
    <div class="thumb">${thumb}</div>
    <div class="card-body">
      <span class="card-title" title="${escapeHtml(part.title)}">${escapeHtml(part.title)}</span>
      <span class="card-meta">${escapeHtml(part.part_number || '')}</span>
      <span class="card-meta">${escapeHtml(rev)} ${escapeHtml(maturity)}</span>
      <div class="card-states">
        <span class="state state-${part.state}">${escapeHtml(part.state_label)}</span>
        <span class="plan plan-${part.plan_status}">${planLabel(part.plan_status)}</span>
      </div>
    </div>`;

  node.addEventListener('click', () => openDetail(part.eng_item_id));
  return node;
}

function planLabel(status) {
  switch (status) {
    case 'CURRENT': return 'Plan';
    case 'DOWNLOADED':
    case 'UPDATED_FROM_REMOTE': return 'Plan (remote)';
    case 'STALE': return 'Stale plan';
    default: return 'No plan';
  }
}

function escapeHtml(value) {
  return String(value == null ? '' : value).replace(/[&<>"']/g, (c) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[c]));
}

/* The plan strip shows a lot of timestamps in a little space, so they are
 * rendered as elapsed time and carry the absolute value as a tooltip. */
function relativeTime(iso) {
  if (!iso) return '—';
  const then = Date.parse(iso.endsWith('Z') || iso.includes('+') ? iso : iso + 'Z');
  if (Number.isNaN(then)) return iso;
  const seconds = Math.max(0, (Date.now() - then) / 1000);
  const scales = [
    [60, 's'], [3600, 'm', 60], [86400, 'h', 3600], [604800, 'd', 86400],
  ];
  for (const [limit, unit, divisor] of scales) {
    if (seconds < limit) {
      return divisor ? `${Math.floor(seconds / divisor)}${unit} ago` : 'just now';
    }
  }
  return `${Math.floor(seconds / 604800)}w ago`;
}

function planCounts(plan) {
  return [
    plan.num_regions ? `${plan.num_regions}r` : null,
    plan.num_clusters ? `${plan.num_clusters}c` : null,
    plan.num_viewpoints ? `${plan.num_viewpoints}v` : null,
  ].filter(Boolean).join(' · ') || 'empty';
}

function syncLabel(plan) {
  switch (plan.upload_status) {
    case 'synced': return '3DX';
    case 'uploading': return 'uploading…';
    case 'failed': return 'upload failed';
    default: return 'local';
  }
}

function render() {
  const parts = visibleParts();
  el.grid.replaceChildren(...parts.map(card));
  el.empty.classList.toggle('hidden', parts.length > 0);
  el.statusLine.textContent =
    `${parts.length} of ${state.parts.length} parts · `
    + `${state.parts.filter((p) => p.step_cached).length} ready`;
}

async function loadParts() {
  const data = await api('/api/parts');
  state.parts = data.parts;
  render();
  if (state.openId) openDetail(state.openId, true);
}

async function loadStatus() {
  try {
    const status = await api('/api/status');
    el.rosBadge.textContent = status.ros ? 'ROS connected' : 'ROS offline';
    el.rosBadge.className = 'badge ' + (status.ros ? 'badge-ok' : 'badge-muted');
    el.rosBadge.title = status.ros
      ? 'Selections are published on /catalog/part_selected'
      : 'ROS is not running; selections cache the STEP but are not announced';
    if (!status.configured) {
      toast('3DX credentials are not configured; only cached parts are available.',
            'error');
    }
  } catch (e) {
    el.rosBadge.textContent = 'offline';
    el.rosBadge.className = 'badge badge-bad';
  }
}

async function openDetail(engItemId, keepScroll) {
  const switchingPart = state.openId !== engItemId;
  state.openId = engItemId;
  if (switchingPart) {
    state.chosenPlanId = null;
    state.plans = [];
  }
  const part = state.parts.find((p) => p.eng_item_id === engItemId);
  if (!part) return;

  el.detail.classList.remove('hidden');
  el.detailBody.innerHTML = `
    <h2>${escapeHtml(part.title)}</h2>
    <div class="card-meta">${escapeHtml(part.part_number)} · rev ${escapeHtml(part.revision)}
      · ${escapeHtml((part.maturity || '').replace('_', ' '))}
      · ${escapeHtml(part.collab_space)}</div>
    <h3>Model</h3>
    <dl>
      <dt>STEP</dt><dd>${part.step_cached ? 'cached locally' :
        (part.step_available ? 'available on 3DX, not downloaded' : 'none published')}</dd>
      <dt>Sync</dt><dd>${escapeHtml(part.sync_status)} · last ${escapeHtml(part.last_synced)}</dd>
      <dt>Item id</dt><dd>${escapeHtml(part.eng_item_id)}</dd>
    </dl>
    <div class="plans-head">
      <h3>Inspection plans</h3>
      <button class="btn btn-small" id="adopt-btn"
              title="Record plan files already on disk that the catalog never captured">
        Scan for files</button>
    </div>
    <div id="plan-strip" class="plan-strip">loading…</div>
    <div id="plan-box" class="plan-box"></div>
    <h3>Inspection history</h3>
    <div id="runs-box">loading…</div>
    <div class="detail-actions">
      <button class="btn btn-primary" id="select-btn">Load part</button>
    </div>`;

  document.getElementById('select-btn')
    .addEventListener('click', () => selectPart(engItemId));
  document.getElementById('adopt-btn')
    .addEventListener('click', () => adoptPlans(engItemId));

  if (!keepScroll) el.detail.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

  await loadPlans(engItemId);

  try {
    const data = await api(`/api/parts/${encodeURIComponent(engItemId)}/runs`);
    const box = document.getElementById('runs-box');
    if (!box) return;
    if (!data.runs.length) {
      box.innerHTML = '<em>No inspections recorded for this part.</em>';
      return;
    }
    box.innerHTML = `
      <table class="runs">
        <thead><tr><th>Run</th><th>Result</th><th>Anomalies</th>
          <th>Max score</th><th>Upload</th></tr></thead>
        <tbody>${data.runs.map((run) => `
          <tr>
            <td>${escapeHtml(run.started_at)}</td>
            <td class="result-${escapeHtml(run.overall_result || '')}">
              ${escapeHtml(run.overall_result || run.status || '—')}</td>
            <td>${run.anomalies_found ?? '—'}</td>
            <td>${run.max_anomaly_score != null ? run.max_anomaly_score.toFixed(2) : '—'}</td>
            <td>${escapeHtml(run.upload_status || '—')}</td>
          </tr>`).join('')}</tbody>
      </table>`;
  } catch (e) { /* the panel keeps its loading text */ }
}

async function loadPlans(engItemId) {
  try {
    const data = await api(`/api/parts/${encodeURIComponent(engItemId)}/plans`);
    state.plans = data.plans;
    // Default the selection to the catalog's current plan, so the Load button
    // does the same thing it always did until the operator picks otherwise.
    if (state.chosenPlanId === null && data.current_plan_id) {
      state.chosenPlanId = data.current_plan_id;
    }
  } catch (e) {
    state.plans = [];
  }
  renderPlanStrip();
}

function renderPlanStrip() {
  const strip = document.getElementById('plan-strip');
  if (!strip) return;

  const chips = [`
    <button class="plan-chip chip-none ${state.chosenPlanId === '' ? 'selected' : ''}"
            data-plan-id="" title="Load the model without a plan">
      <span class="chip-stage">model only</span>
      <span class="chip-counts">no plan</span>
      <span class="chip-time">&nbsp;</span>
    </button>`];

  for (const plan of state.plans) {
    const classes = [
      'plan-chip', `stage-${plan.stage}`,
      plan.plan_id === state.chosenPlanId ? 'selected' : '',
      plan.available ? '' : 'missing',
      plan.stale ? 'stale' : '',
    ].filter(Boolean).join(' ');
    const flags = [
      plan.is_current ? '<span class="chip-flag" title="Current plan">★</span>' : '',
      plan.stale ? '<span class="chip-flag warn" title="Generated against an earlier CAD revision">⚠</span>' : '',
      plan.available ? '' : '<span class="chip-flag warn" title="File is missing on disk">✕</span>',
    ].join('');
    chips.push(`
      <button class="${classes}" data-plan-id="${escapeHtml(plan.plan_id)}"
              title="${escapeHtml(plan.file_path)}">
        <span class="chip-stage">${escapeHtml(plan.stage)}${flags}</span>
        <span class="chip-counts">${escapeHtml(planCounts(plan))}</span>
        <span class="chip-time" title="${escapeHtml(plan.generated_at || '')}">
          ${escapeHtml(relativeTime(plan.generated_at))}</span>
        <span class="chip-sync sync-${escapeHtml(plan.upload_status || 'local')}">
          ${escapeHtml(syncLabel(plan))}</span>
      </button>`);
  }

  strip.innerHTML = state.plans.length
    ? chips.join('')
    : chips.join('') + '<span class="plan-empty">No plans recorded yet — '
      + 'run the pipeline, or use "Scan for files".</span>';

  strip.querySelectorAll('.plan-chip').forEach((chip) => {
    chip.addEventListener('click', () => {
      state.chosenPlanId = chip.dataset.planId;
      renderPlanStrip();
    });
  });

  renderPlanBox();
}

function renderPlanBox() {
  const box = document.getElementById('plan-box');
  if (!box) return;

  const button = document.getElementById('select-btn');
  const plan = state.plans.find((p) => p.plan_id === state.chosenPlanId);

  if (button) {
    if (state.chosenPlanId === '') button.textContent = 'Load model only';
    else if (plan) button.textContent = `Load with ${plan.stage} plan`;
    else button.textContent = 'Load part';
  }

  if (!plan) {
    box.innerHTML = state.chosenPlanId === ''
      ? '<em>The model will be loaded with no regions or viewpoints.</em>'
      : '';
    return;
  }

  const uploadable = ['local', 'failed'].includes(plan.upload_status);
  box.innerHTML = `
    <dl>
      <dt>File</dt><dd>${escapeHtml(plan.file_path)}</dd>
      <dt>Stage</dt><dd>${escapeHtml(plan.stage)}
        ${plan.stale ? '— <span class="warn-text">targets an earlier CAD revision; '
          + 'loading it is honoured but its viewpoints may not match the geometry</span>' : ''}</dd>
      <dt>Contents</dt><dd>${plan.num_regions ?? '—'} regions →
        ${plan.num_clusters ?? '—'} clusters → ${plan.num_viewpoints ?? '—'} viewpoints</dd>
      <dt>Generated</dt><dd>${escapeHtml(plan.generated_at || '—')}
        (${escapeHtml(plan.source || 'local')})</dd>
      <dt>3DX document</dt><dd>${escapeHtml(plan.plan_doc_id || 'not uploaded')}
        (${escapeHtml(plan.upload_status || 'local')})
        ${uploadable ? '<button class="btn btn-small" id="upload-plan-btn">Upload</button>' : ''}</dd>
    </dl>`;

  const uploadButton = document.getElementById('upload-plan-btn');
  if (uploadButton) {
    uploadButton.addEventListener('click', () => uploadPlan(plan.plan_id));
  }
}

async function uploadPlan(planId) {
  toast('Uploading plan to 3DEXPERIENCE…');
  try {
    const result = await api(`/api/plans/${encodeURIComponent(planId)}/upload`,
                             { method: 'POST' });
    toast(result.message, result.success ? 'ok' : 'error');
  } catch (e) {
    toast(e.message, 'error');
  }
  await loadPlans(state.openId);
}

async function adoptPlans(engItemId) {
  toast('Scanning for unrecorded plan files…');
  try {
    const result = await api(
      `/api/parts/${encodeURIComponent(engItemId)}/plans/adopt`, { method: 'POST' });
    toast(result.message, result.adopted ? 'ok' : null);
  } catch (e) {
    toast(e.message, 'error');
  }
  await loadPlans(engItemId);
  await loadParts();
}

async function selectPart(engItemId) {
  state.busy.add(engItemId);
  render();
  toast('Preparing part…');
  try {
    const body = { eng_item_id: engItemId };
    if (state.chosenPlanId === '') body.model_only = true;
    else if (state.chosenPlanId) body.plan_id = state.chosenPlanId;
    const result = await api('/api/select', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    state.selectedId = engItemId;
    el.selection.textContent =
      `Selected: ${result.title} — plan ${result.plan_status}`
      + (result.published ? '' : ' (not announced: ROS offline)');
    toast(result.message, result.published ? 'ok' : null);
    await loadParts();
  } catch (e) {
    toast(e.message, 'error');
  } finally {
    state.busy.delete(engItemId);
    render();
  }
}

async function runSync() {
  el.syncBtn.disabled = true;
  el.syncBtn.textContent = 'Syncing…';
  toast('Synchronizing with 3DEXPERIENCE…');
  try {
    const result = await api('/api/sync', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ full: true }),
    });
    toast(result.message || 'Sync complete.', result.success ? 'ok' : 'error');
    await loadParts();
  } catch (e) {
    toast(e.message, 'error');
  } finally {
    el.syncBtn.disabled = false;
    el.syncBtn.textContent = 'Sync';
  }
}

el.syncBtn.addEventListener('click', runSync);
el.search.addEventListener('input', render);
el.maturity.addEventListener('change', render);
el.stateFilter.addEventListener('change', render);
el.detailClose.addEventListener('click', () => {
  el.detail.classList.add('hidden');
  state.openId = null;
});

loadStatus();
loadParts().catch((e) => toast(e.message, 'error'));
setInterval(loadStatus, 30000);
