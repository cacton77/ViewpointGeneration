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
  state.openId = engItemId;
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
    <h3>Inspection plan</h3>
    <div id="plan-box">loading…</div>
    <h3>Inspection history</h3>
    <div id="runs-box">loading…</div>
    <div class="detail-actions">
      <button class="btn btn-primary" id="select-btn">
        ${part.step_cached ? 'Load part' : 'Fetch &amp; load'}
      </button>
    </div>`;

  document.getElementById('select-btn')
    .addEventListener('click', () => selectPart(engItemId));

  if (!keepScroll) el.detail.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

  try {
    const plan = await api(`/api/parts/${encodeURIComponent(engItemId)}/plan`);
    const box = document.getElementById('plan-box');
    if (!box) return;
    if (plan.status === 'NONE') {
      box.innerHTML = '<em>No plan yet — viewpoint generation runs on selection.</em>';
    } else {
      const p = plan.plan || {};
      box.innerHTML = `
        <dl>
          <dt>Status</dt><dd class="plan-${plan.status}">${escapeHtml(plan.status)}
            ${plan.stale_reason ? '— ' + escapeHtml(plan.stale_reason) : ''}</dd>
          <dt>File</dt><dd>${escapeHtml(plan.file_path)}</dd>
          <dt>Regions</dt><dd>${p.num_regions ?? '—'} → ${p.num_clusters ?? '—'} clusters
            → ${p.num_viewpoints ?? '—'} viewpoints</dd>
          <dt>Segmentation</dt><dd>${escapeHtml(p.seg_algorithm || '—')}</dd>
          <dt>Generated</dt><dd>${escapeHtml(p.generated_at || '—')}</dd>
          <dt>3DX document</dt><dd>${escapeHtml(p.plan_doc_id || 'not uploaded')}
            (${escapeHtml(p.upload_status || 'local')})</dd>
        </dl>`;
    }
  } catch (e) { /* the panel keeps its loading text */ }

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

async function selectPart(engItemId) {
  state.busy.add(engItemId);
  render();
  toast('Preparing part…');
  try {
    const result = await api('/api/select', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ eng_item_id: engItemId }),
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
