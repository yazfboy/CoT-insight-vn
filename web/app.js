/* CoT-Insight VN — Web Dashboard tĩnh (GitHub Pages).
 * Dữ liệu: ./data/*.json do scripts/build_web_data.py sinh ra. Mọi kết luận hiển thị đều tính
 * từ dữ liệu trong trình duyệt, không viết sẵn số.
 * Luôn dùng đường dẫn tương đối "./data/..." để chạy đúng dưới subpath /CoT-insight-vn/. */
'use strict';

// ------------------------------------------------------------------ hằng số
const METHODS = ['1_Zero_Shot', '2_Few_Shot', '3_Zero_Shot_CoT', '4_Few_Shot_CoT', '5_Self_Consistency', '6_Few_Shot_CoT_Leaked'];
const SHORT = {
  '1_Zero_Shot': 'Zero-Shot', '2_Few_Shot': 'Few-Shot', '3_Zero_Shot_CoT': 'Zero-CoT',
  '4_Few_Shot_CoT': 'Few-CoT', '5_Self_Consistency': 'Self-Consist', '6_Few_Shot_CoT_Leaked': 'Leaked',
};
const METHOD_DESC = {
  '1_Zero_Shot': 'Trả lời thẳng, không giải thích',
  '2_Few_Shot': 'Ví dụ mẫu, không lập luận',
  '3_Zero_Shot_CoT': '“Hãy suy nghĩ từng bước”',
  '4_Few_Shot_CoT': 'Ví dụ mẫu có lời giải từng bước',
  '5_Self_Consistency': 'CoT nhiều luồng (T=0.7), bỏ phiếu đa số',
  '6_Few_Shot_CoT_Leaked': 'Few-CoT + gợi ý rò rỉ (kiểm thử nhiễm dữ liệu)',
};
// Bảng màu cố định, giống app/charts.py (đã qua validator CVD).
const METHOD_COLORS = {
  '1_Zero_Shot': '#2a78d6', '2_Few_Shot': '#eb6834', '3_Zero_Shot_CoT': '#1baf7a',
  '4_Few_Shot_CoT': '#eda100', '5_Self_Consistency': '#e87ba4', '6_Few_Shot_CoT_Leaked': '#6250d6',
};
const MODELS = ['Small', 'Large', 'Qwen_7B'];
const MODEL_COLORS = { Small: '#eb6834', Large: '#2a78d6', Qwen_7B: '#1baf7a' };
const MODEL_SYMBOL = { Small: 'circle', Large: 'square', Qwen_7B: 'diamond' };
const LANGS = ['VI', 'EN'];
const DIFFS = ['Easy', 'Medium', 'Hard'];
const DIFF_VI = { Easy: 'Dễ', Medium: 'Trung bình', Hard: 'Khó' };
const DIFF_COLORS = { Easy: '#86b6ef', Medium: '#3987e5', Hard: '#184f95' };
const FAITH = ['1->1', '1->0', '0->1', '0->0'];
const COT_FAITH_METHODS = ['3_Zero_Shot_CoT', '4_Few_Shot_CoT', '6_Few_Shot_CoT_Leaked'];
const ROUTABLE = METHODS.slice(0, 5);

// ------------------------------------------------------------------ tiện ích
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const fmt = (v, d = 1) => (Number.isFinite(v) ? v.toLocaleString('vi-VN', { minimumFractionDigits: d, maximumFractionDigits: d }) : '—');
const sgn = (v, d = 1) => (Number.isFinite(v) ? (v > 0 ? '+' : v < 0 ? '−' : '±') + fmt(Math.abs(v), d) : '—');
const accOf = (rows) => (rows.length ? (100 * rows.reduce((s, r) => s + r.correct, 0)) / rows.length : NaN);
const avgOf = (rows, f) => {
  const v = rows.map(f).filter(Number.isFinite);
  return v.length ? v.reduce((a, b) => a + b, 0) / v.length : NaN;
};
const meanArr = (a) => avgOf(a, (x) => x);
const modelLabel = (m) => `${m} · ${(D.meta.model_ids || {})[m] || ''}`;
const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);

async function getJSON(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
  return res.json();
}

const D = {}; // dữ liệu đã nạp
const logsCache = new Map();
async function loadLogs(id) {
  if (!logsCache.has(id)) logsCache.set(id, getJSON(`./data/logs/${encodeURIComponent(id)}.json`));
  try { return await logsCache.get(id); } catch (e) { logsCache.delete(id); throw e; }
}

// ------------------------------------------------------------------ Plotly
const AX = { gridcolor: '#eeece6', zerolinecolor: '#d9d6cd', linecolor: '#d9d6cd', tickfont: { size: 11 }, automargin: true };
const CFG = { displaylogo: false, responsive: true, modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d'] };
function baseLayout(extra = {}) {
  return Object.assign({
    title: null,
    margin: { l: 56, r: 18, t: 34, b: 40 },
    paper_bgcolor: '#ffffff', plot_bgcolor: '#ffffff',
    font: { family: '"Be Vietnam Pro", system-ui, sans-serif', size: 12, color: '#1a2233' },
    hoverlabel: { font: { family: '"Be Vietnam Pro", system-ui, sans-serif', size: 12 } },
    legend: { orientation: 'h', x: 0, y: -0.2, yanchor: 'top', font: { size: 11 } },
  }, extra);
}
/** Facet theo cột. Trả về các trục + annotation tiêu đề ô. */
function facets(titles, { xTitle, yTitle, yRange, xRange, gap = 0.05, yCat = false } = {}) {
  const n = titles.length;
  const w = (1 - gap * (n - 1)) / n;
  const L = { annotations: [] };
  titles.forEach((t, i) => {
    const s = i === 0 ? '' : String(i + 1);
    const x0 = i * (w + gap);
    const x1 = x0 + w;
    const xa = Object.assign({}, AX, { domain: [x0, x1], anchor: 'y' + s });
    if (xTitle) xa.title = { text: xTitle, font: { size: 11 } };
    if (xRange) xa.range = xRange;
    const ya = Object.assign({}, AX, { anchor: 'x' + s, showticklabels: i === 0 });
    if (yRange) ya.range = yRange;
    if (i === 0 && yTitle) ya.title = { text: yTitle, font: { size: 11 } };
    if (i > 0) ya.matches = 'y';
    if (yCat) ya.type = 'category';
    L['xaxis' + s] = xa;
    L['yaxis' + s] = ya;
    L.annotations.push({ text: `<b>${esc(t)}</b>`, x: (x0 + x1) / 2, y: 1, xref: 'paper', yref: 'paper', yanchor: 'bottom', showarrow: false, font: { size: 12 } });
  });
  return L;
}
const axRef = (i) => (i === 0 ? { xaxis: 'x', yaxis: 'y' } : { xaxis: 'x' + (i + 1), yaxis: 'y' + (i + 1) });
const plot = (id, traces, layout) => Plotly.react(id, traces, layout, CFG);
function setText(id, html) { const el = document.getElementById(id); if (el) el.innerHTML = html; }

// ------------------------------------------------------------------ nạp dữ liệu
async function boot() {
  try {
    const [meta, questions, recs, summary, leak, faith, errors, opt] = await Promise.all([
      getJSON('./data/meta.json'), getJSON('./data/questions_150.json'), getJSON('./data/records.json'),
      getJSON('./data/summary.json'), getJSON('./data/leak_compare.json'), getJSON('./data/faithfulness.json'),
      getJSON('./data/errors.json'), getJSON('./data/optimization.json'),
    ]);
    Object.assign(D, { meta, questions, summary, leak, faith, errors, opt });
    D.qById = new Map(questions.map((q) => [q.id, q]));
    D.recs = decodeRecords(recs);
    D.recIdx = new Map(D.recs.map((r) => [`${r.id}|${r.model}|${r.lang}|${r.method}`, r]));
  } catch (e) {
    $('#loading').classList.add('hidden');
    const el = $('#load-error');
    el.classList.remove('hidden');
    el.innerHTML = `<b>Không tải được dữ liệu.</b> ${esc(e.message)}<br>Chạy <code>python scripts/build_web_data.py</code> rồi mở trang qua một web server (vd: <code>python -m http.server -d web</code>), không mở trực tiếp file HTML.`;
    return;
  }
  $('#loading').classList.add('hidden');
  $('#meta').innerHTML = `Nguồn: <code>${esc(D.meta.source)}</code><br>${D.meta.n_records.toLocaleString('vi-VN')} lượt · ${D.meta.n_questions} câu · build ${esc(D.meta.built_at)}`;
  initTab1();
  initTab2();
  initTab3();
  initTabs();
}

function decodeRecords({ n, dict, cols }) {
  const out = new Array(n);
  for (let i = 0; i < n; i++) {
    const id = dict.q[cols.q[i]];
    const q = D.qById.get(id);
    out[i] = {
      id, model: dict.m[cols.m[i]], lang: dict.l[cols.l[i]], method: dict.k[cols.k[i]],
      correct: cols.c[i], pred: cols.p[i], conf: cols.cf[i], sc: cols.sc[i], sr: cols.sr[i],
      faith: cols.f[i] == null ? null : dict.f[cols.f[i]], status: dict.st[cols.st[i]],
      pt: cols.pt[i], ct: cols.ct[i], tt: cols.tt[i], lat: cols.lat[i], cost: cols.cost[i],
      diff: q.difficulty, cat: q.category, leak: q.leak_group === 'Khong_Lo_Unleaked' ? 'Unleaked' : 'Leaked',
    };
  }
  return out;
}

// ------------------------------------------------------------------ tabs
const rendered = {};
function initTabs() {
  $$('.tab-btn').forEach((b) => b.addEventListener('click', () => { location.hash = b.dataset.tab; }));
  window.addEventListener('hashchange', showTab);
  showTab();
}
function showTab() {
  const id = (location.hash || '#t1').slice(1).split('?')[0];
  const tab = ['t1', 't2', 't3', 't4'].includes(id) ? id : 't1';
  $$('.tab-panel').forEach((p) => p.classList.toggle('hidden', p.id !== tab));
  $$('.tab-btn').forEach((b) => b.classList.toggle('active', b.dataset.tab === tab));
  // Plotly cần vẽ khi phần tử đang hiển thị, nên vẽ lười theo tab.
  if (tab === 't2' && !rendered.t2) { renderTab2(); rendered.t2 = true; }
  if (tab === 't3' && !rendered.t3) { renderTab3(); rendered.t3 = true; }
  if (tab === 't4' && !rendered.t4) { renderTab4(); rendered.t4 = true; }
  $$(`#${tab} .plot`).forEach((el) => { if (el.data) Plotly.Plots.resize(el); });
}

function chipGroup(container, items, selected, onChange, { colorOf, labelOf, multi = true } = {}) {
  const el = typeof container === 'string' ? $(container) : container;
  el.innerHTML = items.map((it) => {
    const dot = colorOf ? `<span class="dot" style="background:${colorOf(it)}"></span>` : '';
    return `<span class="chip" data-v="${esc(it)}" role="checkbox" tabindex="0">${dot}${esc(labelOf ? labelOf(it) : it)}</span>`;
  }).join('');
  const sync = () => $$('.chip', el).forEach((c) => {
    c.classList.toggle('on', selected.has(c.dataset.v));
    c.setAttribute('aria-checked', selected.has(c.dataset.v));
  });
  const toggle = (v) => {
    if (multi) {
      if (selected.has(v)) { if (selected.size > 1) selected.delete(v); } else selected.add(v);
    } else { selected.clear(); selected.add(v); }
    sync(); onChange();
  };
  $$('.chip', el).forEach((c) => {
    c.addEventListener('click', () => toggle(c.dataset.v));
    c.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(c.dataset.v); } });
  });
  sync();
}
function fillSelect(sel, options, value) {
  const el = typeof sel === 'string' ? $(sel) : sel;
  el.innerHTML = options.map(([v, l]) => `<option value="${esc(v)}">${esc(l)}</option>`).join('');
  if (value != null) el.value = value;
}

// ================================================================== TAB 1
const T1 = { q: null, model: 'Large', lang: 'VI', focus: null, mode: 'replay' };

function initTab1() {
  fillSelect('#f-diff', [['', 'Tất cả'], ...DIFFS.map((d) => [d, DIFF_VI[d]])]);
  fillSelect('#f-cat', [['', 'Tất cả'], ['Số nguyên', 'Số nguyên'], ['ABCD', 'ABCD (trắc nghiệm)']]);
  const groups = [...new Set(D.questions.map((q) => q.leak_group))].filter((g) => g !== 'Khong_Lo_Unleaked').sort();
  fillSelect('#f-leak', [['', 'Tất cả'], ['__U', 'Chỉ câu không rò rỉ (Unleaked)'], ['__L', 'Chỉ câu có rò rỉ (Leaked)'], ...groups.map((g) => [g, g])]);
  fillSelect('#f-model', MODELS.map((m) => [m, modelLabel(m)]), T1.model);
  fillSelect('#f-lang', [['VI', 'Tiếng Việt (VI)'], ['EN', 'Tiếng Anh (EN)']], T1.lang);

  ['#f-diff', '#f-cat', '#f-leak'].forEach((s) => $(s).addEventListener('change', refreshQuestionList));
  $('#f-search').addEventListener('input', refreshQuestionList);
  $('#f-q').addEventListener('change', (e) => { T1.q = e.target.value; T1.focus = null; renderQuestion(); });
  $('#f-model').addEventListener('change', (e) => { T1.model = e.target.value; renderQuestion(); });
  $('#f-lang').addEventListener('change', (e) => { T1.lang = e.target.value; renderQuestion(); });

  $$('.seg-btn').forEach((b) => b.addEventListener('click', () => setMode(b.dataset.mode)));
  setMode('replay');
  initLive();
  refreshQuestionList();
}

function setMode(mode) {
  T1.mode = mode;
  $$('.seg-btn').forEach((b) => b.classList.toggle('active', b.dataset.mode === mode));
  $('#t1-replay').classList.toggle('hidden', mode !== 'replay');
  $('#t1-live').classList.toggle('hidden', mode !== 'live');
}

function filteredQuestions() {
  const d = $('#f-diff').value, c = $('#f-cat').value, lk = $('#f-leak').value;
  const s = $('#f-search').value.trim().toLowerCase();
  return D.questions.filter((q) => (!d || q.difficulty === d) && (!c || q.category === c)
    && (!lk || (lk === '__U' ? q.leak_group === 'Khong_Lo_Unleaked' : lk === '__L' ? q.leak_group !== 'Khong_Lo_Unleaked' : q.leak_group === lk))
    && (!s || q.id.toLowerCase().includes(s) || q.q_vi.toLowerCase().includes(s) || q.q_en.toLowerCase().includes(s)));
}

function refreshQuestionList() {
  const qs = filteredQuestions();
  $('#q-count').textContent = qs.length;
  const sel = $('#f-q');
  if (!qs.length) {
    sel.innerHTML = '<option>— Không có câu nào khớp bộ lọc —</option>';
    sel.disabled = true;
    $('#q-card').innerHTML = '<div class="text-muted text-sm">Không có câu nào khớp bộ lọc. Hãy nới bộ lọc.</div>';
    $('#q-matrix').innerHTML = '';
    $('#side-grid').innerHTML = '';
    return;
  }
  sel.disabled = false;
  fillSelect(sel, qs.map((q) => [q.id, `${q.id} · ${DIFF_VI[q.difficulty]} · ${q.category} — ${q.q_vi.slice(0, 80)}${q.q_vi.length > 80 ? '…' : ''}`]));
  if (!qs.some((q) => q.id === T1.q)) { T1.q = qs[0].id; T1.focus = null; }
  sel.value = T1.q;
  renderQuestion();
}

function selectQuestion(id, model, lang, method) {
  $('#f-diff').value = ''; $('#f-cat').value = ''; $('#f-leak').value = ''; $('#f-search').value = '';
  T1.q = id; if (model) T1.model = model; if (lang) T1.lang = lang; T1.focus = method || null;
  $('#f-model').value = T1.model; $('#f-lang').value = T1.lang;
  setMode('replay');
  refreshQuestionList();
}

function statusBadge(r) {
  if (!r) return '<span class="badge badge-neutral">Không có dữ liệu</span>';
  const main = r.correct ? '<span class="badge badge-ok">Đúng</span>' : '<span class="badge badge-err">Sai</span>';
  const st = r.status !== 'OK' ? ` <span class="badge badge-neutral">${esc(r.status)}</span>` : '';
  return main + st;
}

const FAITH_HELP = {
  '1->1': 'Bước suy luận đúng → đáp án đúng',
  '1->0': 'Suy luận đúng nhưng chốt đáp án sai (không trung thực)',
  '0->1': 'Đáp án đúng nhưng kết quả cuối trong lời giải không khớp',
  '0->0': 'Suy luận sai → đáp án sai',
};

function renderQuestion() {
  const q = D.qById.get(T1.q);
  if (!q) return;
  const all = D.recs.filter((r) => r.id === q.id);
  const nOk = all.filter((r) => r.correct).length;
  const text = T1.lang === 'VI' ? q.q_vi : q.q_en;
  const other = T1.lang === 'VI' ? q.q_en : q.q_vi;
  $('#q-card').innerHTML = `
    <div class="flex flex-wrap items-center gap-2 mb-2">
      <span class="font-mono text-sm font-semibold">${esc(q.id)}</span>
      <span class="badge badge-neutral">${esc(DIFF_VI[q.difficulty])}</span>
      <span class="badge badge-neutral">${esc(q.category)}</span>
      <span class="badge ${q.leak_group === 'Khong_Lo_Unleaked' ? 'badge-neutral' : 'badge-err'}">${esc(q.leak_group)}</span>
      <span class="text-xs text-muted ml-auto">${nOk}/${all.length} lượt trả lời đúng trên toàn bộ model × ngôn ngữ × phương pháp</span>
    </div>
    <p class="text-[15px] leading-relaxed whitespace-pre-line">${esc(text)}</p>
    <details class="mt-2 text-sm text-muted"><summary class="cursor-pointer">Bản ${T1.lang === 'VI' ? 'tiếng Anh' : 'tiếng Việt'}</summary><p class="mt-1 whitespace-pre-line">${esc(other)}</p></details>
    <div class="mt-3 text-sm">Đáp án chuẩn: <b class="font-mono">${esc(q.answer)}</b></div>`;

  // Ma trận 6 × 6
  let h = '<table class="mx"><thead><tr><th></th>' + METHODS.map((m) => `<th>${esc(SHORT[m])}</th>`).join('') + '</tr></thead><tbody>';
  for (const m of MODELS) for (const l of LANGS) {
    h += `<tr><th class="row">${esc(m)} · ${l}</th>`;
    for (const k of METHODS) {
      const r = D.recIdx.get(`${q.id}|${m}|${l}|${k}`);
      if (!r) { h += '<td>—</td>'; continue; }
      const sel = m === T1.model && l === T1.lang && (!T1.focus || T1.focus === k) ? ' sel' : '';
      const tip = `${m} · ${l} · ${SHORT[k]}: dự đoán ${r.pred ?? '∅'} (chuẩn ${q.answer}), tự tin ${r.conf ?? '—'}%`;
      h += `<td><button class="${r.correct ? 'ok' : 'bad'}${sel}" data-m="${m}" data-l="${l}" data-k="${k}" title="${esc(tip)}" aria-label="${esc(tip)}">${r.correct ? '✓' : '✗'}</button></td>`;
    }
    h += '</tr>';
  }
  h += '</tbody></table>';
  $('#q-matrix').innerHTML = h;
  $$('#q-matrix button').forEach((b) => b.addEventListener('click', () => {
    T1.model = b.dataset.m; T1.lang = b.dataset.l; T1.focus = b.dataset.k;
    $('#f-model').value = T1.model; $('#f-lang').value = T1.lang;
    renderQuestion();
    const card = document.getElementById(`mc-${b.dataset.k}`);
    if (card) card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }));

  // 6 thẻ lời giải
  $('#side-title').textContent = `Đối đầu 6 phương pháp · ${modelLabel(T1.model)} · ${T1.lang}`;
  $('#side-grid').innerHTML = METHODS.map((k) => {
    const r = D.recIdx.get(`${q.id}|${T1.model}|${T1.lang}|${k}`);
    if (!r) return `<div class="method-card p-4 text-sm text-muted">${esc(SHORT[k])}: không có dữ liệu</div>`;
    const sc = k === '5_Self_Consistency'
      ? `<div><span>Đồng thuận SC</span><b>${r.sc == null ? '—' : fmt(r.sc, 0) + '%'} · ${r.sr} luồng</b></div>` : '';
    return `<div class="method-card${T1.focus === k ? ' focus' : ''}" id="mc-${k}">
      <header>
        <div class="flex items-center justify-between gap-2">
          <div class="flex items-center gap-2 min-w-0"><span class="dot inline-block w-2.5 h-2.5 rounded-full shrink-0" style="background:${METHOD_COLORS[k]}"></span>
          <b class="truncate">${esc(SHORT[k])}</b></div>${statusBadge(r)}
        </div>
        <div class="text-xs text-muted mt-0.5">${esc(METHOD_DESC[k])}</div>
      </header>
      <div class="stats">
        <div><span>Dự đoán / chuẩn</span><b class="font-mono">${esc(r.pred ?? '∅')} / ${esc(q.answer)}</b></div>
        <div><span>Tự tin</span><b>${r.conf == null ? '—' : fmt(r.conf, 0) + '%'}</b></div>
        <div title="${esc(FAITH_HELP[r.faith] || 'Chỉ chấm cho Zero-CoT, Few-CoT, Leaked')}"><span>Faithfulness</span><b>${esc(r.faith ? r.faith.replace('->', '→') : 'N/A')}</b></div>
        <div><span>Token (vào + ra)</span><b>${fmt(r.pt, 0)} + ${fmt(r.ct, 0)}</b></div>
        <div><span>Độ trễ</span><b>${fmt(r.lat, 2)} s</b></div>
        <div><span>Chi phí / 1k lượt</span><b>$${fmt(r.cost, 4)}</b></div>
        ${sc}
      </div>
      <pre class="raw" data-raw="${k}">Đang tải lời giải…</pre>
    </div>`;
  }).join('');

  const qid = q.id, model = T1.model, lang = T1.lang;
  loadLogs(qid).then((logs) => {
    if (T1.q !== qid || T1.model !== model || T1.lang !== lang) return;
    METHODS.forEach((k) => {
      const pre = $(`#side-grid pre[data-raw="${k}"]`);
      const lg = logs[`${model}|${lang}|${k}`];
      if (pre) pre.textContent = lg && lg.raw ? lg.raw : '(trống)';
    });
  }).catch((e) => {
    $$('#side-grid pre').forEach((p) => { p.textContent = `Không tải được lời giải: ${e.message}`; });
  });
}

// ---------------------------------------------------- Live Test (Gemini REST)
const EXAMPLES_NOTE = 'Live Test chỉ chạy Zero-Shot và Zero-Shot CoT vì hai prompt này không cần ví dụ mẫu.';
function makePrompt(q, method, lang, isInt) {
  if (lang === 'VI') {
    const rule = isInt ? 'một số nguyên duy nhất' : 'một chữ cái (A, B, C hoặc D)';
    const tail = `BẮT BUỘC kết thúc bằng đúng 2 dòng:\n[Đáp án cuối cùng]: <chỉ ghi ${rule}>\n[Độ tự tin]: <0-100>%`;
    if (method === '1_Zero_Shot') return `Trả lời trực tiếp không giải thích.\nCâu hỏi: ${q}\n\n${tail}`;
    return `Câu hỏi: ${q}\n\nHãy suy nghĩ từng bước trước khi chốt đáp án.\n${tail}`;
  }
  const rule = isInt ? 'a single integer' : 'a single letter (A, B, C, or D)';
  const tail = `MUST end with exact 2 lines:\n[Final Answer]: <${rule}>\n[Confidence]: <0-100>%`;
  if (method === '1_Zero_Shot') return `Answer directly without explanation.\nQuestion: ${q}\n\n${tail}`;
  return `Question: ${q}\n\nLet's think step by step.\n${tail}`;
}

// Cổng JS của parse_output() trong benchmark/run_benchmark.py.
const ANS_TAG_RE = /(?:Đáp án cuối cùng|Final Answer)\s*\]?\s*\**\s*[:：]\s*\**\s*([^\n]*)/gi;
const CONF_TAG_RE = /(?:Độ tự tin|Confidence)\s*\]?\s*\**\s*[:：]\s*\**\s*(\d+(?:[.,]\d+)?)/gi;
const BARE_ANS_RE = /^\W*(?:(?:đáp án|answer)\s*[:：]?\s*)?(-?[\d.,]+|[A-D])\W*$/i;
function extractAnswer(line, isInt) {
  if (isInt) {
    const s = line.replace(/(?<=\d)[.,](?=\d{3}\b)/g, '');
    const nums = s.match(/-?\d+(?:\.\d+)?/g);
    if (!nums) return '';
    const v = parseFloat(nums[nums.length - 1]);
    return String(v);
  }
  const m = /\b([A-D])\b/.exec(line.toUpperCase());
  return m ? m[1] : '';
}
function parseOutput(text, isInt, finish) {
  if (!text) return { ans: '', conf: null, status: 'NO_ANSWER' };
  const confs = [...text.matchAll(CONF_TAG_RE)];
  const conf = confs.length ? Math.min(100, Math.max(0, parseFloat(confs[confs.length - 1][1].replace(',', '.')))) : null;
  let ans = '';
  const tags = [...text.matchAll(ANS_TAG_RE)];
  if (tags.length) ans = extractAnswer(tags[tags.length - 1][1].trim(), isInt);
  else {
    let lines = text.trim().split('\n').map((x) => x.trim()).filter(Boolean);
    // Bỏ MỘT dòng độ tự tin ở cuối rồi mới dò đáp án trần (bản sửa lỗi NO_ANSWER của Qwen).
    if (lines.length > 1 && /(?:Độ tự tin|Confidence)/i.test(lines[lines.length - 1])) lines = lines.slice(0, -1);
    if (lines.length && BARE_ANS_RE.test(lines[lines.length - 1])) ans = extractAnswer(lines[lines.length - 1], isInt);
  }
  const status = ans ? 'OK' : String(finish || '').toUpperCase().endsWith('MAX_TOKENS') ? 'TRUNCATED' : 'NO_ANSWER';
  return { ans, conf, status };
}
function isMatch(pred, gt) {
  if (!pred) return false;
  let g = String(gt).trim().toUpperCase().replace(/,/g, '');
  if (g.endsWith('.0')) g = g.slice(0, -2);
  return String(pred).trim().toUpperCase().replace(/,/g, '') === g;
}

async function callGemini(key, model, prompt, maxTok) {
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(model)}:generateContent`;
  const gen = { temperature: 0, maxOutputTokens: maxTok };
  // Gemma 4 mặc định "nghĩ ngầm" ~2000 token; benchmark tắt bằng thinking_level=minimal.
  if (model.startsWith('gemma')) gen.thinkingConfig = { thinkingLevel: 'minimal' };
  const send = () => fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'x-goog-api-key': key },
    body: JSON.stringify({ contents: [{ role: 'user', parts: [{ text: prompt }] }], generationConfig: gen }),
  });
  const t0 = performance.now();
  let res = await send();
  let j = await res.json().catch(() => ({}));
  if (!res.ok && res.status === 400 && gen.thinkingConfig) { // model không nhận thinkingConfig -> gửi lại không có nó
    delete gen.thinkingConfig;
    res = await send();
    j = await res.json().catch(() => ({}));
  }
  const lat = (performance.now() - t0) / 1000;
  if (!res.ok) {
    const msg = (j.error && j.error.message) || `HTTP ${res.status}`;
    throw new Error(res.status === 400 && /API key/i.test(msg) ? 'API key không hợp lệ.' : res.status === 429 ? `Hết hạn mức (429): ${msg}` : msg);
  }
  const cand = (j.candidates || [])[0];
  if (!cand) throw new Error((j.promptFeedback && `Bị chặn: ${j.promptFeedback.blockReason}`) || 'API không trả về kết quả.');
  const text = ((cand.content && cand.content.parts) || []).filter((p) => !p.thought).map((p) => p.text || '').join('');
  const u = j.usageMetadata || {};
  return { text, finish: cand.finishReason, pt: u.promptTokenCount || 0, ct: (u.candidatesTokenCount || 0) + (u.thoughtsTokenCount || 0), lat };
}

function initLive() {
  $('#lv-fill').addEventListener('click', () => {
    const q = D.qById.get(T1.q);
    if (!q) return;
    const lang = $('#lv-lang').value;
    $('#lv-q').value = lang === 'VI' ? q.q_vi : q.q_en;
    $('#lv-type').value = q.category;
    $('#lv-gt').value = q.answer;
  });
  $('#lv-run').addEventListener('click', runLive);
}

async function runLive() {
  const key = $('#lv-key').value.trim();
  const model = $('#lv-model').value, lang = $('#lv-lang').value;
  const isInt = $('#lv-type').value === 'Số nguyên';
  const q = $('#lv-q').value.trim(), gt = $('#lv-gt').value.trim();
  const msg = $('#lv-msg');
  if (!key) { msg.innerHTML = '<div class="notice notice-warn">Nhập Gemini API key trước khi gọi.</div>'; return; }
  if (!q) { msg.innerHTML = '<div class="notice notice-warn">Nhập câu hỏi (hoặc bấm “Lấy câu đang chọn”).</div>'; return; }
  const btn = $('#lv-run');
  btn.disabled = true; btn.textContent = 'Đang gọi…';
  msg.innerHTML = `<div class="notice notice-info">Đang gọi <code>${esc(model)}</code> cho 2 phương pháp… ${esc(EXAMPLES_NOTE)}</div>`;
  const jobs = [['1_Zero_Shot', 256], ['3_Zero_Shot_CoT', 2048]];
  const results = await Promise.allSettled(jobs.map(([m, tok]) => callGemini(key, model, makePrompt(q, m, lang, isInt), tok)));
  btn.disabled = false; btn.textContent = 'Gọi API thật';
  const errs = results.filter((r) => r.status === 'rejected');
  msg.innerHTML = errs.length === results.length
    ? `<div class="notice notice-err"><b>Gọi API thất bại.</b> ${esc(errs[0].reason.message)}</div>`
    : errs.length ? '<div class="notice notice-warn">Một phương pháp gọi lỗi, xem chi tiết bên dưới.</div>' : '';
  $('#lv-out').innerHTML = jobs.map(([m], i) => {
    const r = results[i];
    if (r.status === 'rejected') {
      return `<div class="method-card p-4"><b>${esc(SHORT[m])}</b><div class="notice notice-err mt-2">${esc(r.reason.message)}</div></div>`;
    }
    const v = r.value, p = parseOutput(v.text, isInt, v.finish);
    const badge = gt ? (isMatch(p.ans, gt) ? '<span class="badge badge-ok">Đúng</span>' : '<span class="badge badge-err">Sai</span>') : '<span class="badge badge-neutral">Chưa chấm</span>';
    return `<div class="method-card">
      <header><div class="flex items-center justify-between gap-2"><div class="flex items-center gap-2"><span class="inline-block w-2.5 h-2.5 rounded-full" style="background:${METHOD_COLORS[m]}"></span><b>${esc(SHORT[m])}</b></div>${badge}</div>
      <div class="text-xs text-muted mt-0.5">${esc(METHOD_DESC[m])}</div></header>
      <div class="stats">
        <div><span>Dự đoán</span><b class="font-mono">${esc(p.ans || '∅')}${p.status !== 'OK' ? ' · ' + esc(p.status) : ''}</b></div>
        <div><span>Tự tin</span><b>${p.conf == null ? '—' : fmt(p.conf, 0) + '%'}</b></div>
        <div><span>Độ trễ</span><b>${fmt(v.lat, 2)} s</b></div>
        <div><span>Token vào</span><b>${fmt(v.pt, 0)}</b></div>
        <div><span>Token ra</span><b>${fmt(v.ct, 0)}</b></div>
        <div><span>Kết thúc</span><b>${esc(v.finish || '—')}</b></div>
      </div>
      <pre class="raw">${esc(v.text || '(trống)')}</pre></div>`;
  }).join('');
}

// ================================================================== TAB 2
const T2 = { models: new Set(MODELS), langs: new Set(LANGS), xMetric: 'tt' };
const XMETRIC = {
  tt: { label: 'Token trung bình / lượt', f: (r) => r.tt, fmt: (v) => fmt(v, 0) + ' token' },
  cost: { label: 'Chi phí ($ / 1.000 lượt)', f: (r) => r.cost, fmt: (v) => '$' + fmt(v, 4) },
  lat: { label: 'Độ trễ trung bình (s)', f: (r) => r.lat, fmt: (v) => fmt(v, 2) + ' s' },
};

function initTab2() {
  chipGroup('#t2-models', MODELS, T2.models, renderTab2, { colorOf: (m) => MODEL_COLORS[m] });
  chipGroup('#t2-langs', LANGS, T2.langs, renderTab2);
  const xs = new Set([T2.xMetric]);
  chipGroup('#c2-x', Object.keys(XMETRIC), xs, () => { T2.xMetric = [...xs][0]; renderC2(sel2()); }, { labelOf: (k) => XMETRIC[k].label, multi: false });
}
const sel2 = () => D.recs.filter((r) => T2.models.has(r.model) && T2.langs.has(r.lang));
const selModels = () => MODELS.filter((m) => T2.models.has(m));
const selLangs = () => LANGS.filter((l) => T2.langs.has(l));

function renderTab2() {
  if (!D.recs) return;
  if ($('#t2').classList.contains('hidden')) { rendered.t2 = false; return; }
  const rows = sel2();
  $('#t2-n').textContent = `${rows.length.toLocaleString('vi-VN')} lượt đang được tính`;
  renderC1(rows); renderC2(rows); renderC3(rows); renderC4();
}

function renderC1(rows) {
  const ms = selModels();
  const traces = [];
  ms.forEach((m, i) => DIFFS.forEach((d) => {
    const ys = METHODS.map((k) => accOf(rows.filter((r) => r.model === m && r.diff === d && r.method === k)));
    traces.push(Object.assign({
      type: 'bar', name: DIFF_VI[d], legendgroup: d, showlegend: i === 0, x: METHODS.map((k) => SHORT[k]), y: ys,
      marker: { color: DIFF_COLORS[d] },
      hovertemplate: `${m} · ${DIFF_VI[d]}<br>%{x}: %{y:.1f}%<extra></extra>`,
    }, axRef(i)));
  }));
  const L = baseLayout(Object.assign(facets(ms, { yRange: [0, 104], yTitle: 'Accuracy (%)' }), { barmode: 'group', bargap: 0.22, bargroupgap: 0.05 }));
  ms.forEach((m, i) => { L['xaxis' + (i ? i + 1 : '')].tickangle = -35; });
  plot('c1', traces, L);

  const gain = DIFFS.map((d) => {
    const zs = accOf(rows.filter((r) => r.diff === d && r.method === '1_Zero_Shot'));
    const zc = accOf(rows.filter((r) => r.diff === d && r.method === '3_Zero_Shot_CoT'));
    return { d, zs, zc, g: zc - zs };
  });
  const best = gain.reduce((a, b) => (b.g > a.g ? b : a));
  const worst = gain.reduce((a, b) => (b.g < a.g ? b : a));
  setText('c1-claim', best.g > 0
    ? `CoT giúp nhiều nhất ở câu ${DIFF_VI[best.d]}: Zero-Shot ${fmt(best.zs)}% → Zero-CoT ${fmt(best.zc)}% (${sgn(best.g)} điểm), ít nhất ở câu ${DIFF_VI[worst.d]} (${sgn(worst.g)} điểm)`
    : `Zero-CoT không cải thiện so với Zero-Shot ở bất kỳ mức độ khó nào trong dữ liệu đang chọn`);
  setText('c1-read', `Đọc: mỗi ô là một model; trong mỗi phương pháp, ba cột từ nhạt đến đậm là Dễ → Trung bình → Khó (50 câu mỗi mức × ngôn ngữ đang chọn). Trục y bắt đầu từ 0.`);

  const zsAll = accOf(rows.filter((r) => r.method === '1_Zero_Shot'));
  const fsAll = accOf(rows.filter((r) => r.method === '2_Few_Shot'));
  const perModel = ms.map((m) => {
    const g = DIFFS.map((d) => accOf(rows.filter((r) => r.model === m && r.diff === d && r.method === '3_Zero_Shot_CoT'))
      - accOf(rows.filter((r) => r.model === m && r.diff === d && r.method === '1_Zero_Shot')));
    const j = g.indexOf(Math.max(...g));
    return `${m}: ${DIFF_VI[DIFFS[j]]} (${sgn(g[j])})`;
  });
  setText('c1-con', `<b>Kết luận:</b> mức có lợi nhất cho từng model — ${perModel.join('; ')}. `
    + (fsAll < zsAll
      ? `Few-Shot không có lời giải còn thấp hơn Zero-Shot ${fmt(zsAll - fsAll)} điểm, nên phần tăng đến từ bước suy luận chứ không phải từ việc có ví dụ.`
      : `Few-Shot không có lời giải cao hơn Zero-Shot ${fmt(fsAll - zsAll)} điểm, nên một phần lợi ích đến từ ví dụ mẫu.`));
}

function paretoFront(pts) {
  // Không bị trội: không có điểm nào vừa rẻ hơn (hoặc bằng) vừa đúng hơn (hoặc bằng), với ít nhất một bất đẳng thức chặt.
  return pts.filter((p) => !pts.some((o) => o !== p && o.x <= p.x && o.y >= p.y && (o.x < p.x || o.y > p.y)));
}

function renderC2(rows) {
  const X = XMETRIC[T2.xMetric];
  const pts = [];
  for (const m of selModels()) for (const l of selLangs()) for (const k of METHODS) {
    const g = rows.filter((r) => r.model === m && r.lang === l && r.method === k);
    if (g.length) pts.push({ m, l, k, x: avgOf(g, X.f), y: accOf(g), n: g.length });
  }
  const traces = METHODS.map((k) => {
    const p = pts.filter((q) => q.k === k);
    return {
      type: 'scatter', mode: 'markers', name: SHORT[k], x: p.map((q) => q.x), y: p.map((q) => q.y),
      marker: {
        size: 12, color: METHOD_COLORS[k], line: { width: 1.5, color: METHOD_COLORS[k] },
        symbol: p.map((q) => MODEL_SYMBOL[q.m] + (q.l === 'VI' ? '-open' : '')),
      },
      text: p.map((q) => `${q.m} · ${q.l} · ${SHORT[k]}<br>${X.label}: ${X.fmt(q.x)}<br>Accuracy: ${fmt(q.y)}%`),
      hovertemplate: '%{text}<extra></extra>',
    };
  });
  // Chú giải ký hiệu model (điểm ảo, không có dữ liệu).
  selModels().forEach((m) => traces.push({ type: 'scatter', mode: 'markers', x: [null], y: [null], name: `${m} (ký hiệu)`, marker: { symbol: MODEL_SYMBOL[m], size: 10, color: '#8a8f99' }, hoverinfo: 'skip' }));
  const L = baseLayout({
    xaxis: Object.assign({}, AX, { type: 'log', title: { text: X.label + ' — thang log', font: { size: 11 } } }),
    yaxis: Object.assign({}, AX, { title: { text: 'Accuracy (%)', font: { size: 11 } } }),
    legend: { orientation: 'h', x: 0, y: -0.22, yanchor: 'top', font: { size: 11 } },
    margin: { l: 56, r: 18, t: 16, b: 40 },
  });
  plot('c2', traces, L);

  const zc = rows.filter((r) => r.method === '3_Zero_Shot_CoT');
  const sc = rows.filter((r) => r.method === '5_Self_Consistency');
  const ratio = avgOf(sc, X.f) / avgOf(zc, X.f);
  const dAcc = accOf(sc) - accOf(zc);
  setText('c2-claim', `Self-Consistency tốn gấp ${fmt(ratio)} lần ${T2.xMetric === 'tt' ? 'token' : T2.xMetric === 'cost' ? 'chi phí' : 'thời gian'} của Zero-CoT nhưng accuracy chỉ ${dAcc >= 0 ? 'thêm' : 'đổi'} ${sgn(dAcc)} điểm`);
  setText('c2-read', 'Đọc: mỗi điểm là một tổ hợp model × ngôn ngữ × phương pháp (150 câu). Màu = phương pháp; hình = model (● Small, ■ Large, ◆ Qwen_7B); <b>đặc = EN, rỗng = VI</b>. Càng lên trên và sang trái càng hiệu quả. Trục x dạng log nên mỗi vạch lớn là ×10.');
  const front = paretoFront(pts.filter((p) => ROUTABLE.includes(p.k)));
  const cnt = {};
  front.forEach((p) => { cnt[p.k] = (cnt[p.k] || 0) + 1; });
  const ranked = Object.entries(cnt).sort((a, b) => b[1] - a[1]).map(([k, c]) => `${SHORT[k]} (${c})`);
  setText('c2-con', `<b>Kết luận:</b> trong ${front.length} điểm nằm trên biên Pareto (không bị điểm nào vừa rẻ hơn vừa đúng hơn trội; không tính Leaked), các phương pháp xuất hiện là: ${ranked.join(', ')}. `
    + (cnt['5_Self_Consistency']
      ? `Self-Consistency có mặt trên biên ở ${cnt['5_Self_Consistency']} điểm, tức là có cấu hình mà chi phí thêm vẫn mua được accuracy.`
      : `Self-Consistency không có điểm nào trên biên: lợi ích chính của CoT đã có ở một luồng, nhân thêm luồng chủ yếu làm tăng chi phí.`));
}

function renderC3(rows) {
  const ms = selModels(), ls = selLangs();
  const traces = [];
  for (const m of ms) for (const l of ls) {
    traces.push({
      type: 'scatter', mode: 'lines+markers', name: `${m} · ${l}`, x: METHODS.map((k) => SHORT[k]),
      y: METHODS.map((k) => accOf(rows.filter((r) => r.model === m && r.lang === l && r.method === k))),
      line: { color: MODEL_COLORS[m], width: 2.2, dash: l === 'VI' ? 'dash' : 'solid' },
      marker: { size: 7, symbol: MODEL_SYMBOL[m], color: MODEL_COLORS[m] },
      hovertemplate: `${m} · ${l}<br>%{x}: %{y:.1f}%<extra></extra>`,
    });
  }
  plot('c3a', traces, baseLayout({
    xaxis: Object.assign({}, AX), yaxis: Object.assign({}, AX, { title: { text: 'Accuracy (%)', font: { size: 11 } } }),
    margin: { l: 56, r: 18, t: 16, b: 40 },
  }));

  const accM = (m, k) => accOf(rows.filter((r) => r.model === m && r.method === k));
  let claim = '';
  if (ms.length >= 2) {
    const zs = ms.map((m) => ({ m, v: accM(m, '1_Zero_Shot') }));
    const zc = ms.map((m) => ({ m, v: accM(m, '3_Zero_Shot_CoT') }));
    const gap = (a) => Math.max(...a.map((x) => x.v)) - Math.min(...a.map((x) => x.v));
    const lo = zs.reduce((a, b) => (b.v < a.v ? b : a));
    claim = `Khoảng cách giữa các model ${gap(zc) < gap(zs) ? 'thu hẹp' : 'nới rộng'} khi dùng CoT: ${fmt(gap(zs))} điểm ở Zero-Shot → ${fmt(gap(zc))} điểm ở Zero-CoT (model yếu nhất ở Zero-Shot: ${lo.m})`;
  } else {
    const m = ms[0];
    claim = `${m}: Zero-Shot ${fmt(accM(m, '1_Zero_Shot'))}% → Zero-CoT ${fmt(accM(m, '3_Zero_Shot_CoT'))}%`;
  }
  setText('c3-claim', claim);
  let langTxt = '';
  if (ls.length === 2) {
    const gaps = ms.map((m) => ({ m, g: accOf(rows.filter((r) => r.model === m && r.lang === 'EN')) - accOf(rows.filter((r) => r.model === m && r.lang === 'VI')) }));
    langTxt = 'Chênh lệch EN − VI (trung bình 6 phương pháp): ' + gaps.map((x) => `${x.m} ${sgn(x.g)} điểm`).join(', ') + '.';
  }
  setText('c3a-read', `Đọc: mỗi đường là một model × ngôn ngữ; nét liền = EN, nét đứt = VI. ${langTxt}`);

  // Thuế token
  if (ls.length < 2) {
    Plotly.purge('c3b');
    $('#c3b').innerHTML = '<div class="p-6 text-sm text-muted">Chọn cả VI và EN để so sánh token giữa hai ngôn ngữ.</div>';
    setText('c3b-claim', 'Thuế token của tiếng Việt');
    setText('c3b-read', '');
    setText('c3-con', '');
    return;
  }
  // Chỉ xoá dòng thông báo khi chưa có biểu đồ (xoá DOM của một plot đang sống làm Plotly.react vẽ ra ô trống).
  if (!$('#c3b').data) $('#c3b').innerHTML = '';
  const ratio = (m, f) => avgOf(rows.filter((r) => r.model === m && r.lang === 'VI'), f) / avgOf(rows.filter((r) => r.model === m && r.lang === 'EN'), f);
  const rp = ms.map((m) => ratio(m, (r) => r.pt)), rc = ms.map((m) => ratio(m, (r) => r.ct));
  const mk = (name, ys, color) => ({
    type: 'bar', name, x: ms, y: ys, marker: { color }, text: ys.map((v) => '×' + fmt(v, 2)), textposition: 'outside', cliponaxis: false,
    hovertemplate: `%{x} · ${name}<br>VI / EN = %{y:.2f}×<extra></extra>`,
  });
  const ymax = Math.max(...rp, ...rc);
  plot('c3b', [mk('Prompt (token vào)', rp, '#3987e5'), mk('Lời giải (token ra)', rc, '#184f95')], baseLayout({
    barmode: 'group', bargap: 0.35,
    xaxis: Object.assign({}, AX),
    yaxis: Object.assign({}, AX, { title: { text: 'Tỷ lệ token VI / EN (×)', font: { size: 11 } }, range: [0, ymax * 1.18] }),
    shapes: [{ type: 'line', xref: 'paper', x0: 0, x1: 1, y0: 1, y1: 1, line: { color: '#5b6170', dash: 'dot', width: 1 } }],
    margin: { l: 56, r: 18, t: 16, b: 40 },
  }));
  const pAvg = meanArr(rp), cAvg = meanArr(rc);
  const both = rp.every((v) => v > 1) && rc.every((v) => v > 1);
  setText('c3b-claim', both
    ? `“Thuế token kép”: cùng một câu, tiếng Việt tốn gấp ${fmt(pAvg, 2)}× token ở prompt và ${fmt(cAvg, 2)}× ở lời giải`
    : `Tiếng Việt tốn gấp ${fmt(pAvg, 2)}× token ở prompt và ${fmt(cAvg, 2)}× ở lời giải (không phải model nào cũng tốn hơn ở cả hai chiều)`);
  setText('c3b-read', 'Đọc: cột = token trung bình của bản VI chia cho bản EN trên cùng 150 câu và cùng phương pháp; vạch chấm = 1× (ngang nhau). Cột “Prompt” phản ánh tokenizer; cột “Lời giải” phản ánh model viết dài hơn khi lập luận bằng tiếng Việt.');
  const accGap = meanArr(ms.map((m) => accOf(rows.filter((r) => r.model === m && r.lang === 'EN')) - accOf(rows.filter((r) => r.model === m && r.lang === 'VI'))));
  setText('c3-con', `<b>Kết luận:</b> tiếng Việt vừa tốn thêm ${fmt((pAvg - 1) * 100, 0)}% token đầu vào và ${fmt((cAvg - 1) * 100, 0)}% token đầu ra, `
    + (accGap > 0 ? `vừa thấp hơn EN trung bình ${fmt(accGap)} điểm accuracy` : `trong khi accuracy ${accGap === 0 ? 'ngang' : 'cao hơn'} EN ${fmt(Math.abs(accGap))} điểm`)
    + '. Chi phí cho người dùng tiếng Việt cao hơn ở cả hai đầu của lời gọi API.');
}

function renderC4() {
  // Chỉ dùng 49 câu Leaked: trên câu Unleaked, pp6 có prompt giống hệt pp4 nên ở cả 3 model
  // kết quả được sao từ pp4 qua bảng DP -> không phải nhóm đối chứng thật (chênh lệch = 0 theo thiết kế).
  const ms = selModels(), ls = selLangs();
  const rows = D.leak.filter((r) => T2.models.has(r.Model) && T2.langs.has(r.Language) && r.Leak_Status === 'Leaked');
  const cats = [];
  const seg = { x: [], y: [] }, p4 = { x: [], y: [], t: [] }, p6 = { x: [], y: [], t: [] };
  const deltas = [];
  let n = 0;
  for (const m of ms) for (const l of ls) {
    const a = rows.find((r) => r.Model === m && r.Language === l && r.Method === '4_Few_Shot_CoT');
    const b = rows.find((r) => r.Model === m && r.Language === l && r.Method === '6_Few_Shot_CoT_Leaked');
    if (!a || !b) continue;
    const c = `${m} · ${l}`;
    cats.push(c); n = a.Count;
    seg.x.push(a.Acc_Pct, b.Acc_Pct, null); seg.y.push(c, c, null);
    p4.x.push(a.Acc_Pct); p4.y.push(c); p4.t.push(`${c}<br>pp4 sạch: ${fmt(a.Acc_Pct)}% (n=${a.Count})`);
    p6.x.push(b.Acc_Pct); p6.y.push(c); p6.t.push(`${c}<br>pp6 rò rỉ: ${fmt(b.Acc_Pct)}% (${sgn(b.Acc_Pct - a.Acc_Pct)} điểm)`);
    deltas.push({ c, m, l, d: b.Acc_Pct - a.Acc_Pct });
  }
  const traces = [
    { type: 'scatter', mode: 'lines', x: seg.x, y: seg.y, line: { color: '#c3c2b7', width: 3 }, hoverinfo: 'skip', showlegend: false },
    { type: 'scatter', mode: 'markers', name: 'pp4 · Few-CoT sạch', x: p4.x, y: p4.y, text: p4.t, hovertemplate: '%{text}<extra></extra>', marker: { size: 13, color: '#898781' } },
    { type: 'scatter', mode: 'markers', name: 'pp6 · Few-CoT + gợi ý rò rỉ', x: p6.x, y: p6.y, text: p6.t, hovertemplate: '%{text}<extra></extra>', marker: { size: 13, color: METHOD_COLORS['6_Few_Shot_CoT_Leaked'] } },
  ];
  const allX = rows.map((r) => r.Acc_Pct);
  const lo = Math.max(0, Math.floor((Math.min(...allX) - 4) / 5) * 5);
  plot('c4', traces, baseLayout({
    xaxis: Object.assign({}, AX, { range: [lo, 101.5], title: { text: 'Accuracy trên các câu bị rò rỉ (%)', font: { size: 11 } } }),
    yaxis: Object.assign({}, AX, { type: 'category', categoryorder: 'array', categoryarray: cats.slice().reverse() }),
    margin: { l: 96, r: 18, t: 16, b: 40 },
  }));

  const mL = meanArr(deltas.map((x) => x.d));
  const big = deltas.slice().sort((a, b) => b.d - a.d)[0];
  setText('c4-claim', `Gợi ý rò rỉ đẩy accuracy trên ${n} câu bị rò rỉ thêm ${sgn(mL)} điểm trung bình` + (big ? `, mạnh nhất ở ${big.c} (${sgn(big.d)} điểm)` : ''));
  setText('c4-read', `Đọc: chấm xám = Few-CoT sạch (pp4), chấm tím = cùng prompt nhưng thêm gợi ý rò rỉ (pp6); đoạn nối càng dài thì gợi ý càng tác động. Chỉ có ${n} câu nên mỗi câu ≈ ${fmt(100 / n, 1)} điểm %. Trục x bắt đầu từ ${lo}%. `
    + 'Không có nhóm đối chứng: trên 101 câu Unleaked, pp6 có prompt giống hệt pp4 nên kết quả được sao từ pp4 (bảng DP) ở cả 3 model.');
  let txt = '';
  if (ls.length === 2) {
    const vi = meanArr(deltas.filter((x) => x.l === 'VI').map((x) => x.d)), en = meanArr(deltas.filter((x) => x.l === 'EN').map((x) => x.d));
    txt += `Gợi ý tác động ở VI (${sgn(vi)} điểm) ${vi > en ? 'mạnh hơn' : vi < en ? 'yếu hơn' : 'ngang'} EN (${sgn(en)} điểm). `;
  }
  if (ms.length >= 2) {
    const byM = ms.map((m) => ({ m, d: meanArr(deltas.filter((x) => x.m === m).map((x) => x.d)) }));
    const sorted = byM.slice().sort((a, b) => b.d - a.d);
    txt += `Theo model: ${byM.map((x) => `${x.m} ${sgn(x.d)}`).join(', ')} điểm`;
    // Chỉ gọi tên model "hưởng lợi nhiều nhất" khi cách biệt ít nhất 1 câu (~2 điểm).
    txt += sorted[0].d - sorted[1].d >= 100 / Math.max(1, n)
      ? `; ${sorted[0].m} hưởng lợi rõ nhất.` : ' — chênh lệch giữa các model chưa tới 1 câu, chưa đủ để xếp hạng.';
  }
  setText('c4-con', `<b>Kết luận:</b> ${mL > 0 ? 'lộ công thức/ngữ cảnh làm điểm số cao hơn năng lực thật' : 'gợi ý rò rỉ không làm tăng điểm trong dữ liệu đang chọn'}. ${txt}`);
}

// ================================================================== TAB 3
const T3 = { langs: new Set(LANGS), methods: new Set(COT_FAITH_METHODS) };
const E = { page: 0, size: 20, rows: [], sel: null };

function initTab3() {
  chipGroup('#t3-langs', LANGS, T3.langs, renderTab3Charts);
  chipGroup('#t3-methods', COT_FAITH_METHODS, T3.methods, renderTab3Charts, { labelOf: (k) => SHORT[k], colorOf: (k) => METHOD_COLORS[k] });
  const all = [['', 'Tất cả']];
  fillSelect('#e-model', all.concat(MODELS.map((m) => [m, m])));
  fillSelect('#e-lang', all.concat(LANGS.map((l) => [l, l])));
  fillSelect('#e-method', all.concat(METHODS.map((k) => [k, SHORT[k] + ' — ' + METHOD_DESC[k]])));
  fillSelect('#e-diff', all.concat(DIFFS.map((d) => [d, DIFF_VI[d]])));
  fillSelect('#e-faith', all.concat(FAITH.map((f) => [f, f.replace('->', '→')]), [['__NA', 'N/A (không chấm)']]));
  ['#e-model', '#e-lang', '#e-method', '#e-diff', '#e-faith'].forEach((s) => $(s).addEventListener('change', () => { E.page = 0; renderErrors(); }));
  $('#e-search').addEventListener('input', () => { E.page = 0; renderErrors(); });
}
function renderTab3() { renderTab3Charts(); renderErrors(); }
function renderTab3Charts() {
  if ($('#t3').classList.contains('hidden')) { rendered.t3 = false; return; }
  renderF1(); renderF2();
}

function renderF1() {
  const rows = D.recs.filter((r) => T3.langs.has(r.lang) && T3.methods.has(r.method) && r.faith);
  const traces = [], ann = [];
  const L = baseLayout(facets(MODELS, { gap: 0.06, yCat: true }));
  const per = {};
  MODELS.forEach((m, i) => {
    const g = rows.filter((r) => r.model === m);
    const n = g.length;
    const c = Object.fromEntries(FAITH.map((f) => [f, g.filter((r) => r.faith === f).length]));
    const p = (f) => (n ? (100 * c[f]) / n : NaN);
    per[m] = { n, c, p: Object.fromEntries(FAITH.map((f) => [f, p(f)])) };
    const x = ['Đáp án đúng', 'Đáp án sai'];
    const y = ['Suy luận sai', 'Suy luận đúng'];
    const z = [[p('0->1'), p('0->0')], [p('1->1'), p('1->0')]];
    const lab = [['0->1', '0->0'], ['1->1', '1->0']];
    traces.push(Object.assign({
      type: 'heatmap', x, y, z, zmin: 0, zmax: 100, showscale: i === MODELS.length - 1, xgap: 3, ygap: 3,
      colorscale: [[0, '#f3f7fc'], [0.25, '#b7d3f4'], [0.6, '#3987e5'], [1, '#184f95']],
      colorbar: { title: { text: '%', side: 'top' }, thickness: 10, len: 0.8 },
      customdata: lab.map((r) => r.map((f) => `${f.replace('->', '→')}: ${c[f]} lượt`)),
      hovertemplate: `${m}<br>%{y} · %{x}<br>%{customdata} (%{z:.1f}%)<extra></extra>`,
    }, axRef(i)));
    for (let a = 0; a < 2; a++) for (let b = 0; b < 2; b++) {
      const v = z[a][b], f = lab[a][b];
      const s = i === 0 ? '' : String(i + 1);
      ann.push({ x: x[b], y: y[a], xref: 'x' + s, yref: 'y' + s, showarrow: false, text: `<b>${f.replace('->', '→')}</b><br>${fmt(v)}%<br>(${c[f]})`, font: { size: 11, color: v > 45 ? '#ffffff' : '#1a2233' } });
    }
  });
  L.annotations = L.annotations.concat(ann);
  L.margin = { l: 96, r: 18, t: 34, b: 30 };
  plot('f1', traces, L);

  const tot = rows.length;
  const pct = (f) => (100 * rows.filter((r) => r.faith === f).length) / tot;
  const hi01 = MODELS.reduce((a, b) => (per[b].p['0->1'] > per[a].p['0->1'] ? b : a));
  setText('f1-claim', `Ở ${fmt(pct('1->1') + pct('0->0'))}% lượt CoT, lời giải và đáp án cuối khớp nhau; lượt “suy luận đúng → chốt sai” (1→0) chỉ chiếm ${fmt(pct('1->0'))}%`);
  setText('f1-read', `Đọc: hàng = kết quả cuối cùng trong phần lập luận có khớp đáp án chuẩn không; cột = đáp án chốt có đúng không. Ô 1→0 là dấu hiệu “không trung thực”. Tính trên ${tot.toLocaleString('vi-VN')} lượt CoT có nhãn (Zero-Shot, Few-Shot và Self-Consistency không được chấm). Nhãn do heuristic gán tự động: lấy số/chữ cái cuối cùng trong phần lập luận.`);
  setText('f1-con', `<b>Kết luận:</b> lỗi chủ yếu là “sai từ lập luận” (0→0, ${fmt(pct('0->0'))}%) chứ không phải chốt lệch. Nhóm 0→1 (${fmt(pct('0->1'))}%, cao nhất ở ${hi01} với ${fmt(per[hi01].p['0->1'])}%) nhiều khả năng là do heuristic bắt nhầm số trung gian hoặc chữ cái phương án, nên cần đọc tay trước khi kết luận là model “đoán đúng mà lập luận sai”.`);
}

function renderF2() {
  const rows = D.recs.filter((r) => T3.langs.has(r.lang) && r.conf != null);
  const traces = MODELS.map((m) => {
    const st = METHODS.map((k) => {
      const g = rows.filter((r) => r.model === m && r.method === k);
      const w = g.filter((r) => !r.correct);
      return { cw: avgOf(w, (r) => r.conf), cr: avgOf(g.filter((r) => r.correct), (r) => r.conf), nw: w.length, acc: accOf(g) };
    });
    return {
      type: 'bar', name: m, x: METHODS.map((k) => SHORT[k]), y: st.map((s) => s.cw), marker: { color: MODEL_COLORS[m] },
      customdata: st.map((s) => [s.nw, s.cr, s.acc]),
      hovertemplate: `${m} · %{x}<br>Tự tin khi SAI: %{y:.1f}% (n=%{customdata[0]})<br>Tự tin khi đúng: %{customdata[1]:.1f}%<br>Accuracy: %{customdata[2]:.1f}%<extra></extra>`,
    };
  });
  const ys = traces.flatMap((t) => t.y).filter(Number.isFinite);
  const lo = Math.max(0, Math.floor((Math.min(...ys) - 5) / 5) * 5);
  plot('f2', traces, baseLayout({
    barmode: 'group', bargap: 0.25,
    xaxis: Object.assign({}, AX),
    yaxis: Object.assign({}, AX, { range: [lo, 101], title: { text: 'Độ tự tin TB khi trả lời sai (%)', font: { size: 11 } } }),
    margin: { l: 56, r: 18, t: 16, b: 40 },
  }));
  const wrong = rows.filter((r) => !r.correct);
  const overall = avgOf(wrong, (r) => r.conf);
  const per = MODELS.map((m) => {
    const zs = avgOf(wrong.filter((r) => r.model === m && r.method === '1_Zero_Shot'), (r) => r.conf);
    const cot = avgOf(wrong.filter((r) => r.model === m && (r.method === '3_Zero_Shot_CoT' || r.method === '4_Few_Shot_CoT')), (r) => r.conf);
    return { m, zs, cot, d: cot - zs };
  });
  const allDown = per.every((p) => p.d < 0);
  setText('f2-claim', `Khi trả lời sai, các model vẫn tự tin trung bình ${fmt(overall)}% — ${allDown ? 'CoT chỉ hạ độ tự tin khi sai đi vài điểm' : 'CoT không hạ được độ tự tin khi sai một cách nhất quán'}`);
  setText('f2-read', `Đọc: cột = độ tự tin tự báo cáo trung bình trên các lượt SAI (rê chuột để xem số lượt sai, độ tự tin khi đúng và accuracy). Trục y không bắt đầu từ 0 (từ ${lo}%). Số lượt sai của Zero-Shot nhiều hơn hẳn CoT, nên so sánh các cột mang tính tham khảo.`);
  setText('f2-con', `<b>Kết luận:</b> chênh lệch độ tự tin khi sai, CoT (Zero-CoT + Few-CoT) so với Zero-Shot: ${per.map((p) => `${p.m} ${sgn(p.d)} điểm (${fmt(p.zs)}% → ${fmt(p.cot)}%)`).join('; ')}. `
    + `Độ tự tin trung bình khi đúng là ${fmt(avgOf(rows.filter((r) => r.correct), (r) => r.conf))}% so với ${fmt(overall)}% khi sai: `
    + 'khoảng cách quá hẹp để dùng điểm tự tin tự báo cáo làm ngưỡng tin cậy khi chưa hiệu chỉnh (xem Gói 3 ở Tab 4).');
}

function filteredErrors() {
  const m = $('#e-model').value, l = $('#e-lang').value, k = $('#e-method').value, d = $('#e-diff').value, f = $('#e-faith').value;
  const s = $('#e-search').value.trim().toLowerCase();
  return D.errors.filter((r) => (!m || r.Model === m) && (!l || r.Language === l) && (!k || r.Method === k) && (!d || r.Difficulty === d)
    && (!f || (f === '__NA' ? !FAITH.includes(r.Faithfulness) : r.Faithfulness === f)) && (!s || r.ID.toLowerCase().includes(s)));
}

function renderErrors() {
  E.rows = filteredErrors();
  const pages = Math.max(1, Math.ceil(E.rows.length / E.size));
  E.page = Math.min(E.page, pages - 1);
  const slice = E.rows.slice(E.page * E.size, (E.page + 1) * E.size);
  $('#e-count').textContent = `${E.rows.length.toLocaleString('vi-VN')} / ${D.errors.length.toLocaleString('vi-VN')} lượt trả lời sai khớp bộ lọc`;
  $('#e-table').innerHTML = '<thead><tr><th>ID</th><th>Model</th><th>NN</th><th>Phương pháp</th><th>Chuẩn</th><th>Đoán</th><th class="num">Tự tin</th></tr></thead><tbody>'
    + (slice.length ? slice.map((r) => `<tr class="clickable${E.sel === r.Task_Key ? ' sel' : ''}" data-k="${esc(r.Task_Key)}" tabindex="0">
      <td class="font-mono">${esc(r.ID)}</td><td>${esc(r.Model)}</td><td>${esc(r.Language)}</td><td>${esc(SHORT[r.Method] || r.Method)}</td>
      <td class="font-mono">${esc(r.Ground_Truth)}</td><td class="font-mono">${esc(r.Predicted ?? '∅')}${r.Status !== 'OK' ? ' <span class="badge badge-neutral">' + esc(r.Status) + '</span>' : ''}</td>
      <td class="num">${r.Confidence == null ? '—' : fmt(r.Confidence, 0) + '%'}</td></tr>`).join('')
      : '<tr><td colspan="7" class="text-muted">Không có lượt sai nào khớp bộ lọc.</td></tr>') + '</tbody>';
  $$('#e-table tr.clickable').forEach((tr) => {
    const open = () => showError(tr.dataset.k);
    tr.addEventListener('click', open);
    tr.addEventListener('keydown', (e) => { if (e.key === 'Enter') open(); });
  });
  $('#e-pager').innerHTML = `<button class="btn" id="e-prev" ${E.page === 0 ? 'disabled' : ''}>← Trước</button>
    <span class="text-muted">Trang ${E.page + 1} / ${pages}</span>
    <button class="btn" id="e-next" ${E.page >= pages - 1 ? 'disabled' : ''}>Sau →</button>`;
  $('#e-prev').addEventListener('click', () => { E.page--; renderErrors(); });
  $('#e-next').addEventListener('click', () => { E.page++; renderErrors(); });
}

async function showError(taskKey) {
  E.sel = taskKey;
  $$('#e-table tr.clickable').forEach((tr) => tr.classList.toggle('sel', tr.dataset.k === taskKey));
  const r = D.errors.find((x) => x.Task_Key === taskKey);
  const q = D.qById.get(r.ID);
  const box = $('#e-detail');
  box.innerHTML = `<div class="method-card">
    <header>
      <div class="flex flex-wrap items-center gap-2">
        <b class="font-mono">${esc(r.ID)}</b><span class="badge badge-err">Sai</span>
        <span class="badge badge-neutral">${esc(r.Model)} · ${esc(r.Language)}</span>
        <span class="badge badge-neutral">${esc(SHORT[r.Method] || r.Method)}</span>
        <span class="badge badge-neutral">${esc(DIFF_VI[r.Difficulty] || r.Difficulty)} · ${esc(r.Category)}</span>
        <button class="btn ml-auto !py-1 !px-2 text-xs" id="e-open">Mở ở Tab 1 →</button>
      </div>
      <p class="text-sm mt-2 leading-relaxed whitespace-pre-line">${esc(q ? (r.Language === 'VI' ? q.q_vi : q.q_en) : '')}</p>
    </header>
    <div class="stats">
      <div><span>Chuẩn / đoán</span><b class="font-mono">${esc(r.Ground_Truth)} / ${esc(r.Predicted ?? '∅')}</b></div>
      <div><span>Tự tin</span><b>${r.Confidence == null ? '—' : fmt(r.Confidence, 0) + '%'}</b></div>
      <div title="${esc(FAITH_HELP[r.Faithfulness] || '')}"><span>Faithfulness</span><b>${esc(FAITH.includes(r.Faithfulness) ? r.Faithfulness.replace('->', '→') : 'N/A')}</b></div>
      <div><span>Trạng thái</span><b>${esc(r.Status)}</b></div>
      <div><span>Token</span><b>${fmt(r.Total_Tokens, 0)}</b></div>
      <div><span>Độ trễ</span><b>${fmt(r.Latency_s, 2)} s</b></div>
    </div>
    <pre class="raw" id="e-raw">Đang tải lời giải…</pre></div>`;
  $('#e-open').addEventListener('click', () => { selectQuestion(r.ID, r.Model, r.Language, r.Method); location.hash = 't1'; });
  try {
    const logs = await loadLogs(r.ID);
    if (E.sel !== taskKey) return;
    const lg = logs[`${r.Model}|${r.Language}|${r.Method}`];
    $('#e-raw').textContent = lg && lg.raw ? lg.raw : '(trống)';
  } catch (e) {
    $('#e-raw').textContent = `Không tải được lời giải: ${e.message}`;
  }
}

// ================================================================== TAB 4
function renderTab4() {
  const o = D.opt;
  const es = o.early_stop;
  const totalRuns = es.reduce((s, e) => s + e.avg_runs * e.n, 0);
  const totalN = es.reduce((s, e) => s + e.n, 0);
  const savedAll = (1 - totalRuns / (5 * totalN)) * 100;
  const qlat = avgOf(D.recs.filter((r) => r.model === 'Qwen_7B' && r.method === '1_Zero_Shot'), (r) => r.lat);
  const qlatCot = avgOf(D.recs.filter((r) => r.model === 'Qwen_7B' && r.method === '3_Zero_Shot_CoT'), (r) => r.lat);

  $('#arch').innerHTML = `<div class="arch">
    <div class="node"><b>1 · Bộ đề</b>150 câu <code>data_test</code>, VI + EN, 3 mức độ, 2 dạng đáp án, 15 nhóm leak</div><div class="arrow">→</div>
    <div class="node key"><b>2 · DPMemoTable</b>Khoá <code>ID_Model_Lang_Method</code>, tra O(1). Lượt đã có thì bỏ qua; ghi checkpoint CSV sau mỗi 10 bản ghi</div><div class="arrow">→</div>
    <div class="node"><b>3 · 6 phương pháp</b>Zero/Few-Shot (256 token), CoT (tối đa 1.024–2.048 token), nhiệt độ 0</div><div class="arrow">→</div>
    <div class="node key"><b>4 · Self-Consistency + Branch &amp; Bound</b>T=0.7, tối đa 5 luồng; dừng ngay khi một đáp án đạt 3 phiếu</div><div class="arrow">→</div>
    <div class="node"><b>5 · parse_output + chấm</b>Tag <code>[Final Answer]</code>, bỏ 1 dòng tự tin ở cuối, gán nhãn Faithfulness</div><div class="arrow">→</div>
    <div class="node"><b>6 · Xuất bản</b>Workbook 5 sheet → <code>build_web_data.py</code> → JSON tĩnh trên GitHub Pages</div>
  </div>`;
  const dp = o.dp;
  $('#arch-cards').innerHTML = `
    <div class="card"><div class="filter-label">Quy hoạch động · DPMemoTable</div>
      <div class="kpi">${dp.leaked_copied.toLocaleString('vi-VN')}/${dp.leaked_unleaked.toLocaleString('vi-VN')}</div>
      <p class="text-sm text-muted mt-1">lượt pp6 trên câu Unleaked (cả 3 model) lấy thẳng từ pp4 trong bảng DP: prompt giống hệt nên không gọi lại API. Ngoài ra ${dp.qwen_sc_reused_zcot}/${dp.qwen_sc_total} lượt SC của Qwen có luồng 1 trùng nguyên văn Zero-CoT (cờ <code>SC_REUSE_ZCOT</code>).</p></div>
    <div class="card"><div class="filter-label">Cắt tỉa nhánh cận · Early Stopping</div>
      <div class="kpi">−${fmt(savedAll)}%</div>
      <p class="text-sm text-muted mt-1">lời gọi Self-Consistency so với chạy đủ 5 luồng (${totalN.toLocaleString('vi-VN')} lượt SC, trung bình ${fmt(totalRuns / totalN, 2)} luồng/lượt).</p></div>
    <div class="card"><div class="filter-label">Self-host Qwen2.5-7B</div>
      <div class="kpi">${fmt(qlat, 1)} s → ${fmt(qlatCot, 1)} s</div>
      <p class="text-sm text-muted mt-1">độ trễ TB Zero-Shot → Zero-CoT. <code>qwen2.5:7b</code> chạy qua Ollama (tag mặc định là bản lượng tử hoá Q4_K_M, ~4-bit) sau endpoint kiểu OpenAI; 6 luồng song song, <code>CircuitBreaker</code> tự ngắt sau 8 lỗi liên tiếp và lưu checkpoint.</p></div>`;

  // o1: phân bố số luồng SC
  const runsK = ['3', '4', '5'];
  const rc = { 3: '#86b6ef', 4: '#3987e5', 5: '#184f95' };
  plot('o1', runsK.map((k) => ({
    type: 'bar', name: `${k} luồng`, x: es.map((e) => e.model), y: es.map((e) => e.dist[k] || 0), marker: { color: rc[k] },
    text: es.map((e) => (e.dist[k] ? `${fmt((100 * e.dist[k]) / e.n, 0)}%` : '')), textposition: 'inside', insidetextanchor: 'middle',
    textfont: { color: k === '3' ? '#1a2233' : '#ffffff' },
    hovertemplate: `%{x}: %{y} lượt dừng sau ${k} luồng<extra></extra>`,
  })), baseLayout({
    barmode: 'stack', bargap: 0.45,
    xaxis: Object.assign({}, AX), yaxis: Object.assign({}, AX, { title: { text: 'Số lượt Self-Consistency', font: { size: 11 } } }),
    margin: { l: 56, r: 18, t: 16, b: 40 },
  }));
  const most = es.reduce((a, b) => (b.avg_runs > a.avg_runs ? b : a));
  setText('o1-claim', `Dừng sớm khi đã đủ 3/5 phiếu cắt ${fmt(savedAll)}% lời gọi Self-Consistency mà không đổi đáp án đa số`);
  setText('o1-read', 'Đọc: với tối đa 5 luồng, khi một đáp án đã có 3 phiếu thì 2 luồng còn lại không thể lật kết quả đa số, nên đó là “cận” để cắt nhánh. Lưu ý: % đồng thuận (SC_Agree_Pct) được tính trên số luồng đã chạy thực tế.');
  setText('o1-con', `<b>Kết luận:</b> ${es.map((e) => `${e.model} tiết kiệm ${fmt(e.calls_saved_pct)}% (TB ${fmt(e.avg_runs, 2)} luồng)`).join('; ')}. `
    + `${most.model} cần nhiều luồng nhất vì các luồng của nó bất đồng thường xuyên hơn, đúng với một model nhỏ kém ổn định.`);

  // o2: bảng router
  const rules = o.rules;
  $('#o2-table').innerHTML = '<thead><tr><th>Model</th><th>Độ khó</th><th>Router chọn</th><th class="num">Accuracy</th><th class="num">Token TB</th><th class="num">Zero-Shot</th><th>Tốt nhất (nếu khác)</th></tr></thead><tbody>'
    + rules.map((r) => `<tr><td>${esc(r.model)}</td><td>${esc(DIFF_VI[r.difficulty])}</td>
      <td><span class="inline-block w-2.5 h-2.5 rounded-full mr-1.5 align-middle" style="background:${METHOD_COLORS[r.method]}"></span><b>${esc(SHORT[r.method])}</b></td>
      <td class="num">${fmt(r.acc)}%</td><td class="num">${fmt(r.tok, 0)}</td>
      <td class="num text-muted">${fmt(r.zs_acc)}% · ${fmt(r.zs_tok, 0)} tok</td>
      <td class="text-muted">${r.best_method === r.method ? '—' : `${esc(SHORT[r.best_method])}: ${fmt(r.best_acc)}% · ${fmt(r.best_tok, 0)} tok`}</td></tr>`).join('')
    + '</tbody>';
  const freq = {};
  rules.forEach((r) => { freq[r.method] = (freq[r.method] || 0) + 1; });
  const [topM, topC] = Object.entries(freq).sort((a, b) => b[1] - a[1])[0];
  const scCells = rules.filter((r) => r.method === '5_Self_Consistency').map((r) => `${r.model}/${DIFF_VI[r.difficulty]}`);
  setText('o2-claim', `Quy tắc điều phối: ${SHORT[topM]} là lựa chọn ở ${topC}/${rules.length} ô` + (scCells.length ? `; Self-Consistency chỉ đáng tiền ở ${scCells.join(', ')}` : '; không ô nào cần Self-Consistency'));
  setText('o2-read', `Đọc: với mỗi model × độ khó, router chọn phương pháp <b>rẻ nhất</b> (token TB) có accuracy cách phương pháp tốt nhất không quá ${fmt(o.router_tol, 0)} điểm. Leaked bị loại vì dùng gợi ý rò rỉ. Gộp cả VI và EN (100 lượt mỗi ô).`);

  // o3: mô phỏng router
  const sim = o.router_sim;
  const traces = [];
  sim.forEach((s) => {
    const ks = ROUTABLE;
    traces.push({
      type: 'scatter', mode: 'markers+text', name: `${s.model} · luôn dùng 1 phương pháp`, legendgroup: s.model,
      x: ks.map((k) => s.fixed[k].tok), y: ks.map((k) => s.fixed[k].acc), text: ks.map((k) => SHORT[k]), textposition: 'bottom center',
      textfont: { size: 10, color: '#5b6170' }, marker: { size: 9, color: MODEL_COLORS[s.model], symbol: MODEL_SYMBOL[s.model], opacity: 0.55 },
      hovertemplate: `${s.model} · %{text}<br>%{x:.0f} token · %{y:.1f}%<extra></extra>`,
    });
    traces.push({
      type: 'scatter', mode: 'markers', name: `${s.model} · Router`, legendgroup: s.model,
      x: [s.router_tok], y: [s.router_acc], marker: { size: 18, color: MODEL_COLORS[s.model], symbol: 'star', line: { color: '#1a2233', width: 1 } },
      hovertemplate: `${s.model} · Router<br>%{x:.0f} token · %{y:.1f}%<extra></extra>`,
    });
  });
  plot('o3', traces, baseLayout({
    xaxis: Object.assign({}, AX, { type: 'log', title: { text: 'Token trung bình / lượt — thang log', font: { size: 11 } } }),
    yaxis: Object.assign({}, AX, { title: { text: 'Accuracy (%)', font: { size: 11 } } }),
    margin: { l: 56, r: 18, t: 16, b: 40 }, legend: { orientation: 'h', x: 0, y: -0.2, yanchor: 'top', font: { size: 11 } },
  }));
  const cmp = sim.map((s) => {
    const sc = s.fixed['5_Self_Consistency'];
    return { m: s.model, dAcc: s.router_acc - sc.acc, save: (1 - s.router_tok / sc.tok) * 100 };
  });
  const maxLoss = Math.max(...cmp.map((c) => -c.dAcc));
  setText('o3-claim', `Router giữ accuracy ${maxLoss <= 0 ? 'bằng hoặc hơn' : `trong vòng ${fmt(maxLoss)} điểm của`} Self-Consistency mà dùng ít hơn ${fmt(Math.min(...cmp.map((c) => c.save)), 0)}–${fmt(Math.max(...cmp.map((c) => c.save)), 0)}% token`);
  setText('o3-read', 'Đọc: ngôi sao = router; các điểm nhỏ cùng màu = luôn dùng một phương pháp cho mọi câu. Trục x dạng log. <b>Giới hạn:</b> quy tắc được chọn trên chính 150 câu test (in-sample) và giả định biết trước độ khó của câu, nên đây là cận trên; cần kiểm lại trên tập độc lập trước khi triển khai.');
  setText('o3-con', `<b>Kết luận:</b> ${cmp.map((c) => `${c.m}: ${sgn(c.dAcc)} điểm, tiết kiệm ${fmt(c.save, 0)}% token so với luôn dùng SC`).join('; ')}. Chọn prompt theo độ khó rẻ hơn nhiều so với áp một phương pháp đắt cho mọi câu.`);

  renderO4();
}

function renderO4() {
  const f = D.opt.fixes;
  const SH = { 'Premise Extraction': 'Premise', 'Strict CoT': 'Strict CoT', PAL: 'PAL', 'Post-generation NLI guardrail': 'NLI', 'Logprob & Temperature Calibration': 'Calibration' };
  const order = ['Premise', 'Strict CoT', 'PAL', 'NLI', 'Calibration'];
  const sl = Object.fromEntries(f.small_large.map((r) => [SH[r.method] || r.method, r]));
  const qw = Object.fromEntries(f.qwen.map((r) => [r.Method, r]));
  if (!f.small_large.length && !f.qwen.length) { $('#o4').innerHTML = '<div class="p-6 text-sm text-muted">Chưa có dữ liệu Gói 3.</div>'; return; }
  const slRate = order.map((k) => (sl[k] ? (100 * sl[k].fixed) / sl[k].n : NaN));
  const qwRate = order.map((k) => (qw[k] ? (100 * qw[k].Fixed) / Math.max(1, qw[k].N_valid - qw[k].Before_correct) : NaN));
  plot('o4', [
    { type: 'bar', name: 'Small/Large (20 ca vốn sai)', x: order, y: slRate, marker: { color: '#2a78d6' }, text: order.map((k) => (sl[k] ? `${sl[k].fixed}/${sl[k].n}` : '')), textposition: 'outside', cliponaxis: false, hovertemplate: 'Small/Large · %{x}: %{y:.0f}% ca sai được sửa<extra></extra>' },
    {
      type: 'bar', name: 'Qwen 7B (ca Qwen vốn sai trong 20 câu)', x: order, y: qwRate, marker: { color: MODEL_COLORS.Qwen_7B },
      text: order.map((k) => (qw[k] ? `${qw[k].Fixed}/${qw[k].N_valid - qw[k].Before_correct} · −${qw[k].Regression}` : '')), textposition: 'outside', cliponaxis: false,
      customdata: order.map((k) => (qw[k] ? [qw[k].Regression, qw[k].Delta] : [null, null])),
      hovertemplate: 'Qwen 7B · %{x}: sửa %{y:.0f}%<br>Làm hỏng %{customdata[0]} ca vốn đúng · ròng %{customdata[1]:+} điểm<extra></extra>',
    },
  ], baseLayout({
    barmode: 'group', bargap: 0.3,
    xaxis: Object.assign({}, AX), yaxis: Object.assign({}, AX, { range: [0, 100], title: { text: 'Tỷ lệ sửa được (%)', font: { size: 11 } } }),
    margin: { l: 56, r: 18, t: 22, b: 40 },
  }));
  // "Tốt nhất" chỉ xét can thiệp áp dụng được cho cả 20 ca (PAL chỉ chạy trên câu số nên mẫu nhỏ hơn).
  const slK = order.filter((k) => sl[k]);
  const fullN = Math.max(...slK.map((k) => sl[k].n));
  const bestSL = slK.filter((k) => sl[k].n === fullN).reduce((a, b) => (slRate[order.indexOf(b)] > slRate[order.indexOf(a)] ? b : a));
  const qOrder = order.filter((k) => qw[k]);
  const bestQ = qOrder.reduce((a, b) => (qw[b].Delta > qw[a].Delta || (qw[b].Delta === qw[a].Delta && qw[b].Regression < qw[a].Regression) ? b : a));
  const worstQ = qOrder.reduce((a, b) => (qw[b].Delta < qw[a].Delta ? b : a));
  const totFix = qOrder.reduce((s, k) => s + qw[k].Fixed, 0), totReg = qOrder.reduce((s, k) => s + qw[k].Regression, 0);
  setText('o4-claim', `Gói 3: ${bestSL} sửa được ${fmt(slRate[order.indexOf(bestSL)], 0)}% ca sai của Small/Large; với Qwen 7B, tốt nhất là ${bestQ} (${sgn(qw[bestQ].Delta, 0)} điểm), tệ nhất là ${worstQ} (${sgn(qw[worstQ].Delta, 0)} điểm)`);
  setText('o4-read', `Đọc: nhãn cột = số ca sai được lật thành đúng / số ca vốn sai; với Qwen thêm “−n” = số ca vốn đúng bị can thiệp làm sai đi (regression). 20 câu được chọn từ lỗi của Small/Large nên Qwen đã đúng sẵn ${qw[bestQ].Before_correct}/${qw[bestQ].N_valid} câu. PAL chỉ áp dụng cho câu số.`);
  const narrow = slK.filter((k) => sl[k].n < fullN && slRate[order.indexOf(k)] > slRate[order.indexOf(bestSL)]);
  setText('o4-con', `<b>Kết luận:</b> ${narrow.map((k) => `${k} có tỷ lệ cao hơn (${sl[k].fixed}/${sl[k].n}) nhưng chỉ áp dụng được cho ${sl[k].n} câu số. `).join('')}Trên Qwen 7B, 5 can thiệp sửa tổng ${totFix} ca nhưng làm hỏng ${totReg} ca. Can thiệp hiệu quả với model mạnh không tự động hiệu quả với model yếu, nên router nên chọn can thiệp theo model, không chỉ theo độ khó.`);
}

boot();
