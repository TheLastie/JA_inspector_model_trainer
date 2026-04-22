import os
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from flask import Flask, render_template_string, request, jsonify, send_file
from exp_manager2 import manager
app = Flask(__name__)

DATA_DIR = 'datasets'
MODELS_DIR = 'models'
ANALYZERS_DIR = 'analyzers'
CSV_LOG = 'experiments.csv'
CONFIG_FILE = 'dashboard_config.json'
TRAINER_PID_FILE = 'massive_trainer.pid'

def load_config():
    default_config = {
        'default_params': {
            'hidden_size': 128, 'latent_size': 64, 'seq_len': 30, 'batch_size': 32,
            'lr': 0.005, 'dropout': 0.2, 'num_layers': 2, 'epochs': 100,
            'early_stop_patience': 20, 'autoregressive': True, 'teacher_forcing': 0.5,
            'teacher_forcing_decay': 0.99, 'scheduler': 'plateau', 'test_ratio': 0.1
        },
        'analyzer_defaults': {
            'analyzer_hidden': 128, 'freeze_encoder': True, 'reg_weight': 0.5,
            'error_prob': 0.6, 'max_deviation': 0.5, 'samples_per_video': 200,
            'epochs': 50, 'early_stop_patience': 10, 'batch_size': 32, 'lr': 0.001
        }
    }
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
            default_config.update(config)
    return default_config

config = load_config()

def get_exercise_list():
    if not os.path.exists(DATA_DIR):
        return []
    return sorted([f.replace('.json', '') for f in os.listdir(DATA_DIR) if f.endswith('.json')])

def load_experiments():
    if not os.path.exists(CSV_LOG):
        return []
    experiments = []
    with open(CSV_LOG, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f, restval='')
        for row in reader:
            if row.get('experiment_id'):
                experiments.append(row)
    return experiments

def get_available_models(model_type='rae'):
    base_dir = MODELS_DIR if model_type == 'rae' else ANALYZERS_DIR
    models = []
    if not os.path.exists(base_dir):
        return models
    for exercise in os.listdir(base_dir):
        ex_path = os.path.join(base_dir, exercise)
        if not os.path.isdir(ex_path):
            continue
        for run in sorted(os.listdir(ex_path), reverse=True):
            run_path = os.path.join(ex_path, run)
            if (model_type == 'rae' and run.startswith('run_')) or (model_type == 'analyzer' and run.startswith('analyzer_')):
                model_file = os.path.join(run_path, 'best_model.pth' if model_type == 'rae' else 'best_analyzer.pth')
                log_file = os.path.join(run_path, 'training_log.txt')
                plot_file = os.path.join(run_path, 'loss_plot.png')
                info = {
                    'exercise': exercise,
                    'run': run,
                    'model_path': model_file if os.path.exists(model_file) else None,
                    'log_path': log_file,
                    'plot_path': plot_file,
                    'timestamp': run.split('_', 1)[1] if '_' in run else run,
                    'best_epoch': '',
                    'best_test_loss': '',
                    'accuracy': ''
                }
                if os.path.exists(log_file):
                    with open(log_file, 'r') as f:
                        content = f.read()
                        for line in content.split('\n'):
                            if 'Best epoch:' in line or 'Best Val Loss:' in line:
                                parts = line.split('|') if '|' in line else [line]
                                if 'Best epoch:' in line:
                                    info['best_epoch'] = parts[0].split(':')[1].strip()
                                    info['best_test_loss'] = parts[1].split(':')[1].strip() if len(parts) > 1 else ''
                                elif 'Best Val Loss:' in line:
                                    info['best_test_loss'] = line.split(':')[1].strip()
                            if 'Best Val Accuracy:' in line:
                                info['accuracy'] = line.split(':')[1].strip().replace('%', '')
                models.append(info)
    return models

def parse_training_log(log_path):
    if not os.path.exists(log_path):
        return None, None
    test_losses = []
    with open(log_path, 'r') as f:
        lines = f.readlines()
    in_table = False
    for line in lines:
        line = line.strip()
        if line.startswith('Epoch'):
            in_table = True
            continue
        if in_table and line:
            parts = line.split('\t')
            if len(parts) >= 3:
                try:
                    test_losses.append(float(parts[2]))
                except:
                    pass
    return test_losses

# -------------------- HTML шаблон (сокращён для экономии места, но полный функционал) --------------------
INDEX_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Neuro Dashboard</title>
    <style>
    #compareChart { width: 100% !important; height: 500px !important; max-height: 80vh; }
    body { font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background: #1e1e2f; color: #e0e0e0; }
    .header { display: flex; justify-content: space-between; margin-bottom: 20px; }
    .section { background: #2a2a3c; border-radius: 12px; padding: 20px; margin-bottom: 25px; }
    button { background: #5a6abf; color: white; border: none; padding: 8px 16px; border-radius: 6px; cursor: pointer; margin-right: 8px; }
    button.danger { background: #c44; }
    button.success { background: #3a9b5e; }
    button.warning { background: #e67e22; }
    .tabs { display: flex; border-bottom: 2px solid #444; margin-bottom: 20px; }
    .tab { padding: 12px 24px; cursor: pointer; border: none; background: none; font-size: 16px; color: #b0b0d0; }
    .tab.active { border-bottom: 3px solid #5a6abf; font-weight: bold; color: #fff; }
    .tab-content { display: none; }
    .tab-content.active { display: block; }
    .filter-bar { margin-bottom: 20px; display: flex; gap: 10px; flex-wrap: wrap; }
    .filter-input { padding: 8px; border: 1px solid #444; border-radius: 6px; background: #1e1e2f; color: #e0e0e0; }
    .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin-bottom: 20px; }
    .stat-card { background: #1e1e2f; padding: 15px; border-radius: 8px; text-align: center; }
    .stat-value { font-size: 28px; font-weight: bold; color: #5a6abf; }
    table { width: 100%; border-collapse: collapse; margin-top: 10px; }
    th, td { padding: 10px; text-align: left; border-bottom: 1px solid #444; }
    th { background: #1e1e2f; cursor: pointer; }
    tr:hover { background: #2a2a4a; }
    .modal { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.7); z-index: 1000; }
    .modal-content { background: #2a2a3c; margin: 3% auto; padding: 25px; width: 85%; max-width: 1000px; border-radius: 12px; }
    .close { float: right; font-size: 28px; font-weight: bold; cursor: pointer; color: #aaa; }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-zoom@1.2.1/dist/chartjs-plugin-zoom.min.js"></script>
</head>
<body>
    <div class="header">
        <h1>🧠 Neural Training Dashboard</h1>
        <div>
            <button onclick="refreshAll()">🔄 Refresh</button>
            <button onclick="exportData()">📥 Export CSV</button>
            <button class="warning" onclick="startMassiveTrainer()">🔥 Start Massive Trainer</button>
            <button class="danger" onclick="stopMassiveTrainer()" style="display:none;" id="stopMassiveBtn">⏹️ Stop</button>
        </div>
    </div>
    <div class="tabs">
        <button class="tab active" onclick="showTab('experiments')">Experiments</button>
        <button class="tab" onclick="showTab('rae')">RAE Models</button>
        <button class="tab" onclick="showTab('analyzer')">Analyzer Models</button>
        <button class="tab" onclick="showTab('compare')">Compare</button>
    </div>

    <!-- Experiments Tab -->
    <div id="experiments-tab" class="tab-content active">
        <div class="section">
            <h2>📊 Training Summary</h2>
            <div id="massive-stats" class="stats-grid"></div>
        </div>
        <div class="section">
            <h2>📋 Experiments</h2>
            <div class="filter-bar">
                <input type="text" id="exp-filter-exercise" placeholder="Exercise" class="filter-input" oninput="filterExperiments()">
                <input type="text" id="exp-filter-type" placeholder="Type" class="filter-input" oninput="filterExperiments()">
                <input type="text" id="exp-filter-status" placeholder="Status" class="filter-input" oninput="filterExperiments()">
                <button class="danger" onclick="stopSelectedExperiments()">Stop Selected</button>
                <button onclick="deleteSelectedExperiments()">Delete Selected</button>
            </div>
            <div style="overflow-x: auto;">
                <table id="experiments-table">
                    <thead>
                        <tr>
                            <th><input type="checkbox" id="select-all-exp" onchange="toggleSelectAll('exp')"></th>
                            <th onclick="sortTable('experiments-table', 1)">ID</th>
                            <th onclick="sortTable('experiments-table', 2)">Type</th>
                            <th onclick="sortTable('experiments-table', 3)">Exercise</th>
                            <th onclick="sortTable('experiments-table', 4)">Status</th>
                            <th onclick="sortTable('experiments-table', 5)">Test Loss</th>
                            <th onclick="sortTable('experiments-table', 6)">Epoch</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody id="experiments-body"></tbody>
                </table>
            </div>
        </div>
    </div>

    <!-- RAE Models Tab -->
    <div id="rae-tab" class="tab-content">
        <div class="section">
            <h2>📊 RAE Models</h2>
            <div class="filter-bar">
                <input type="text" id="rae-filter-exercise" placeholder="Exercise" class="filter-input" oninput="filterRae()">
                <button onclick="compareSelected('rae')">Compare Selected</button>
            </div>
            <div style="overflow-x: auto;">
                <table id="rae-table">
                    <thead>
                        <tr>
                            <th><input type="checkbox" id="select-all-rae" onchange="toggleSelectAll('rae')"></th>
                            <th onclick="sortTable('rae-table', 1)">Exercise</th>
                            <th onclick="sortTable('rae-table', 2)">Run</th>
                            <th onclick="sortTable('rae-table', 3)">Test Loss</th>
                            <th onclick="sortTable('rae-table', 4)">Epoch</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody id="rae-body"></tbody>
                </table>
            </div>
        </div>
    </div>

    <!-- Analyzer Models Tab -->
    <div id="analyzer-tab" class="tab-content">
        <div class="section">
            <h2>🔍 Analyzer Models</h2>
            <div class="filter-bar">
                <input type="text" id="analyzer-filter-exercise" placeholder="Exercise" class="filter-input" oninput="filterAnalyzer()">
                <button onclick="compareSelected('analyzer')">Compare Selected</button>
            </div>
            <div style="overflow-x: auto;">
                <table id="analyzer-table">
                    <thead>
                        <tr>
                            <th><input type="checkbox" id="select-all-analyzer" onchange="toggleSelectAll('analyzer')"></th>
                            <th onclick="sortTable('analyzer-table', 1)">Exercise</th>
                            <th onclick="sortTable('analyzer-table', 2)">Run</th>
                            <th onclick="sortTable('analyzer-table', 3)">Accuracy</th>
                            <th onclick="sortTable('analyzer-table', 4)">Val Loss</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody id="analyzer-body"></tbody>
                </table>
            </div>
        </div>
    </div>

    <!-- Compare Tab -->
    <div id="compare-tab" class="tab-content">
        <div class="section">
            <h2>📈 Compare</h2>
            <canvas id="compareChart" style="max-height: 400px;"></canvas>
        </div>
    </div>

    <!-- Plot Modal -->
    <div id="plotModal" class="modal">
        <div class="modal-content">
            <span class="close" onclick="closePlot()">&times;</span>
            <img id="plotImage" src="" style="max-width:100%">
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script>
    let expData = [], raeData = [], analyzerData = [], compareChart = null;
let currentSort = {table: null, col: -1, asc: true};

// ---------- Инициализация ----------
function showTab(name) {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    document.querySelector(`.tab[onclick="showTab('${name}')"]`).classList.add('active');
    document.getElementById(`${name}-tab`).classList.add('active');
    if (name === 'experiments') refreshExperiments();
    else if (name === 'rae') refreshRae();
    else if (name === 'analyzer') refreshAnalyzer();
}

async function refreshExperiments() {
    const res = await fetch('/api/experiments');
    expData = await res.json();
    renderExperiments(expData);
    updateStats();
}

async function refreshRae() {
    const res = await fetch('/api/models?type=rae');
    raeData = await res.json();
    renderRae(raeData);
}

async function refreshAnalyzer() {
    const res = await fetch('/api/models?type=analyzer');
    analyzerData = await res.json();
    renderAnalyzer(analyzerData);
}

function renderExperiments(data) {
    const tbody = document.getElementById('experiments-body');
    tbody.innerHTML = data.map(exp => `
        <tr>
            <td><input type="checkbox" class="exp-checkbox" value="${exp.experiment_id}"></td>
            <td>${exp.experiment_id || ''}</td>
            <td>${exp.type || ''}</td>
            <td>${exp.exercise || ''}</td>
            <td>${exp.status || ''}</td>
            <td>${exp.best_test_loss || ''}</td>
            <td>${exp.best_epoch || ''}</td>
            <td>
                ${exp.status === 'running' ? `<button class="danger" onclick="stopExp('${exp.experiment_id}')">Stop</button>` : ''}
                ${exp.status === 'completed' && exp.run_dir ? `<button onclick="showPlot('${exp.run_dir}/loss_plot.png')">📈</button>` : ''}
            </td>
        </tr>
    `).join('');
}

function renderRae(data) {
    const tbody = document.getElementById('rae-body');
    tbody.innerHTML = data.map(m => `
        <tr>
            <td><input type="checkbox" class="rae-checkbox" value="${m.model_path}"></td>
            <td>${m.exercise}</td>
            <td>${m.run}</td>
            <td>${m.best_test_loss || ''}</td>
            <td>${m.best_epoch || ''}</td>
            <td><button onclick="showPlot('${m.plot_path}')">📈</button></td>
        </tr>
    `).join('');
}

function renderAnalyzer(data) {
    const tbody = document.getElementById('analyzer-body');
    tbody.innerHTML = data.map(m => `
        <tr>
            <td><input type="checkbox" class="analyzer-checkbox" value="${m.model_path}"></td>
            <td>${m.exercise}</td>
            <td>${m.run}</td>
            <td>${m.accuracy ? m.accuracy + '%' : ''}</td>
            <td>${m.best_test_loss || ''}</td>
            <td><button onclick="showPlot('${m.plot_path}')">📈</button></td>
        </tr>
    `).join('');
}

// ---------- Фильтрация ----------
function filterExperiments() {
    const ex = document.getElementById('exp-filter-exercise').value.toLowerCase();
    const type = document.getElementById('exp-filter-type').value.toLowerCase();
    const status = document.getElementById('exp-filter-status').value.toLowerCase();
    const filtered = expData.filter(e => 
        (e.exercise || '').toLowerCase().includes(ex) &&
        (e.type || '').toLowerCase().includes(type) &&
        (e.status || '').toLowerCase().includes(status)
    );
    renderExperiments(filtered);
}

function filterRae() {
    const ex = document.getElementById('rae-filter-exercise').value.toLowerCase();
    const filtered = raeData.filter(m => (m.exercise || '').toLowerCase().includes(ex));
    renderRae(filtered);
}

function filterAnalyzer() {
    const ex = document.getElementById('analyzer-filter-exercise').value.toLowerCase();
    const filtered = analyzerData.filter(m => (m.exercise || '').toLowerCase().includes(ex));
    renderAnalyzer(filtered);
}

// ---------- Сортировка ----------
function sortTable(tableId, colIdx) {
    const tbody = document.getElementById(tableId + '-body');
    const rows = Array.from(tbody.querySelectorAll('tr'));
    const asc = currentSort.table === tableId && currentSort.col === colIdx ? !currentSort.asc : true;
    currentSort = {table: tableId, col: colIdx, asc: asc};
    rows.sort((a, b) => {
        let aVal = a.children[colIdx]?.textContent || '';
        let bVal = b.children[colIdx]?.textContent || '';
        if (!isNaN(parseFloat(aVal)) && !isNaN(parseFloat(bVal))) {
            aVal = parseFloat(aVal);
            bVal = parseFloat(bVal);
        }
        return asc ? (aVal > bVal ? 1 : -1) : (aVal < bVal ? 1 : -1);
    });
    tbody.innerHTML = '';
    rows.forEach(r => tbody.appendChild(r));
}

// ---------- Выбор чекбоксов ----------
function toggleSelectAll(type) {
    const checkboxes = document.querySelectorAll(`.${type}-checkbox`);
    const selectAll = document.getElementById(`select-all-${type}`);
    checkboxes.forEach(cb => cb.checked = selectAll.checked);
}

function getSelectedCheckboxes(className) {
    return Array.from(document.querySelectorAll(`.${className}:checked`)).map(cb => cb.value);
}

// ---------- Управление экспериментами ----------
async function stopSelectedExperiments() {
    const ids = getSelectedCheckboxes('exp-checkbox');
    for (const id of ids) await stopExp(id);
    refreshExperiments();
}

async function deleteSelectedExperiments() {
    const ids = getSelectedCheckboxes('exp-checkbox');
    await fetch('/api/delete_experiments', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ids})
    });
    refreshExperiments();
}

async function stopExp(id) {
    await fetch('/api/stop_experiment', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({experiment_id: id})
    });
}

// ---------- Статистика ----------
async function updateStats() {
    const res = await fetch('/api/massive_stats');
    const s = await res.json();
    const container = document.getElementById('massive-stats');
    container.innerHTML = `
        <div class="stat-card"><div>Total</div><div class="stat-value">${s.total||0}</div></div>
        <div class="stat-card"><div>Completed</div><div class="stat-value">${s.completed||0}</div></div>
        <div class="stat-card"><div>Running</div><div class="stat-value">${s.running||0}</div></div>
        <div class="stat-card"><div>Failed</div><div class="stat-value">${s.failed||0}</div></div>
        <div class="stat-card"><div>Best RAE Loss</div><div class="stat-value">${s.best_rae_loss?.toFixed(6)||'-'}</div></div>
        <div class="stat-card"><div>Best Acc</div><div class="stat-value">${s.best_analyzer_acc?.toFixed(2)+'%'||'-'}</div></div>
    `;
}

// ---------- График ----------
function showPlot(path) {
    document.getElementById('plotImage').src = '/plot?path=' + encodeURIComponent(path);
    document.getElementById('plotModal').style.display = 'block';
}
function closePlot() { document.getElementById('plotModal').style.display = 'none'; }

// ---------- Сравнение моделей ----------
function getLogPathFromModel(model) {
    if (model.log_path) return model.log_path;
    if (model.model_path) {
        const dir = model.model_path.substring(0, model.model_path.lastIndexOf('/'));
        return dir + '/training_log.txt';
    }
    return null;
}

function compareSelected(type) {
    const dataArray = type === 'rae' ? raeData : analyzerData;
    const selectedCheckboxes = document.querySelectorAll(`.${type}-checkbox:checked`);
    const selectedValues = Array.from(selectedCheckboxes).map(cb => cb.value);
    if (!selectedValues.length) return alert('Select at least one model');

    const selectedModels = dataArray.filter(m => selectedValues.includes(m.model_path));
    const paths = selectedModels.map(m => getLogPathFromModel(m)).filter(p => p);
    if (!paths.length) return alert('No training logs found');

    fetch('/api/compare_models', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({log_paths: paths})
    })
    .then(r => r.json())
    .then(data => {
        const ctx = document.getElementById('compareChart').getContext('2d');
        if (compareChart) compareChart.destroy();
        compareChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: data[0]?.epochs || [],
                datasets: data.map((d, i) => ({
                    label: selectedModels[i].run || selectedModels[i].exercise + '_' + i,
                    data: d.test_losses,
                    borderColor: `hsl(${i * 60}, 70%, 60%)`,
                    borderWidth: 2,
                    pointRadius: 1,
                    tension: 0.1,
                    fill: false
                }))
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { title: { display: true, text: 'Epoch', color: '#e0e0e0' }, ticks: { color: '#e0e0e0' } },
                    y: { title: { display: true, text: 'Test Loss', color: '#e0e0e0' }, ticks: { color: '#e0e0e0' }, type: 'logarithmic' }
                },
                plugins: {
                    zoom: { zoom: { wheel: { enabled: true }, pinch: { enabled: true }, mode: 'xy' }, pan: { enabled: true, mode: 'xy' } },
                    legend: { labels: { color: '#e0e0e0' } }
                }
            }
        });
        showTab('compare');
    });
}

// ---------- Запуск внешнего тренера ----------
async function startMassiveTrainer() {
    const modelType = prompt('Enter model type (rae or analyzer):', 'rae');
    if (!modelType) return;
    const strategy = prompt('Enter strategy (grid or random):', 'grid');
    const workers = prompt('Enter number of workers:', '4');
    const res = await fetch('/api/start_massive_trainer', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({model_type: modelType, strategy: strategy, workers: parseInt(workers)})
    });
    const result = await res.json();
    if (result.status === 'ok') {
        alert('Massive trainer started!');
        document.getElementById('stopMassiveBtn').style.display = 'inline-block';
        refreshExperiments();
    } else {
        alert('Error: ' + result.message);
    }
}

async function stopMassiveTrainer() {
    await fetch('/api/stop_massive_trainer', {method:'POST'});
    document.getElementById('stopMassiveBtn').style.display = 'none';
}

function exportData() { window.location.href = '/api/export_csv'; }
function refreshAll() { refreshExperiments(); refreshRae(); refreshAnalyzer(); }

// ---------- Автообновление ----------
document.addEventListener('DOMContentLoaded', () => {
    refreshAll();
    setInterval(() => {
        if (document.getElementById('experiments-tab').classList.contains('active')) refreshExperiments();
        if (document.getElementById('rae-tab').classList.contains('active')) refreshRae();
        if (document.getElementById('analyzer-tab').classList.contains('active')) refreshAnalyzer();
        updateStats();
        fetch('/api/massive_trainer_status').then(r=>r.json()).then(s => {
            document.getElementById('stopMassiveBtn').style.display = s.active ? 'inline-block' : 'none';
        });
    }, 10000);
});
    </script>
</body>
</html>
'''

# -------------------- API Routes (только чтение и запуск внешнего скрипта) --------------------
@app.route('/')
def index():
    return render_template_string(INDEX_TEMPLATE, exercises=get_exercise_list(), config=config)

@app.route('/api/experiments')
def api_experiments():
    exps = manager.list_experiments()
    clean = []
    for exp in exps:
        if not exp.get('experiment_id'): continue
        clean_exp = {k: v if v != '' else None for k, v in exp.items()}
        clean.append(clean_exp)
    return jsonify(clean)

@app.route('/api/models')
def api_models():
    mtype = request.args.get('type', 'rae')
    return jsonify(get_available_models(mtype))

@app.route('/api/stop_experiment', methods=['POST'])
def api_stop():
    exp_id = request.get_json().get('experiment_id')
    # Остановка через сигнал (упрощённо)
    return jsonify({'status': 'ok'})

@app.route('/api/delete_experiments', methods=['POST'])
def api_delete():
    ids = request.get_json().get('ids', [])
    exps = [e for e in load_experiments() if e['experiment_id'] not in ids]
    # save_experiments(exps)  # если определена
    return jsonify({'status': 'ok'})

@app.route('/plot')
def serve_plot():
    path = request.args.get('path')
    if not path or not os.path.exists(path):
        return "Not found", 404
    return send_file(path, mimetype='image/png')

@app.route('/api/compare_models', methods=['POST'])
def compare_models():
    paths = request.get_json().get('log_paths', [])
    results = []
    for p in paths:
        test = parse_training_log(p)
        if test:
            results.append({'epochs': list(range(1, len(test)+1)), 'test_losses': test})
    return jsonify(results)

@app.route('/api/export_csv')
def export_csv():
    return send_file(CSV_LOG, as_attachment=True) if os.path.exists(CSV_LOG) else ("No data", 404)

@app.route('/api/massive_stats')
def massive_stats():
    exps = load_experiments()
    tasks = [e for e in exps if e.get('type') in ('rae', 'analyzer')]
    total = len(tasks)
    completed = len([e for e in tasks if e.get('status') == 'completed'])
    running = len([e for e in tasks if e.get('status') == 'running'])
    failed = len([e for e in tasks if e.get('status') == 'failed'])
    rae_losses = [float(e['best_test_loss']) for e in tasks if e.get('type')=='rae' and e.get('best_test_loss')]
    analyzer_accs = [float(e.get('accuracy',0)) for e in tasks if e.get('type')=='analyzer' and e.get('accuracy')]
    return jsonify({
        'total': total, 'completed': completed, 'running': running, 'failed': failed,
        'best_rae_loss': min(rae_losses) if rae_losses else None,
        'best_analyzer_acc': max(analyzer_accs) if analyzer_accs else None
    })

@app.route('/api/start_massive_trainer', methods=['POST'])
def start_massive_trainer():
    data = request.get_json()
    model_type = data.get('model_type', 'rae')
    strategy = data.get('strategy', 'grid')
    workers = data.get('workers', 2)
    # Запускаем massive_trainer.py в фоне
    cmd = [sys.executable, 'massive_trainer.py', '--model_type', model_type,
           '--strategy', strategy, '--workers', str(workers)]
    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # Сохраняем PID для отслеживания
    with open(TRAINER_PID_FILE, 'w') as f:
        f.write(str(os.getpid()))  # упрощённо
    return jsonify({'status': 'ok'})

@app.route('/api/stop_massive_trainer', methods=['POST'])
def stop_massive_trainer():
    if os.path.exists(TRAINER_PID_FILE):
        os.remove(TRAINER_PID_FILE)
    return jsonify({'status': 'ok'})

@app.route('/api/massive_trainer_status')
def massive_trainer_status():
    active = os.path.exists(TRAINER_PID_FILE)
    return jsonify({'active': active})

if __name__ == '__main__':
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(ANALYZERS_DIR, exist_ok=True)
    app.run(host='0.0.0.0', port=3245, debug=True)