/**
 * Google Sheets automation script for the multi-spreadsheet weekly plan system.
 *
 * How to use
 * 1. Paste this script into a new standalone Apps Script project (or keep in a repo-managed clasp project).
 * 2. Fill the CONFIG placeholders (shared drive ID, teachers, etc.).
 * 3. Run runBootstrap() once. It creates all spreadsheets, templates, and idempotent state. If the run times out,
 *    simply run runBootstrap() again — it resumes from the saved state without duplicating files.
 * 4. Grant the Drive/Sheets scopes when prompted. The script installs onEdit triggers on every spreadsheet so that
 *    cross-sheet sync keeps working after setup.
 */

// --------- 固定ルール（最上段） ---------
const SUBJECT_CODES = [
  '国', '算', '社', '理', '生活', '音', '図工', '図書', '未来', '家', '道', '外国', '体', 'ク委', '特活', '総合', '算専', '×'
];

// 専科のプルダウンは学年組のみ。必要に応じて増減する。
const CLASSROOMS = Array.from({ length: 6 }, (_, grade) =>
  Array.from({ length: 3 }, (_, idx) => `${grade + 1}-${idx + 1}`)
).flat();

// 専科ファイルの定義（担当学年はフィルタリングに使用する）。
const SPECIAL_SUBJECTS = [
  { code: '音', name: '音楽', grades: ['1', '2', '3', '4', '5', '6'] },
  { code: '家', name: '家庭', grades: ['5', '6'] },
  { code: '外国', name: '外国語', grades: ['5', '6'] },
  { code: '算専', name: '算数専科', grades: ['5', '6'] },
  { code: '体', name: '体育', grades: ['1', '2', '3', '4', '5', '6'] },
  { code: '理', name: '理科', grades: ['3', '4', '5', '6'] }
];

// 支援の先生。児童設定は支援ファイル側のチェックボックスで管理するので、ここでは教員名のみ。
const SUPPORT_TEACHERS = [
  // 例: { name: '支援A' }
];

// --------- 環境設定（必ず書き換える） ---------
const CONFIG = {
  sharedDriveId: 'PUT_SHARED_DRIVE_ID', // 共有ドライブのID
  academicYear: 2025, // 対象年度（西暦）
  weekStartMonth: 4, // 学年開始月（4=4月）
  weekStartDay: 1, // 学年開始日のデフォルト（4/1）
  maxWeeks: 55 // タイムアウト時の再実行を考慮し、上限週数を固定
};

// --------- テンプレ座標（写真テンプレ準拠） ---------
const GRID = {
  dayColumns: ['B', 'C', 'D', 'E', 'F', 'G', 'H'], // 月〜日
  weekdayNames: ['月', '火', '水', '木', '金', '土', '日'],
  dateRow: 2, // 日付行
  eventRow: 3, // 予定行
  firstPeriodRow: 5, // 1限の教科セル行
  periods: 6,
  rowsPerPeriod: 2 // 教科セル + 備考セル
};

// --------- ステート管理 ---------
const STATE_KEYS = {
  MASTER: 'MASTER_FILE',
  CLASS: 'CLASS_',
  SPECIAL: 'SPECIAL_',
  SUPPORT: 'SUPPORT_',
  TRIGGERS: 'TRIGGERS_INSTALLED'
};

function runBootstrap() {
  const state = loadState();
  const master = ensureMasterSpreadsheet(state);
  state[STATE_KEYS.MASTER] = master.getId();

  // 1. クラス別ファイル
  CLASSROOMS.forEach(room => {
    const key = `${STATE_KEYS.CLASS}${room}`;
    if (!state[key]) {
      const ss = createClassSpreadsheet(room);
      state[key] = ss.getId();
    }
    ensureWeeklyTabs(SpreadsheetApp.openById(state[key]));
  });

  // 2. 専科別ファイル
  SPECIAL_SUBJECTS.forEach(subj => {
    const key = `${STATE_KEYS.SPECIAL}${subj.code}`;
    if (!state[key]) {
      const ss = createSpecialSpreadsheet(subj);
      state[key] = ss.getId();
    }
    ensureWeeklyTabs(SpreadsheetApp.openById(state[key]));
  });

  // 3. 支援別ファイル
  SUPPORT_TEACHERS.forEach(teacher => {
    const key = `${STATE_KEYS.SUPPORT}${teacher.name}`;
    if (!state[key]) {
      const ss = createSupportSpreadsheet(teacher);
      state[key] = ss.getId();
    }
    ensureWeeklyTabs(SpreadsheetApp.openById(state[key]));
  });

  // 4. 行事予定・設定・統合表示
  ensureMasterSheets(master);

  // 5. 行事予定を各週案に即時反映
  syncEventsToAll(state);

  // 6. トリガーを冪等に設定
  if (!state[STATE_KEYS.TRIGGERS]) {
    installTriggers(state);
    state[STATE_KEYS.TRIGGERS] = 'YES';
  }

  saveState(state);
}

// --- ステート読書き ---
function loadState() {
  const raw = PropertiesService.getScriptProperties().getProperty('STATE');
  return raw ? JSON.parse(raw) : {};
}

function saveState(state) {
  PropertiesService.getScriptProperties().setProperty('STATE', JSON.stringify(state));
}

// --- マスタ（統合） ---
function ensureMasterSpreadsheet(state) {
  if (state[STATE_KEYS.MASTER]) {
    return SpreadsheetApp.openById(state[STATE_KEYS.MASTER]);
  }
  const ss = SpreadsheetApp.create('統合');
  moveToSharedDrive(ss.getId());
  return ss;
}

function ensureMasterSheets(master) {
  const settings = getOrCreateSheet(master, '設定');
  const events = getOrCreateSheet(master, '行事予定');
  const view = getOrCreateSheet(master, '統合グリッド');

  // 設定シートのヘッダー
  if (settings.getLastRow() === 0) {
    settings.appendRow(['キー', '値', '備考']);
    settings.appendRow(['対象年度', CONFIG.academicYear, '西暦']);
    settings.appendRow(['週開始月', CONFIG.weekStartMonth, '整数']);
    settings.appendRow(['週開始日', CONFIG.weekStartDay, '整数']);
  }

  // 行事予定ヘッダー
  if (events.getLastRow() === 0) {
    events.appendRow(['日付', '曜日', '内容']);
  }

  // 統合グリッドの骨組み
  setupIntegrationGrid(view);
}

function setupIntegrationGrid(sheet) {
  sheet.clear();
  const header = ['限/曜日', ...GRID.weekdayNames];
  sheet.getRange(1, 1, 1, header.length).setValues([header]).setFontWeight('bold');
  for (let p = 0; p < GRID.periods; p++) {
    sheet.getRange(p + 2, 1).setValue(`${p + 1}限`).setFontWeight('bold');
  }
  sheet.setFrozenRows(1);
  sheet.setFrozenColumns(1);
}

// --- ファイル作成 ---
function createClassSpreadsheet(room) {
  const ss = SpreadsheetApp.create(`週案_${room}`);
  moveToSharedDrive(ss.getId());
  const sheet = ss.getSheets()[0];
  sheet.setName('テンプレ');
  applyTemplate(sheet, { type: 'class' });
  setClassValidation(sheet);
  return ss;
}

function createSpecialSpreadsheet(subj) {
  const ss = SpreadsheetApp.create(`週案_${subj.name}`);
  moveToSharedDrive(ss.getId());
  const sheet = ss.getSheets()[0];
  sheet.setName('テンプレ');
  applyTemplate(sheet, { type: 'special' });
  setSpecialValidation(sheet);
  addSpecialMetadata(sheet, subj);
  return ss;
}

function createSupportSpreadsheet(teacher) {
  const ss = SpreadsheetApp.create(`週案_支援_${teacher.name}`);
  moveToSharedDrive(ss.getId());
  const sheet = ss.getSheets()[0];
  sheet.setName('テンプレ');
  applyTemplate(sheet, { type: 'support' });
  setSupportLayout(sheet);
  return ss;
}

function moveToSharedDrive(fileId) {
  if (!CONFIG.sharedDriveId) return;
  Drive.Files.update({ driveId: CONFIG.sharedDriveId }, fileId, null, { supportsAllDrives: true });
}

// --- テンプレ作成 ---
function applyTemplate(sheet, opts) {
  sheet.clear();
  sheet.getRange(GRID.dateRow - 1, 2, 1, GRID.dayColumns.length).setValues([GRID.weekdayNames]).setFontWeight('bold');

  // 日付行と予定行の罫線
  sheet.getRange(GRID.dateRow, 2, 2, GRID.dayColumns.length).setHorizontalAlignment('center').setBackground('#f7f7f7');

  for (let p = 0; p < GRID.periods; p++) {
    const subjectRow = GRID.firstPeriodRow + p * GRID.rowsPerPeriod;
    const noteRow = subjectRow + 1;
    sheet.getRange(subjectRow, 1).setValue(`${p + 1}限`).setFontWeight('bold');
    sheet.getRange(noteRow, 1).setValue('備考').setFontColor('#666');
    sheet.getRange(subjectRow, 2, 1, GRID.dayColumns.length).setBackground('#ffffff');
    sheet.getRange(noteRow, 2, 1, GRID.dayColumns.length).setBackground('#f1f5f9');
  }

  sheet.setFrozenRows(GRID.firstPeriodRow - 1);
  sheet.setFrozenColumns(1);

  if (opts.type === 'support') {
    sheet.getRange('J1').setValue('児童設定（チェックボックス）').setFontWeight('bold');
  }
}

function setClassValidation(sheet) {
  const rule = SpreadsheetApp.newDataValidation().requireValueInList(SUBJECT_CODES, true).build();
  for (let p = 0; p < GRID.periods; p++) {
    const subjectRow = GRID.firstPeriodRow + p * GRID.rowsPerPeriod;
    sheet.getRange(subjectRow, 2, 1, GRID.dayColumns.length).setDataValidation(rule);
  }
}

function setSpecialValidation(sheet) {
  const rule = SpreadsheetApp.newDataValidation().requireValueInList(CLASSROOMS, true).build();
  for (let p = 0; p < GRID.periods; p++) {
    const subjectRow = GRID.firstPeriodRow + p * GRID.rowsPerPeriod;
    sheet.getRange(subjectRow, 2, 1, GRID.dayColumns.length).setDataValidation(rule);
  }
}

function addSpecialMetadata(sheet, subj) {
  sheet.getRange('J1').setValue('専科設定').setFontWeight('bold');
  sheet.getRange('J2').setValue('教科コード');
  sheet.getRange('K2').setValue(subj.code);
  sheet.getRange('J3').setValue('担当学年');
  sheet.getRange('K3').setValue(subj.grades.join(','));
}

function setSupportLayout(sheet) {
  sheet.getRange('J2').setValue('児童名');
  sheet.getRange('K2').setValue('学年組');
  sheet.getRange('L2').setValue('必要教科');
  sheet.getRange('M2').setValue('ON/OFF');
}

// --- 週タブ生成 ---
function ensureWeeklyTabs(ss) {
  const weeks = generateWeeks();
  weeks.forEach(week => {
    const name = `${week.label}週`;
    let sheet = ss.getSheetByName(name);
    if (!sheet) {
      sheet = ss.insertSheet(name);
      applyTemplate(sheet, { type: 'class' });
      setClassValidation(sheet);
    }
    fillWeekDates(sheet, week);
  });
}

function generateWeeks() {
  const firstDate = new Date(CONFIG.academicYear, CONFIG.weekStartMonth - 1, CONFIG.weekStartDay);
  const monday = shiftToMonday(firstDate);
  const weeks = [];
  for (let i = 0; i < CONFIG.maxWeeks; i++) {
    const start = new Date(monday);
    start.setDate(start.getDate() + i * 7);
    if (start.getFullYear() > CONFIG.academicYear + 1) break;
    const label = formatMonthDay(start);
    weeks.push({ start, label });
  }
  return weeks;
}

function shiftToMonday(date) {
  const d = new Date(date);
  const diff = (d.getDay() + 6) % 7; // Monday=0
  d.setDate(d.getDate() - diff);
  return d;
}

function formatMonthDay(date) {
  const m = ('0' + (date.getMonth() + 1)).slice(-2);
  const d = ('0' + date.getDate()).slice(-2);
  return `${m}/${d}`;
}

function fillWeekDates(sheet, week) {
  const dates = [];
  for (let i = 0; i < 7; i++) {
    const day = new Date(week.start);
    day.setDate(day.getDate() + i);
    dates.push(day);
  }
  const range = sheet.getRange(GRID.dateRow, 2, 1, dates.length);
  range.setValues([dates]).setNumberFormat('M/d');
  const weekdayRow = sheet.getRange(GRID.dateRow - 1, 2, 1, dates.length);
  weekdayRow.setValues([GRID.weekdayNames]);
}

// --- 行事予定の反映 ---
function syncEventsToAll(state) {
  const master = SpreadsheetApp.openById(state[STATE_KEYS.MASTER]);
  const eventsSheet = master.getSheetByName('行事予定');
  if (!eventsSheet) return;
  const events = eventsSheet.getDataRange().getValues().slice(1).filter(r => r[0]);

  const applyEvents = ssId => {
    const ss = SpreadsheetApp.openById(ssId);
    ss.getSheets().forEach(sheet => {
      const name = sheet.getName();
      if (!name.includes('週')) return;
      const dateRow = sheet.getRange(GRID.dateRow, 2, 1, GRID.dayColumns.length).getValues()[0];
      const textRow = GRID.weekdayNames.map((_, idx) => {
        const day = dateRow[idx];
        const matches = events.filter(r => sameDay(day, r[0])).map(r => r[2]);
        return Array.from(new Set(matches)).join('\n');
      });
      sheet.getRange(GRID.eventRow, 2, 1, GRID.dayColumns.length).setValues([textRow]);
    });
  };

  Object.keys(state).forEach(key => {
    if (key.startsWith(STATE_KEYS.CLASS) || key.startsWith(STATE_KEYS.SPECIAL) || key.startsWith(STATE_KEYS.SUPPORT)) {
      applyEvents(state[key]);
    }
  });
}

function sameDay(a, b) {
  if (!a || !b) return false;
  const ad = new Date(a); const bd = new Date(b);
  return ad.getFullYear() === bd.getFullYear() && ad.getMonth() === bd.getMonth() && ad.getDate() === bd.getDate();
}

// --- 送受信・衝突処理 ---
function installTriggers(state) {
  ScriptApp.getProjectTriggers().forEach(t => ScriptApp.deleteTrigger(t));
  Object.keys(state).forEach(key => {
    if (key.startsWith(STATE_KEYS.CLASS) || key.startsWith(STATE_KEYS.SPECIAL)) {
      ScriptApp.newTrigger('handleEdit').forSpreadsheet(state[key]).onEdit().create();
    }
  });
}

function handleEdit(e) {
  const ssId = e.source.getId();
  const state = loadState();
  const type = detectType(ssId, state);
  if (!type) return;
  if (!isGridCell(e.range)) return;

  if (type.kind === 'class') {
    processClassEdit(e, state, type);
  } else if (type.kind === 'special') {
    processSpecialEdit(e, state, type);
  }
}

function detectType(id, state) {
  const entry = Object.entries(state).find(([k, v]) => v === id);
  if (!entry) return null;
  const [key] = entry;
  if (key.startsWith(STATE_KEYS.CLASS)) return { kind: 'class', room: key.replace(STATE_KEYS.CLASS, '') };
  if (key.startsWith(STATE_KEYS.SPECIAL)) return { kind: 'special', code: key.replace(STATE_KEYS.SPECIAL, '') };
  return null;
}

function isGridCell(range) {
  const row = range.getRow();
  const col = range.getColumn();
  const maxRow = GRID.firstPeriodRow + GRID.periods * GRID.rowsPerPeriod;
  const minCol = 2;
  const maxCol = minCol + GRID.dayColumns.length - 1;
  return row >= GRID.firstPeriodRow && row <= maxRow && col >= minCol && col <= maxCol && range.getNumRows() === 1 && range.getNumColumns() === 1;
}

function resolveSlot(range) {
  const dayIndex = range.getColumn() - 2; // 0-6
  const periodIndex = Math.floor((range.getRow() - GRID.firstPeriodRow) / GRID.rowsPerPeriod);
  const isSubjectRow = (range.getRow() - GRID.firstPeriodRow) % GRID.rowsPerPeriod === 0;
  return { dayIndex, periodIndex, isSubjectRow };
}

function processClassEdit(e, state, type) {
  const ss = e.source;
  const slot = resolveSlot(e.range);
  if (!slot.isSubjectRow) return;
  const value = (e.value || '').trim();
  const oldValue = (e.oldValue || '').trim();
  const isSpecialRequest = SPECIAL_SUBJECTS.some(s => s.code === value);
  const oldWasSpecial = SPECIAL_SUBJECTS.some(s => s.code === oldValue);

  // 担任→専科リクエスト
  if (isSpecialRequest) {
    const target = SPECIAL_SUBJECTS.find(s => s.code === value);
    const specialId = state[`${STATE_KEYS.SPECIAL}${target.code}`];
    if (!specialId) return;
    const accepted = pushRequestToSpecial(ss, specialId, type.room, slot);
    if (!accepted) {
      paintWarning(e.range, '専科の枠が埋まっています');
    } else {
      clearWarning(e.range);
    }
    return;
  }

  // 担任が別教科に直した場合、未確定（赤文字）だけ消す
  if (!isSpecialRequest && oldWasSpecial) {
    SPECIAL_SUBJECTS.forEach(subj => {
      const specialId = state[`${STATE_KEYS.SPECIAL}${subj.code}`];
      if (!specialId) return;
      removePendingRequest(specialId, type.room, slot);
    });
  }
}

function processSpecialEdit(e, state, type) {
  const slot = resolveSlot(e.range);
  if (!slot.isSubjectRow) return;
  const room = (e.value || '').trim();
  const oldRoom = (e.oldValue || '').trim();
  const canServe = CLASSROOMS.includes(room) && isGradeAllowed(type.code, room);
  const ss = e.source;

  if (room && !canServe) {
    e.range.setComment('担当学年外のため送信しません');
    return;
  }

  if (room) {
    const accepted = pushSpecialToClass(ss, state, type.code, room, slot);
    if (!accepted) {
      e.range.setFontColor('red');
      e.range.setComment('担任側に別専科が入っているため送信を停止');
    } else {
      e.range.setFontColor('black');
      e.range.setComment('');
    }
  }

  // 削除時は担任の青戻し
  if (!room && oldRoom) {
    const targetId = state[`${STATE_KEYS.CLASS}${oldRoom}`];
    if (!targetId) return;
    const classSheet = SpreadsheetApp.openById(targetId).getSheetByName(e.range.getSheet().getName());
    const classRange = mapToClassRange(classSheet, slot);
    classRange.clearContent();
    classRange.setBackground('#c7d2fe'); // 青
  }
}

function pushRequestToSpecial(classSS, specialId, room, slot) {
  const special = SpreadsheetApp.openById(specialId);
  const sheet = special.getSheetByName(classSS.getActiveSheet().getName()) || special.getSheetByName('テンプレ');
  const target = mapToSpecialRange(sheet, slot);
  if (target.getValue()) return false; // 埋まっているので弾く
  target.setValue(room);
  target.setFontColor('red');
  target.setComment('担任リクエスト（未確定）');
  return true;
}

function removePendingRequest(specialId, room, slot) {
  const ss = SpreadsheetApp.openById(specialId);
  const sheet = ss.getSheets().find(s => s.getName().includes('週')) || ss.getSheets()[0];
  const target = mapToSpecialRange(sheet, slot);
  if (target.getFontColor() === 'red' && target.getValue() === room) {
    target.clearContent().setComment('');
  }
}

function pushSpecialToClass(specialSS, state, code, room, slot) {
  const classId = state[`${STATE_KEYS.CLASS}${room}`];
  if (!classId) return false;
  const classSS = SpreadsheetApp.openById(classId);
  const sheet = classSS.getSheetByName(specialSS.getActiveSheet().getName()) || classSS.getSheetByName('テンプレ');
  const target = mapToClassRange(sheet, slot);
  const current = (target.getValue() || '').trim();
  const isOtherSpecial = SPECIAL_SUBJECTS.some(s => s.code === current && s.code !== code);
  if (isOtherSpecial) return false;

  target.setValue(code);
  target.setBackground('#fecdd3'); // 赤背景：専科送信で上書き
  const note = sheet.getRange(target.getRow() + 1, target.getColumn());
  const srcNote = specialSS.getActiveSheet().getRange(target.getRow() + 1, target.getColumn());
  note.setValue(srcNote.getValue());
  return true;
}

function mapToClassRange(sheet, slot) {
  const row = GRID.firstPeriodRow + slot.periodIndex * GRID.rowsPerPeriod;
  const col = 2 + slot.dayIndex;
  return sheet.getRange(row, col);
}

function mapToSpecialRange(sheet, slot) {
  const row = GRID.firstPeriodRow + slot.periodIndex * GRID.rowsPerPeriod;
  const col = 2 + slot.dayIndex;
  return sheet.getRange(row, col);
}

function paintWarning(range, note) {
  range.setBackground('#f97316');
  range.setComment(note || '');
}

function clearWarning(range) {
  range.setBackground('#ffffff');
  range.setComment('');
}

function isGradeAllowed(code, room) {
  const subj = SPECIAL_SUBJECTS.find(s => s.code === code);
  if (!subj) return false;
  const grade = room.split('-')[0];
  return subj.grades.includes(grade);
}

// --- 統合空き表示（管理者用） ---
function refreshAvailability() {
  const state = loadState();
  const master = SpreadsheetApp.openById(state[STATE_KEYS.MASTER]);
  const gridSheet = master.getSheetByName('統合グリッド');
  const sampleClassId = state[`${STATE_KEYS.CLASS}${CLASSROOMS[0]}`];
  if (!sampleClassId) return;
  const weekName = SpreadsheetApp.openById(sampleClassId).getSheets().find(s => s.getName().includes('週')).getName();
  const snapshots = collectSnapshots(state, weekName);
  renderAvailability(gridSheet, snapshots);
}

function collectSnapshots(state, sheetName) {
  const readSheet = id => {
    const ss = SpreadsheetApp.openById(id);
    const sheet = ss.getSheetByName(sheetName);
    const subjects = sheet.getRange(GRID.firstPeriodRow, 2, GRID.periods * GRID.rowsPerPeriod, GRID.dayColumns.length).getValues();
    return subjects;
  };

  const classData = Object.entries(state)
    .filter(([k]) => k.startsWith(STATE_KEYS.CLASS))
    .map(([k, id]) => ({ room: k.replace(STATE_KEYS.CLASS, ''), data: readSheet(id) }));
  const specialData = Object.entries(state)
    .filter(([k]) => k.startsWith(STATE_KEYS.SPECIAL))
    .map(([k, id]) => ({ code: k.replace(STATE_KEYS.SPECIAL, ''), data: readSheet(id) }));
  return { classData, specialData };
}

function renderAvailability(sheet, snapshots) {
  const grid = Array.from({ length: GRID.periods }, () => Array(GRID.dayColumns.length).fill(''));
  for (let p = 0; p < GRID.periods; p++) {
    for (let d = 0; d < GRID.dayColumns.length; d++) {
      const classSlots = snapshots.classData.filter(c => !snapshots.specialData.some(s => s.data[p * GRID.rowsPerPeriod][d] === c.room))
        .filter(c => {
          const val = c.data[p * GRID.rowsPerPeriod][d];
          return !val || SPECIAL_SUBJECTS.some(s => s.code === val);
        })
        .map(c => c.room);
      const specialSlots = snapshots.specialData.filter(s => !s.data[p * GRID.rowsPerPeriod][d]).map(s => s.code);
      const line = [...specialSlots, ...classSlots];
      grid[p][d] = line.join('/');
    }
  }
  sheet.getRange(2, 2, GRID.periods, GRID.dayColumns.length).setValues(grid);
}

// --- 支援読み込み ---
function pullSupportPlans() {
  const state = loadState();
  Object.keys(state)
    .filter(k => k.startsWith(STATE_KEYS.SUPPORT))
    .forEach(key => {
      const ss = SpreadsheetApp.openById(state[key]);
      ss.getSheets().forEach(sheet => {
        if (!sheet.getName().includes('週')) return;
        populateSupportSheet(sheet, state);
      });
    });
}

function populateSupportSheet(sheet, state) {
  const settingsRange = sheet.getRange('J3:M');
  const rows = settingsRange.getValues().filter(r => r[0]);
  const lookups = rows.map(r => ({ name: r[0], room: r[1], subjects: (r[2] || '').split(','), active: r[3] === true }));
  const grid = Array.from({ length: GRID.periods }, () => Array(GRID.dayColumns.length).fill(''));

  lookups.filter(r => r.active).forEach(item => {
    const classId = state[`${STATE_KEYS.CLASS}${item.room}`];
    if (!classId) return;
    const classSheet = SpreadsheetApp.openById(classId).getSheetByName(sheet.getName());
    const data = classSheet.getRange(GRID.firstPeriodRow, 2, GRID.periods * GRID.rowsPerPeriod, GRID.dayColumns.length).getValues();
    for (let p = 0; p < GRID.periods; p++) {
      for (let d = 0; d < GRID.dayColumns.length; d++) {
        const subject = data[p * GRID.rowsPerPeriod][d];
        const note = data[p * GRID.rowsPerPeriod + 1][d];
        if (item.subjects.includes(subject)) {
          const entry = note ? `${subject},${note}(${item.name})` : `${subject}(${item.name})`;
          grid[p][d] = grid[p][d] ? `${grid[p][d]}／${entry}` : entry;
        }
      }
    }
  });

  sheet.getRange(GRID.firstPeriodRow, 2, GRID.periods, GRID.dayColumns.length).setValues(grid);
}

