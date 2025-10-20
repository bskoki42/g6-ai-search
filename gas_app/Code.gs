/**
 * Google Apps Script backend for the reflection workflow web application.
 * The implementation favours explicit comments and defensive checks to help
 * teachers customise the script after deployment. The functions are organised
 * into logical sections: configuration, utilities, sheet helpers, web API,
 * approval workflow, and bootstrap/testing helpers.
 */

const SCRIPT_PROPERTIES = PropertiesService.getScriptProperties();
const SHEET_ID = SCRIPT_PROPERTIES.getProperty('SHEET_ID');
const GEMINI_API_KEY = SCRIPT_PROPERTIES.getProperty('GEMINI_API_KEY');
const APP_ACCESS_KEY = SCRIPT_PROPERTIES.getProperty('APP_ACCESS_KEY');

// Sheet names and default headers (ordered to match the specification).
const SHEETS = {
  reflections: 'Reflections',
  approvals: 'Approvals',
  roster: 'Roster',
  logs: 'Logs'
};

const REFLECTION_HEADERS = [
  'timestamp', // ISO string in UTC
  'studentEmail',
  'studentName',
  'subject',
  'reflection',
  'status',
  'eval',
  'teacherComment',
  'key'
];

const APPROVAL_HEADERS = [
  'timestamp',
  'studentEmail',
  'studentName',
  'subject',
  'reflection',
  'evaluation',
  'teacherComment',
  'approved',
  'statusLog',
  'rawResponse'
];

const SUBJECT_OPTIONS = ['国', '算', '社', '理', '音', '図工', '図書', '未来', '家', '道', '外', '体', 'ク委', '特活'];
const MAX_REFLECTION_LENGTH = 2000;
const MIN_REFLECTION_LENGTH = 1;

/**
 * Adds custom menu items when the spreadsheet is opened.
 */
function onOpen() {
  const ui = SpreadsheetApp.getUi();
  ui.createMenu('📤 承認ワークフロー')
    .addItem('① 抽出（教科・日付）', 'menuExtractForApproval')
    .addItem('② AI評価（Gemini）', 'menuGenerateAIForApprovals')
    .addItem('③ 承認反映', 'menuApplyApprovals')
    .addToUi();
}

/**
 * Generic handler for GET requests. Routes HTML views and JSON API endpoints
 * based on the path (e.pathInfo).
 */
function doGet(e) {
  ensureSheets();
  const path = (e && e.pathInfo) ? e.pathInfo.replace(/^\/+|\/+$/g, '') : '';
  if (!path || path === '') {
    return renderTemplate('views/index', { title: '振り返りポータル', appKey: APP_ACCESS_KEY || '' });
  }
  if (path === 'teacher') {
    return renderTemplate('views/teacher', { title: '教員ビュー', appKey: APP_ACCESS_KEY || '' });
  }
  if (path.startsWith('api/')) {
    return routeApiGet(path.substring(4), e);
  }
  return ContentService.createTextOutput('Not Found').setMimeType(ContentService.MimeType.TEXT);
}

/**
 * Handler for POST requests. Supports JSON APIs under /api/.
 */
function doPost(e) {
  ensureSheets();
  const path = (e && e.pathInfo) ? e.pathInfo.replace(/^\/+|\/+$/g, '') : '';
  if (path.startsWith('api/')) {
    return routeApiPost(path.substring(4), e);
  }
  return ContentService.createTextOutput('Not Found').setMimeType(ContentService.MimeType.TEXT);
}

/**
 * Ensures that core sheets and headers exist. Idempotent.
 */
function ensureSheets() {
  if (!SHEET_ID) {
    throw new Error('SHEET_ID is not set in Script Properties.');
  }
  const ss = SpreadsheetApp.openById(SHEET_ID);
  ensureSheetWithHeaders(ss, SHEETS.reflections, REFLECTION_HEADERS);
  ensureSheetWithHeaders(ss, SHEETS.approvals, APPROVAL_HEADERS);
  ensureSheetWithHeaders(ss, SHEETS.roster, ['email', 'displayName', 'kanaName', 'class']);
  ensureSheetWithHeaders(ss, SHEETS.logs, ['timestamp', 'level', 'event', 'detail']);
}

function ensureSheetWithHeaders(ss, sheetName, headers) {
  let sheet = ss.getSheetByName(sheetName);
  if (!sheet) {
    sheet = ss.insertSheet(sheetName);
  }
  const firstRow = sheet.getRange(1, 1, 1, headers.length).getValues()[0];
  const needsHeaders = headers.some((header, idx) => firstRow[idx] !== header);
  if (needsHeaders) {
    sheet.getRange(1, 1, 1, headers.length).setValues([headers]);
  }
  sheet.setFrozenRows(1);
}

/**
 * Renders an HTML template file.
 */
function renderTemplate(path, data) {
  const template = HtmlService.createTemplateFromFile(path);
  Object.keys(data || {}).forEach(key => template[key] = data[key]);
  return template.evaluate().setTitle(data && data.title ? data.title : '振り返り');
}

/**
 * Routes API GET requests.
 */
function routeApiGet(path, e) {
  switch (path) {
    case 'me':
      return jsonResponse(getCurrentUserProfile());
    case 'previous':
      return jsonResponse(handleGetPrevious(e));
    case 'approvals':
      return jsonResponse(listApprovalQueue(e));
    default:
      return notFound();
  }
}

/**
 * Routes API POST requests.
 */
function routeApiPost(path, e) {
  validateCsrf(e);
  switch (path) {
    case 'submit':
      return jsonResponse(handleSubmit(e));
    case 'approve':
      return jsonResponse(handleApproveApi(e));
    case 'bootstrap':
      return jsonResponse(bootstrapDummyData());
    default:
      return notFound();
  }
}

function notFound() {
  return ContentService.createTextOutput(JSON.stringify({ error: 'not_found' }))
    .setMimeType(ContentService.MimeType.JSON)
    .setResponseCode(404);
}

function jsonResponse(payload) {
  return ContentService.createTextOutput(JSON.stringify(payload))
    .setMimeType(ContentService.MimeType.JSON);
}

/**
 * Validates the custom header for POST endpoints to mitigate CSRF risks.
 */
function validateCsrf(e) {
  if (!APP_ACCESS_KEY) {
    // When no key is configured we allow the request but log a warning. The
    // default is still safe because the web app requires Google login.
    logEvent('WARN', 'csrf_skip', 'APP_ACCESS_KEY is not configured.');
    return;
  }
  const headers = (e && e.headers) || {};
  const key = headers['x-app-key'] || headers['X-App-Key'];
  if (key !== APP_ACCESS_KEY) {
    throw new Error('Invalid X-App-Key header.');
  }
}

/**
 * Returns the active user's email and display name (katakana when available).
 */
function getCurrentUserProfile() {
  const email = getUserEmail();
  const ss = SpreadsheetApp.openById(SHEET_ID);
  const rosterSheet = ss.getSheetByName(SHEETS.roster);
  const rosterMap = buildRosterMap(rosterSheet);
  const rosterEntry = rosterMap[email] || {};
  const displayName = rosterEntry.kanaName || rosterEntry.displayName || ''; // fallback order
  return {
    email: email,
    displayName: displayName || getProfileFromGIS(),
    domain: email ? email.split('@')[1] : '',
    timestamp: new Date().toISOString()
  };
}

/**
 * Retrieves the active user email. Falls back to GIS verification if provided
 * in the request parameter `idToken`.
 */
function getUserEmail(optionalToken) {
  const sessionEmail = Session.getActiveUser().getEmail();
  if (sessionEmail) {
    return sessionEmail;
  }
  if (optionalToken) {
    const tokenInfo = verifyGoogleIdToken(optionalToken);
    if (tokenInfo && tokenInfo.email) {
      return tokenInfo.email;
    }
  }
  throw new Error('Unable to determine user email. Ensure domain-restricted deployment or provide GIS ID token.');
}

function getProfileFromGIS() {
  // Placeholder for GIS profile retrieval if you extend the front-end to send
  // the display name explicitly. Currently returns an empty string and relies
  // on the roster fallback.
  return '';
}

function verifyGoogleIdToken(idToken) {
  if (!idToken) {
    return null;
  }
  const endpoint = 'https://oauth2.googleapis.com/tokeninfo?id_token=' + idToken;
  const response = UrlFetchApp.fetch(endpoint, { method: 'get', muteHttpExceptions: true });
  if (response.getResponseCode() !== 200) {
    throw new Error('Failed to verify ID token: ' + response.getContentText());
  }
  return JSON.parse(response.getContentText());
}

/**
 * Fetches the previous reflection for the logged-in user and subject.
 */
function handleGetPrevious(e) {
  const subject = (e.parameter && e.parameter.subject) ? e.parameter.subject : '';
  const email = getUserEmail(e.parameter && e.parameter.idToken);
  if (!subject || SUBJECT_OPTIONS.indexOf(subject) === -1) {
    throw new Error('Invalid subject parameter.');
  }
  const ss = SpreadsheetApp.openById(SHEET_ID);
  const sheet = ss.getSheetByName(SHEETS.reflections);
  const data = sheet.getDataRange().getValues();
  const headers = data.shift();
  const rows = data
    .filter(row => row[headers.indexOf('studentEmail')] === email && row[headers.indexOf('subject')] === subject)
    .sort((a, b) => new Date(b[0]).getTime() - new Date(a[0]).getTime());
  if (!rows.length) {
    return { hasPrevious: false };
  }
  const latest = rows[0];
  return {
    hasPrevious: true,
    timestamp: latest[headers.indexOf('timestamp')],
    status: latest[headers.indexOf('status')],
    reflection: latest[headers.indexOf('reflection')],
    evaluation: latest[headers.indexOf('eval')],
    teacherComment: latest[headers.indexOf('teacherComment')]
  };
}

/**
 * Handles reflection submission from students.
 */
function handleSubmit(e) {
  const body = parseJsonBody(e);
  const subject = body.subject;
  const reflection = body.reflection ? String(body.reflection).trim() : '';
  const idToken = body.idToken || (e.parameter ? e.parameter.idToken : null);
  if (!subject || SUBJECT_OPTIONS.indexOf(subject) === -1) {
    throw new Error('Invalid subject.');
  }
  if (reflection.length < MIN_REFLECTION_LENGTH || reflection.length > MAX_REFLECTION_LENGTH) {
    throw new Error('Reflection text must be between ' + MIN_REFLECTION_LENGTH + ' and ' + MAX_REFLECTION_LENGTH + ' characters.');
  }
  const email = getUserEmail(idToken);
  return withSheetLock(() => {
    const ss = SpreadsheetApp.openById(SHEET_ID);
    const sheet = ss.getSheetByName(SHEETS.reflections);
    const timestamp = new Date().toISOString();
    const key = buildDuplicateKey(email, subject, timestamp);
    upsertReflectionRow(sheet, {
      timestamp,
      studentEmail: email,
      studentName: lookupStudentName(ss, email),
      subject,
      reflection,
      status: 'submitted',
      eval: '',
      teacherComment: '',
      key
    });
    logEvent('INFO', 'submit', 'Reflection submitted', JSON.stringify({ email, subject, key }));
    return { success: true, key: key, timestamp: timestamp };
  });
}

function parseJsonBody(e) {
  if (!e || !e.postData || !e.postData.contents) {
    throw new Error('Request body missing.');
  }
  try {
    return JSON.parse(e.postData.contents);
  } catch (err) {
    throw new Error('Invalid JSON body: ' + err.message);
  }
}

function buildDuplicateKey(email, subject, isoTimestamp) {
  const month = isoTimestamp.substring(0, 7); // YYYY-MM
  return [email, subject, month].join('#');
}

function upsertReflectionRow(sheet, rowData) {
  const data = sheet.getDataRange().getValues();
  const headers = data.shift();
  const keyIndex = headers.indexOf('key');
  const key = rowData.key;
  let rowNumber = null;
  data.forEach((row, idx) => {
    if (row[keyIndex] === key) {
      rowNumber = idx + 2; // include header offset
    }
  });
  const values = REFLECTION_HEADERS.map(header => rowData[header] || '');
  if (rowNumber) {
    sheet.getRange(rowNumber, 1, 1, values.length).setValues([values]);
  } else {
    sheet.appendRow(values);
  }
}

/**
 * Returns the queue of reflections for approval (teacher API helper).
 */
function listApprovalQueue(e) {
  const subject = e.parameter && e.parameter.subject;
  const date = e.parameter && e.parameter.date; // yyyy-mm-dd
  const ss = SpreadsheetApp.openById(SHEET_ID);
  const reflectionsSheet = ss.getSheetByName(SHEETS.reflections);
  const data = reflectionsSheet.getDataRange().getValues();
  const headers = data.shift();
  const filtered = data.filter(row => {
    const matchesSubject = subject ? row[headers.indexOf('subject')] === subject : true;
    const matchesDate = date ? row[0].startsWith(date) : true;
    return matchesSubject && matchesDate && row[headers.indexOf('status')] !== 'approved';
  });
  return filtered.map(row => ({
    timestamp: row[headers.indexOf('timestamp')],
    studentEmail: row[headers.indexOf('studentEmail')],
    studentName: row[headers.indexOf('studentName')],
    subject: row[headers.indexOf('subject')],
    reflection: row[headers.indexOf('reflection')],
    status: row[headers.indexOf('status')],
    eval: row[headers.indexOf('eval')],
    teacherComment: row[headers.indexOf('teacherComment')]
  }));
}

/**
 * API endpoint that mirrors the spreadsheet approval flow. Not mandatory but
 * enables integration tests.
 */
function handleApproveApi(e) {
  const body = parseJsonBody(e);
  const approvals = body.approvals || [];
  if (!approvals.length) {
    return { success: false, message: 'No approvals submitted.' };
  }
  const ss = SpreadsheetApp.openById(SHEET_ID);
  const sheet = ss.getSheetByName(SHEETS.reflections);
  const headers = sheet.getDataRange().getValues()[0];
  const keyIndex = headers.indexOf('key');
  const statusIndex = headers.indexOf('status');
  const evalIndex = headers.indexOf('eval');
  const commentIndex = headers.indexOf('teacherComment');
  return withSheetLock(() => {
    approvals.forEach(item => {
      const row = findRowByKey(sheet, keyIndex, item.key);
      if (row) {
        sheet.getRange(row, statusIndex + 1).setValue('approved');
        sheet.getRange(row, evalIndex + 1).setValue(item.eval);
        sheet.getRange(row, commentIndex + 1).setValue(item.teacherComment);
      }
    });
    logEvent('INFO', 'approve_api', 'Approved via API', JSON.stringify({ count: approvals.length }));
    return { success: true, count: approvals.length };
  });
}

function findRowByKey(sheet, keyIndex, key) {
  const values = sheet.getRange(2, 1, sheet.getLastRow() - 1, sheet.getLastColumn()).getValues();
  for (let i = 0; i < values.length; i++) {
    if (values[i][keyIndex] === key) {
      return i + 2; // offset for header
    }
  }
  return null;
}

/**
 * Approval workflow menu actions.
 */
function menuExtractForApproval() {
  withSheetLock(() => {
    const ss = SpreadsheetApp.openById(SHEET_ID);
    const approvalsSheet = ss.getSheetByName(SHEETS.approvals);
    const config = readApprovalConfig(approvalsSheet);
    const queue = listApprovalQueue({ parameter: { subject: config.subject, date: config.date } });
    const rows = queue.map(item => [
      item.timestamp,
      item.studentEmail,
      item.studentName,
      item.subject,
      item.reflection,
      '',
      '',
      false,
      '抽出済み',
      ''
    ]);
    clearSheetBody(approvalsSheet);
    if (rows.length) {
      approvalsSheet.getRange(2, 1, rows.length, APPROVAL_HEADERS.length).setValues(rows);
    }
    logEvent('INFO', 'menu_extract', 'Queue extracted', JSON.stringify({ count: rows.length }));
  });
}

function menuGenerateAIForApprovals() {
  withSheetLock(() => {
    const ss = SpreadsheetApp.openById(SHEET_ID);
    const approvalsSheet = ss.getSheetByName(SHEETS.approvals);
    const config = readApprovalConfig(approvalsSheet);
    const rows = approvalsSheet.getRange(2, 1, Math.max(approvalsSheet.getLastRow() - 1, 0), APPROVAL_HEADERS.length).getValues();
    if (!rows.length) {
      SpreadsheetApp.getUi().alert('抽出された行がありません。');
      return;
    }
    rows.forEach((row, idx) => {
      const prompt = buildGeminiPrompt({
        subject: row[3],
        reflection: row[4],
        criteria: config.criteria,
        commentPrompt: config.commentPrompt,
        studentName: row[2] || '生徒',
        date: row[0]
      });
      const aiResult = callGemini(prompt, config.model, config.temperature);
      row[5] = aiResult.eval;
      row[6] = aiResult.comment;
      row[7] = true; // auto approve suggestion
      row[8] = 'AI生成済み';
      row[9] = aiResult.raw;
      approvalsSheet.getRange(idx + 2, 1, 1, row.length).setValues([row]);
      Utilities.sleep(300); // avoid rate limits
    });
    logEvent('INFO', 'menu_ai', 'AI generated', JSON.stringify({ rows: rows.length }));
  });
}

function menuApplyApprovals() {
  withSheetLock(() => {
    const ss = SpreadsheetApp.openById(SHEET_ID);
    const reflectionsSheet = ss.getSheetByName(SHEETS.reflections);
    const approvalsSheet = ss.getSheetByName(SHEETS.approvals);
    const rows = approvalsSheet.getRange(2, 1, Math.max(approvalsSheet.getLastRow() - 1, 0), APPROVAL_HEADERS.length).getValues();
    if (!rows.length) {
      SpreadsheetApp.getUi().alert('承認対象がありません。');
      return;
    }
    const headers = reflectionsSheet.getRange(1, 1, 1, reflectionsSheet.getLastColumn()).getValues()[0];
    const keyIndex = headers.indexOf('key');
    const statusIndex = headers.indexOf('status');
    const evalIndex = headers.indexOf('eval');
    const commentIndex = headers.indexOf('teacherComment');
    rows.forEach(row => {
      if (!row[7]) {
        return;
      }
      const key = buildDuplicateKey(row[1], row[3], row[0]);
      const rowNumber = findRowByKey(reflectionsSheet, keyIndex, key);
      if (rowNumber) {
        reflectionsSheet.getRange(rowNumber, statusIndex + 1).setValue('approved');
        reflectionsSheet.getRange(rowNumber, evalIndex + 1).setValue(row[5]);
        reflectionsSheet.getRange(rowNumber, commentIndex + 1).setValue(row[6]);
      }
    });
    SpreadsheetApp.getUi().alert('承認結果を反映しました。');
    logEvent('INFO', 'menu_apply', 'Approvals applied', JSON.stringify({ count: rows.length }));
  });
}

function readApprovalConfig(sheet) {
  const subject = sheet.getRange('B2').getValue();
  const date = sheet.getRange('B3').getDisplayValue();
  const criteria = sheet.getRange('B4').getValue();
  const commentPrompt = sheet.getRange('B5').getValue();
  const modelParam = sheet.getRange('B6').getValue();
  const [model, temperature] = parseModelParam(modelParam);
  return { subject, date, criteria, commentPrompt, model, temperature };
}

function parseModelParam(param) {
  if (!param) {
    return ['gemini-1.5-flash', 0.2];
  }
  try {
    const parsed = JSON.parse(param);
    return [parsed.model || 'gemini-1.5-flash', parsed.temperature || 0.2];
  } catch (err) {
    const parts = String(param).split(',').map(p => p.trim());
    return [parts[0] || 'gemini-1.5-flash', parts[1] ? parseFloat(parts[1]) : 0.2];
  }
}

function buildGeminiPrompt(options) {
  const prompt = [
    'あなたは小学校の先生です。以下の情報に基づいて振り返りを評価してください。',
    '評価は必ず S/A/B のいずれかで1文字のみ。',
    'コメントは60〜120文字で、肯定から始め最後に次への提案を含めてください。',
    '---',
    '教科: ' + options.subject,
    '氏名: ' + options.studentName,
    '日付: ' + options.date,
    '評価基準: ' + options.criteria,
    'コメント指示: ' + options.commentPrompt,
    '振り返り本文:\n' + options.reflection
  ].join('\n');
  return prompt;
}

function callGemini(prompt, model, temperature) {
  if (!GEMINI_API_KEY) {
    throw new Error('GEMINI_API_KEY is not configured.');
  }
  const endpoint = 'https://generativelanguage.googleapis.com/v1beta/models/' + (model || 'gemini-1.5-flash') + ':generateContent?key=' + GEMINI_API_KEY;
  const payload = {
    contents: [{ role: 'user', parts: [{ text: prompt }]}],
    safetySettings: [],
    generationConfig: {
      temperature: temperature != null ? temperature : 0.2,
      maxOutputTokens: 256
    }
  };
  const response = UrlFetchApp.fetch(endpoint, {
    method: 'post',
    contentType: 'application/json',
    payload: JSON.stringify(payload),
    muteHttpExceptions: true
  });
  const code = response.getResponseCode();
  if (code !== 200) {
    throw new Error('Gemini API error: ' + response.getContentText());
  }
  const data = JSON.parse(response.getContentText());
  const text = data.candidates && data.candidates[0] && data.candidates[0].content && data.candidates[0].content.parts[0].text || '';
  const evalMatch = text.match(/eval\s*[:：]\s*([SAB])/i);
  const commentMatch = text.match(/comment\s*[:：]\s*([^\n]+)/i);
  const evaluation = evalMatch ? evalMatch[1].toUpperCase() : extractFirstGrade(text);
  const comment = commentMatch ? commentMatch[1].trim() : text.trim();
  return {
    eval: evaluation,
    comment: comment,
    raw: text
  };
}

function extractFirstGrade(text) {
  const match = text.match(/\b([SAB])\b/);
  return match ? match[1] : 'B';
}

/**
 * Applies a lock while executing a callback to prevent concurrent edits.
 */
function withSheetLock(callback) {
  const lock = LockService.getScriptLock();
  if (!lock.tryLock(5000)) {
    throw new Error('Failed to obtain lock. Please retry.');
  }
  try {
    return callback();
  } finally {
    lock.releaseLock();
  }
}

/**
 * Clears the body (rows below the header) of the specified sheet.
 */
function clearSheetBody(sheet) {
  const lastRow = sheet.getLastRow();
  if (lastRow > 1) {
    sheet.getRange(2, 1, lastRow - 1, sheet.getLastColumn()).clearContent();
  }
}

function buildRosterMap(sheet) {
  const data = sheet.getDataRange().getValues();
  data.shift();
  const map = {};
  data.forEach(row => {
    if (row[0]) {
      map[row[0]] = {
        displayName: row[1],
        kanaName: row[2],
        class: row[3]
      };
    }
  });
  return map;
}

function lookupStudentName(ss, email) {
  const rosterSheet = ss.getSheetByName(SHEETS.roster);
  const data = rosterSheet.getDataRange().getValues();
  data.shift();
  for (let i = 0; i < data.length; i++) {
    if (data[i][0] === email) {
      return data[i][2] || data[i][1] || '';
    }
  }
  return '';
}

function logEvent(level, event, message, detail) {
  try {
    const ss = SpreadsheetApp.openById(SHEET_ID);
    const sheet = ss.getSheetByName(SHEETS.logs);
    sheet.appendRow([
      new Date().toISOString(),
      level,
      event,
      message + (detail ? ' ' + detail : '')
    ]);
  } catch (err) {
    console.log('Logging failed: ' + err.message);
  }
}

/**
 * Bootstraps dummy data to help testers verify the UI.
 */
function bootstrapDummyData() {
  return withSheetLock(() => {
    const ss = SpreadsheetApp.openById(SHEET_ID);
    const reflectionsSheet = ss.getSheetByName(SHEETS.reflections);
    const rosterSheet = ss.getSheetByName(SHEETS.roster);
    const now = new Date();
    const email = Session.getActiveUser().getEmail() || 'student@example.jp';
    rosterSheet.appendRow([email, '山田太郎', 'ヤマダ タロウ', '6-1']);
    const base = new Date(now.getFullYear(), now.getMonth() - 1, 15).toISOString();
    const key = buildDuplicateKey(email, '国', base);
    reflectionsSheet.appendRow([
      base,
      email,
      'ヤマダ タロウ',
      '国',
      '前回の国語の振り返りサンプルです。',
      'approved',
      'A',
      '丁寧に読めていました。次は意見を増やしましょう。',
      key
    ]);
    logEvent('INFO', 'bootstrap', 'Dummy data inserted', email);
    return { success: true };
  });
}

/**
 * Exposes include() for templated HTML files.
 */
function include(filename) {
  return HtmlService.createHtmlOutputFromFile(filename).getContent();
}
