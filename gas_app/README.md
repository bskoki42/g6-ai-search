# 振り返り Web アプリ（Google Apps Script + Google スプレッドシート）

このフォルダーには、学習者の振り返り投稿と教員の承認ワークフローを Google Apps Script で実装するためのソースコードと手順をまとめています。プロジェクトは以下の 3 ステップで共有する想定です。

1. **プロジェクト構成（ファイルツリー）**
2. **各ファイルの完全コード**
3. **セットアップ・実行・テスト手順**

## 1. プロジェクト構成

```
gas_app/
  Code.gs                 # Apps Script のメインコード。API・シート I/O・AI 連携を担当。
  README.md               # 本ドキュメント。
  views/
    index.html            # 学習者向け UI（ログイン必須）。
    teacher.html          # 教員向け簡易ビュー。
  public/
    app.html              # フロントエンド JS。学習者・教員ビュー共通のロジック。
    style.html            # 共通スタイルシート。
```

> **補足**: Apps Script のエディタでは `Code.gs` のほかに `views/index.html` などを HTML ファイルとして追加します。`include('public/style')` で読み込むために拡張子は `.html` にしています。

## 2. 各ファイルの完全コード

### 2.1 `Code.gs`

```gs
<?!= include('Code') ?>
```

> Apps Script ではこの README から直接貼り付けるのではなく、`Code.gs` ファイルをそのまま使用してください。内容はリポジトリの `Code.gs` と同一です。

### 2.2 `views/index.html`

```html
<?!= include('views/index') ?>
```

### 2.3 `views/teacher.html`

```html
<?!= include('views/teacher') ?>
```

### 2.4 `public/app.html`

```html
<?!= include('public/app') ?>
```

### 2.5 `public/style.html`

```html
<?!= include('public/style') ?>
```

> **メモ**: 上記の `<?!= include(...) ?>` は README を簡潔に保つためのショートカットです。実際のコードは各ファイルを参照してください。

## 3. セットアップ手順

1. **スプレッドシートの準備**
   - ID: `1tF0WOdhnbpRDJ5-9mOWwThjzXb-CjnHhwp7Dka9vuko` のスプレッドシートに以下のシートを作成し、1 行目にヘッダーを入力します。
     - `Reflections`: `timestamp, studentEmail, studentName, subject, reflection, status, eval, teacherComment, key`
     - `Approvals`: `timestamp, studentEmail, studentName, subject, reflection, evaluation, teacherComment, approved, statusLog, rawResponse`
     - `Roster`: `email, displayName, kanaName, class`
     - `Logs`: `timestamp, level, event, detail`
2. **Apps Script プロジェクト作成**
   - スプレッドシートから「拡張機能 ▶ Apps Script」を開き、既存ファイルを削除して以下を追加します。
     - `Code.gs`（本リポジトリの内容をコピー）
     - HTML ファイル `views/index`, `views/teacher`, `public/app`, `public/style`
3. **スクリプトプロパティ設定**
   - `SHEET_ID` に管理スプレッドシート ID (`1tF0WOdhnbpRDJ5-9mOWwThjzXb-CjnHhwp7Dka9vuko`)
   - `GEMINI_API_KEY` に Gemini API キー（値は Apps Script 側でのみ設定）
   - 任意で `APP_ACCESS_KEY` を設定し、フロントエンド JS に `window.APP_ACCESS_KEY = '...';` を埋め込むと CSRF 対策ヘッダーが利用できます。
4. **公開設定**
   - 「デプロイ ▶ 新しいデプロイ ▶ ウェブアプリ」から、実行するユーザーを「アクセスするユーザー」とし、アクセス権を「ドメイン内のユーザーのみ」に設定します。
5. **メニュー確認**
   - スプレッドシートを再読み込みすると `📤 承認ワークフロー` メニューが追加されます。

## 4. 実行 / テスト手順

1. **ダミーデータ挿入**
   - 学習者画面で「ダミーデータ生成」をクリックすると、`Roster` と `Reflections` にサンプルが登録されます。
2. **学習者送信テスト**
   - 教科を選択し、振り返り本文を入力して送信します。
   - 送信後に「前回の振り返り」が即時更新されることを確認します。
3. **教員承認テスト**
   - スプレッドシートで `📤 承認ワークフロー ▶ ① 抽出` を実行し、教科と日付を指定して投稿を取得します。
   - `② AI評価` を実行すると Gemini API で `evaluation` と `teacherComment` が埋まります。
   - `③ 承認反映` を実行すると `Reflections` に評価と先生コメントが書き戻され、学習者画面の前回表示にも反映されます。
4. **異常系確認**
   - ログアウト状態でアクセスすると Google ログインが要求されることを確認します。
   - 長文（2000 文字超）を送信しようとするとバリデーションエラーになることを確認します。

## 5. 補足情報

- `LockService` を利用して送信・承認の同時編集を防いでいます。
- `Logs` シートに主要イベントを書き込み、監査証跡として利用できます。
- Gemini API が失敗した場合は例外を投げるため、教員は手動でコメントを入力する運用に切り替えられます。
- Google Identity Services を利用する場合は、フロントエンドで ID トークンを取得して `/api/submit` のペイロードに含めることでドメイン外ユーザーにも対応可能です。
