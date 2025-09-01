# REST API 設計

## 共通エラーレスポンス形式
```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "エラー内容"
  }
}
```

## 翻訳関連

### GET /api/translations
**認証**: 不要
**説明**: 保存された翻訳履歴を一覧取得する。

リクエスト例:
```bash
curl -X GET http://localhost:8000/api/translations
```

レスポンス例:
```json
[
  {
    "id": 1,
    "source_text": "Hello",
    "source_lang": "en",
    "translated_text": "こんにちは",
    "target_lang": "ja",
    "created_at": "2025-01-01T12:00:00Z"
  }
]
```

**エラー**:
- 500 INTERNAL_SERVER_ERROR

### POST /api/translations
**認証**: 不要
**説明**: テキストを翻訳し履歴として保存する。

リクエスト例:
```bash
curl -X POST http://localhost:8000/api/translations \
  -H "Content-Type: application/json" \
  -d '{"source_text":"Hello","source_lang":"en","target_lang":"ja"}'
```

レスポンス例:
```json
{
  "id": 2,
  "source_text": "Hello",
  "source_lang": "en",
  "translated_text": "こんにちは",
  "target_lang": "ja",
  "created_at": "2025-01-01T12:01:00Z"
}
```

**エラー**:
- 400 INVALID_REQUEST
- 500 INTERNAL_SERVER_ERROR

## 表示設定関連

### GET /api/settings
**認証**: 不要
**説明**: 字幕表示に関する設定を取得する。

リクエスト例:
```bash
curl -X GET http://localhost:8000/api/settings
```

レスポンス例:
```json
{
  "font_size": 16,
  "background_color": "#000000",
  "original_color": "#0000ff",
  "translated_color": "#ffa500"
}
```

**エラー**:
- 500 INTERNAL_SERVER_ERROR

### POST /api/settings
**認証**: 不要
**説明**: 表示設定を更新する。

リクエスト例:
```bash
curl -X POST http://localhost:8000/api/settings \
  -H "Content-Type: application/json" \
  -d '{"font_size":18,"background_color":"#ffffff"}'
```

レスポンス例:
```json
{
  "font_size": 18,
  "background_color": "#ffffff",
  "original_color": "#0000ff",
  "translated_color": "#ffa500"
}
```

**エラー**:
- 400 INVALID_REQUEST
- 500 INTERNAL_SERVER_ERROR
