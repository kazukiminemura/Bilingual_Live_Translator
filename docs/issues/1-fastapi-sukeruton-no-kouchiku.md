#1 FastAPIスケルトンの構築

概要 / 目的
FastAPIでHTMLとWebSocketを提供する最小構成を作成する。
• 依存:
• ラベル: backend

■ スコープ / 作業項目
- FastAPIアプリケーションの初期化とテンプレート設定
- `/` エンドポイントで `index.html` を返すルートを実装
- `/ws` WebSocketでクライアント接続を受け付け、接続中クライアントをセットで管理
- 最小限のログ出力を追加
- READMEに起動手順を追記

■ ゴール / 完了条件 (Acceptance Criteria)
- [ ] `/` が `index.html` を返す
- [ ] `/ws` でWebSocket接続を受け付ける
- [ ] 接続中クライアントをセットで管理する
- [ ] 簡単なログ出力を行う
- [ ] READMEに起動手順を記載する

■ テスト観点
- リクエスト / WebSocket
- 検証方法: curlで`/`へリクエストしHTMLを取得、ブラウザ/ツールでWebSocket接続確認

（必要なら）要確認事項:
- 特になし
