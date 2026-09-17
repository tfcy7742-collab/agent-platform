"""SQLite 数据访问层。

存放三类数据：
1. **观测数据**：``runs`` / ``steps`` —— 每次提问的执行轨迹（可观测性的落地）；
2. **会话数据**：``sessions`` / ``messages`` —— 多轮对话历史（Planner 改写问题要用）；
3. **文档元数据**：``documents`` —— 已入库文档的映射关系，用于列表展示与按文档删除向量。

设计要点
--------
* 使用标准库 ``sqlite3``，零外部依赖；开启 **WAL** 模式提升并发读性能；
* 连接按线程缓存（``threading.local``），因为 FastAPI 的线程池会复用不同线程；
* 所有 SQL 一律参数化，杜绝注入；
* 表结构在 ``init_db()`` 中幂等创建，进程启动时调用一次即可。
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

from config.settings import get_settings

# 线程本地连接缓存
_local = threading.local()

SCHEMA_STATEMENTS: List[str] = [
    # ---------------- 文档元数据 ----------------
    """
    CREATE TABLE IF NOT EXISTS documents (
        doc_id       TEXT PRIMARY KEY,
        file_name    TEXT NOT NULL,
        stored_name  TEXT NOT NULL,
        ext          TEXT NOT NULL,
        size_bytes   INTEGER NOT NULL DEFAULT 0,
        chunk_count  INTEGER NOT NULL DEFAULT 0,
        char_count   INTEGER NOT NULL DEFAULT 0,
        page_count   INTEGER,
        ingest_ms    INTEGER NOT NULL DEFAULT 0,
        degraded     INTEGER NOT NULL DEFAULT 0,
        created_at   TEXT NOT NULL
    )
    """,
    # ---------------- 会话 ----------------
    """
    CREATE TABLE IF NOT EXISTS sessions (
        session_id   TEXT PRIMARY KEY,
        title        TEXT DEFAULT '',
        created_at   TEXT NOT NULL,
        updated_at   TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS messages (
        seq          INTEGER PRIMARY KEY AUTOINCREMENT,
        msg_id       TEXT NOT NULL UNIQUE,
        session_id   TEXT NOT NULL,
        role         TEXT NOT NULL,
        content      TEXT NOT NULL,
        run_id       TEXT,
        created_at   TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, created_at)",
    # ---------------- 执行轨迹 ----------------
    """
    CREATE TABLE IF NOT EXISTS runs (
        run_id       TEXT PRIMARY KEY,
        session_id   TEXT NOT NULL,
        question     TEXT NOT NULL,
        rewritten    TEXT,
        answer       TEXT DEFAULT '',
        refused      INTEGER NOT NULL DEFAULT 0,
        status       TEXT NOT NULL DEFAULT 'success',
        steps        INTEGER NOT NULL DEFAULT 0,
        tools_used   TEXT DEFAULT '',
        total_ms     INTEGER NOT NULL DEFAULT 0,
        total_tokens INTEGER NOT NULL DEFAULT 0,
        cost_est     REAL NOT NULL DEFAULT 0,
        degraded     INTEGER NOT NULL DEFAULT 0,
        created_at   TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_runs_created ON runs(created_at DESC)",
    """
    CREATE TABLE IF NOT EXISTS steps (
        seq          INTEGER PRIMARY KEY AUTOINCREMENT,
        step_id      TEXT NOT NULL UNIQUE,
        run_id       TEXT NOT NULL,
        idx          INTEGER NOT NULL,
        type         TEXT NOT NULL,
        tool_name    TEXT,
        args         TEXT DEFAULT '{}',
        ok           INTEGER NOT NULL DEFAULT 1,
        degraded     INTEGER NOT NULL DEFAULT 0,
        latency_ms   INTEGER NOT NULL DEFAULT 0,
        prompt_tokens     INTEGER NOT NULL DEFAULT 0,
        completion_tokens INTEGER NOT NULL DEFAULT 0,
        error        TEXT,
        error_type   TEXT,
        detail       TEXT DEFAULT '{}',
        created_at   TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_steps_run ON steps(run_id, idx)",
]


def _now() -> str:
    """当前时间字符串（本地时区，秒级）。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def new_id(prefix: str = "") -> str:
    """生成短 id（16 位十六进制），可带前缀，便于日志里肉眼区分类型。"""
    return f"{prefix}{uuid.uuid4().hex[:16]}"


def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """获取当前线程的数据库连接（不存在则创建）。

    Args:
        db_path: 覆盖配置中的数据库路径（测试时用临时文件）。
    """
    path = Path(db_path) if db_path else get_settings().db_file
    cached = getattr(_local, "conn", None)
    cached_path = getattr(_local, "path", None)
    if cached is not None and cached_path == str(path):
        return cached

    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=10.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    # WAL：读写并发更好；NORMAL 同步级别在本地场景下足够安全且更快
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    _local.conn = conn
    _local.path = str(path)
    return conn


def close_connection() -> None:
    """关闭当前线程的连接（测试收尾时调用）。"""
    conn = getattr(_local, "conn", None)
    if conn is not None:
        conn.close()
        _local.conn = None
        _local.path = None


@contextmanager
def transaction(db_path: Optional[Path] = None) -> Iterator[sqlite3.Connection]:
    """事务上下文：正常提交，异常回滚。"""
    conn = get_connection(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def init_db(db_path: Optional[Path] = None) -> None:
    """幂等初始化表结构。"""
    conn = get_connection(db_path)
    for statement in SCHEMA_STATEMENTS:
        conn.execute(statement)
    conn.commit()


# ---------------------------------------------------------------------------
# 文档元数据
# ---------------------------------------------------------------------------
def upsert_document(record: Dict[str, Any], db_path: Optional[Path] = None) -> None:
    """写入（或覆盖）一条文档记录；``doc_id`` 为文件内容哈希，天然去重。"""
    record.setdefault("created_at", _now())
    with transaction(db_path) as conn:
        conn.execute(
            """
            INSERT INTO documents (doc_id, file_name, stored_name, ext, size_bytes,
                                   chunk_count, char_count, page_count, ingest_ms,
                                   degraded, created_at)
            VALUES (:doc_id, :file_name, :stored_name, :ext, :size_bytes,
                    :chunk_count, :char_count, :page_count, :ingest_ms,
                    :degraded, :created_at)
            ON CONFLICT(doc_id) DO UPDATE SET
                file_name=excluded.file_name,
                stored_name=excluded.stored_name,
                size_bytes=excluded.size_bytes,
                chunk_count=excluded.chunk_count,
                char_count=excluded.char_count,
                page_count=excluded.page_count,
                ingest_ms=excluded.ingest_ms,
                degraded=excluded.degraded,
                created_at=excluded.created_at
            """,
            record,
        )


def list_documents(db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """按上传时间倒序列出所有文档。"""
    conn = get_connection(db_path)
    rows = conn.execute("SELECT * FROM documents ORDER BY created_at DESC").fetchall()
    return [dict(row) for row in rows]


def get_document(doc_id: str, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """按 doc_id 查询单条文档记录。"""
    conn = get_connection(db_path)
    row = conn.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
    return dict(row) if row else None


def delete_document(doc_id: str, db_path: Optional[Path] = None) -> bool:
    """删除文档元数据记录，返回是否确实删掉了一条。"""
    with transaction(db_path) as conn:
        cursor = conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
        return cursor.rowcount > 0


def document_stats(db_path: Optional[Path] = None) -> Dict[str, Any]:
    """文档汇总统计（供 /health 与 UI 顶部展示）。"""
    conn = get_connection(db_path)
    row = conn.execute(
        "SELECT COUNT(*) AS docs, COALESCE(SUM(chunk_count), 0) AS chunks, "
        "COALESCE(SUM(char_count), 0) AS chars FROM documents"
    ).fetchone()
    return {"documents": row["docs"], "chunks": row["chunks"], "chars": row["chars"]}


# ---------------------------------------------------------------------------
# 会话与消息
# ---------------------------------------------------------------------------
def ensure_session(session_id: str, title: str = "", db_path: Optional[Path] = None) -> str:
    """确保会话存在（不存在则创建），返回 session_id。"""
    now = _now()
    with transaction(db_path) as conn:
        conn.execute(
            """
            INSERT INTO sessions (session_id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET updated_at=excluded.updated_at
            """,
            (session_id, title, now, now),
        )
    return session_id


def add_message(
    session_id: str,
    role: str,
    content: str,
    run_id: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> str:
    """追加一条消息（user / assistant）。"""
    msg_id = new_id("msg_")
    ensure_session(session_id, db_path=db_path)
    with transaction(db_path) as conn:
        conn.execute(
            """
            INSERT INTO messages (msg_id, session_id, role, content, run_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (msg_id, session_id, role, content, run_id, _now()),
        )
    return msg_id


def get_history(
    session_id: str,
    limit: int = 20,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """获取会话最近 ``limit`` 条消息，按对话顺序（时间正序）返回。

    排序用自增列 ``seq`` 而不是时间戳：同一秒内产生的多条消息时间戳相同，
    只按 ``created_at`` 排序会得到不确定的顺序，而 ``seq`` 是单调递增的插入序。
    """
    conn = get_connection(db_path)
    rows = conn.execute(
        """
        SELECT * FROM (
            SELECT * FROM messages WHERE session_id = ? ORDER BY seq DESC LIMIT ?
        ) ORDER BY seq ASC
        """,
        (session_id, limit),
    ).fetchall()
    return [dict(row) for row in rows]


def clear_history(session_id: str, db_path: Optional[Path] = None) -> int:
    """清空某会话的消息，返回删除条数。"""
    with transaction(db_path) as conn:
        cursor = conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        return cursor.rowcount


# ---------------------------------------------------------------------------
# 执行轨迹
# ---------------------------------------------------------------------------
def insert_run(record: Dict[str, Any], db_path: Optional[Path] = None) -> str:
    """写入一条 run 记录。"""
    record.setdefault("created_at", _now())
    fields = (
        "run_id", "session_id", "question", "rewritten", "answer", "refused", "status",
        "steps", "tools_used", "total_ms", "total_tokens", "cost_est", "degraded", "created_at",
    )
    values = {key: record.get(key) for key in fields}
    with transaction(db_path) as conn:
        conn.execute(
            f"INSERT OR REPLACE INTO runs ({', '.join(fields)}) "
            f"VALUES ({', '.join(':' + key for key in fields)})",
            values,
        )
    return str(values["run_id"])


def insert_step(record: Dict[str, Any], db_path: Optional[Path] = None) -> str:
    """写入一条 step 记录；``args`` / ``detail`` 以 JSON 文本存储。"""
    record = dict(record)
    record.setdefault("step_id", new_id("step_"))
    record.setdefault("created_at", _now())
    for key in ("args", "detail"):
        value = record.get(key)
        if not isinstance(value, str):
            record[key] = json.dumps(value or {}, ensure_ascii=False)
    fields = (
        "step_id", "run_id", "idx", "type", "tool_name", "args", "ok", "degraded",
        "latency_ms", "prompt_tokens", "completion_tokens", "error", "error_type",
        "detail", "created_at",
    )
    values = {key: record.get(key) for key in fields}
    with transaction(db_path) as conn:
        conn.execute(
            f"INSERT OR REPLACE INTO steps ({', '.join(fields)}) "
            f"VALUES ({', '.join(':' + key for key in fields)})",
            values,
        )
    return str(values["step_id"])


def list_runs(
    limit: int = 20,
    session_id: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """列出最近的 run 记录。"""
    conn = get_connection(db_path)
    if session_id:
        rows = conn.execute(
            "SELECT * FROM runs WHERE session_id = ? ORDER BY created_at DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [_decode_run(dict(row)) for row in rows]


def get_run(run_id: str, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """查询单次执行的完整明细（含分步）。"""
    conn = get_connection(db_path)
    row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    if row is None:
        return None
    run = _decode_run(dict(row))
    steps = conn.execute(
        "SELECT * FROM steps WHERE run_id = ? ORDER BY idx ASC", (run_id,)
    ).fetchall()
    decoded_steps: List[Dict[str, Any]] = []
    for step in steps:
        item = dict(step)
        for key in ("args", "detail"):
            try:
                item[key] = json.loads(item.get(key) or "{}")
            except (json.JSONDecodeError, TypeError):
                item[key] = {}
        decoded_steps.append(item)
    run["steps_detail"] = decoded_steps
    return run


def _decode_run(row: Dict[str, Any]) -> Dict[str, Any]:
    """把 run 行里的 0/1 字段还原成布尔值，tools_used 还原成列表。"""
    for key in ("refused", "degraded"):
        if key in row:
            row[key] = bool(row[key])
    if "tools_used" in row:
        row["tools_used"] = [t for t in str(row["tools_used"] or "").split(",") if t]
    return row


def trace_stats(db_path: Optional[Path] = None) -> Dict[str, Any]:
    """轨迹汇总统计（成本看板与 /health 使用）。"""
    conn = get_connection(db_path)
    row = conn.execute(
        """
        SELECT COUNT(*) AS runs,
               COALESCE(SUM(total_tokens), 0) AS tokens,
               COALESCE(SUM(cost_est), 0) AS cost,
               COALESCE(SUM(refused), 0) AS refused,
               COALESCE(SUM(degraded), 0) AS degraded,
               COALESCE(AVG(total_ms), 0) AS avg_ms
        FROM runs
        """
    ).fetchone()
    return {
        "runs": row["runs"],
        "tokens": row["tokens"],
        "cost_est": round(float(row["cost"]), 6),
        "refused": row["refused"],
        "degraded": row["degraded"],
        "avg_ms": round(float(row["avg_ms"]), 1),
    }


def reset_all(db_path: Optional[Path] = None, tables: Optional[Iterable[str]] = None) -> None:
    """清空数据（测试或"重置演示数据"用）。"""
    target = list(tables) if tables else [
        "steps", "runs", "messages", "sessions", "documents"
    ]
    with transaction(db_path) as conn:
        for table in target:
            conn.execute(f"DELETE FROM {table}")  # 表名为白名单常量，非用户输入
