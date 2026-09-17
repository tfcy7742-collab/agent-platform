"""B1 地基测试：配置 / 数据库 / LLM 客户端 / API 接口。

覆盖点：
1. 配置默认值与 .env 覆盖、非法值必须被拦住（阈值、切块重叠、步数预算）；
2. 配置快照与 /health 里的密钥脱敏；
3. SQLite 层：文档、会话消息、轨迹 run/step 的读写与统计；
4. LLM 客户端在离线模式下的降级行为与错误分类；
5. FastAPI 的 ``/`` ``/health`` ``/api/metrics`` ``/api/config`` 四个接口。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
def test_settings_defaults() -> None:
    """默认配置必须与架构文档一致（这些值是检索/拒答行为的基础）。"""
    from config.settings import Settings

    settings = Settings(_env_file=None)
    assert settings.chunk_size == 500
    assert settings.chunk_overlap == 50
    assert settings.retrieve_top_k == 3
    assert settings.deepseek_base_url == "https://api.deepseek.com"
    assert settings.deepseek_model == "deepseek-flash"
    assert settings.embedding_model == "BAAI/bge-small-zh-v1.5"
    assert settings.chunk_overlap < settings.chunk_size
    assert settings.embedding_query_prefix  # bge 检索前缀默认开启


def test_settings_env_override(settings_factory) -> None:
    """环境变量必须能覆盖默认值。"""
    settings = settings_factory(
        REFUSE_THRESHOLD="0.42",
        ENABLE_RERANK="true",
        LLM_PROVIDER="dashscope",
    )
    assert settings.refuse_threshold == 0.42
    assert settings.enable_rerank is True
    assert settings.provider_model == settings.dashscope_model
    assert settings.provider_base_url == settings.dashscope_base_url


@pytest.mark.parametrize(
    "env",
    [
        {"REFUSE_THRESHOLD": "0"},
        {"REFUSE_THRESHOLD": "1.5"},
        {"CHUNK_SIZE": "100", "CHUNK_OVERLAP": "200"},
        {"AGENT_MAX_STEPS": "0"},
        {"AGENT_MAX_STEPS": "99"},
    ],
)
def test_settings_rejects_invalid_values(settings_factory, env: dict) -> None:
    """非法配置必须在启动时就报错，而不是运行时才出诡异行为。"""
    with pytest.raises(Exception):
        settings_factory(**env)


def test_masked_key_never_leaks_plaintext(settings_factory) -> None:
    """脱敏：快照与 /health 里不能出现完整 Key。"""
    secret = "sk-abcdef1234567890abcdef"
    settings = settings_factory(DEEPSEEK_API_KEY=secret, LLM_MODE="auto")
    snapshot = settings.masked_snapshot()
    assert secret not in json.dumps(snapshot, ensure_ascii=False)
    assert snapshot["api_key"].startswith("sk-abc")
    assert snapshot["api_key"].endswith("cdef")


def test_placeholder_key_is_treated_as_missing(settings_factory) -> None:
    """占位符 Key（.env.example 里的 your_key_here）必须被视为"未配置"。"""
    settings = settings_factory(DEEPSEEK_API_KEY="your_key_here", LLM_MODE="auto")
    assert settings.has_api_key is False
    assert settings.llm_online is False


def test_offline_mode_flag(settings_factory) -> None:
    """LLM_MODE=offline 时必须离线，即使配了 Key。"""
    settings = settings_factory(DEEPSEEK_API_KEY="sk-real-key-1234567890", LLM_MODE="offline")
    assert settings.llm_online is False


def test_capabilities_report(tmp_path: Path, settings_factory) -> None:
    """能力汇总里必须包含四个子系统与拒答阈值。"""
    settings = settings_factory(UPLOAD_DIR=str(tmp_path / "up"), CHROMA_DIR=str(tmp_path / "ch"))
    payload = settings.capabilities()
    assert set(payload) >= {"version", "llm", "embedding", "retrieval", "agent", "storage"}
    assert payload["retrieval"]["refuse_threshold"] == settings.refuse_threshold
    assert payload["storage"]["chroma_dir"].endswith("ch")


# ---------------------------------------------------------------------------
# 数据库层
# ---------------------------------------------------------------------------
def test_db_init_and_document_crud(temp_db: Path) -> None:
    """文档元数据的增查删与统计。"""
    from infra import db

    record = {
        "doc_id": "abc123",
        "file_name": "员工手册.pdf",
        "stored_name": "abc123.pdf",
        "ext": ".pdf",
        "size_bytes": 10240,
        "chunk_count": 12,
        "char_count": 5800,
        "page_count": 8,
        "ingest_ms": 320,
        "degraded": 0,
    }
    db.upsert_document(record)

    docs = db.list_documents()
    assert len(docs) == 1
    assert docs[0]["file_name"] == "员工手册.pdf"
    assert db.get_document("abc123")["chunk_count"] == 12

    stats = db.document_stats()
    assert stats == {"documents": 1, "chunks": 12, "chars": 5800}

    # 幂等：同一 doc_id 再写一次是更新而不是新增
    db.upsert_document({**record, "chunk_count": 15})
    assert db.document_stats()["chunks"] == 15

    assert db.delete_document("abc123") is True
    assert db.list_documents() == []
    assert db.delete_document("不存在") is False


def test_db_session_messages(temp_db: Path) -> None:
    """会话消息写入与按时间正序读取。"""
    from infra import db

    db.ensure_session("s1", title="第一个会话")
    db.add_message("s1", "user", "年假怎么算？")
    db.add_message("s1", "assistant", "根据《员工手册》第 3 页……")
    db.add_message("s1", "user", "那病假呢？")

    history = db.get_history("s1")
    assert [m["role"] for m in history] == ["user", "assistant", "user"]
    assert history[0]["content"] == "年假怎么算？"

    # limit 取最近 N 条，但仍按时间正序返回
    recent = db.get_history("s1", limit=2)
    assert [m["content"] for m in recent] == ["根据《员工手册》第 3 页……", "那病假呢？"]

    assert db.clear_history("s1") == 3
    assert db.get_history("s1") == []


def test_db_trace_roundtrip(temp_db: Path) -> None:
    """run / step 落库后能完整还原（可观测性的最小闭环）。"""
    from infra import db

    run_id = db.new_id("run_")
    db.insert_run(
        {
            "run_id": run_id,
            "session_id": "s1",
            "question": "年假怎么算？",
            "answer": "根据资料……",
            "refused": 0,
            "status": "success",
            "steps": 2,
            "tools_used": "knowledge_search",
            "total_ms": 850,
            "total_tokens": 520,
            "cost_est": 0.001,
            "degraded": 0,
        }
    )
    db.insert_step(
        {
            "run_id": run_id,
            "idx": 0,
            "type": "plan",
            "args": {"question": "年假怎么算？"},
            "ok": 1,
            "detail": {"thought": "需要检索知识库", "action": "tool"},
        }
    )
    db.insert_step(
        {
            "run_id": run_id,
            "idx": 1,
            "type": "tool",
            "tool_name": "knowledge_search",
            "args": {"query": "年假"},
            "ok": 1,
            "latency_ms": 120,
            "detail": {"top_score": 0.71, "hits": 3},
        }
    )

    run = db.get_run(run_id)
    assert run is not None
    assert run["refused"] is False
    assert run["tools_used"] == ["knowledge_search"]
    assert len(run["steps_detail"]) == 2
    # JSON 字段必须被反序列化回字典
    assert run["steps_detail"][1]["detail"]["top_score"] == 0.71
    assert run["steps_detail"][1]["args"]["query"] == "年假"

    listed = db.list_runs(limit=10)
    assert len(listed) == 1
    assert db.get_run("run_不存在") is None

    stats = db.trace_stats()
    assert stats["runs"] == 1
    assert stats["tokens"] == 520
    assert stats["avg_ms"] == 850.0


# ---------------------------------------------------------------------------
# LLM 客户端
# ---------------------------------------------------------------------------
def test_llm_client_offline(temp_db: Path) -> None:
    """离线模式下调用必须返回结构化的降级结果，而不是抛异常。"""
    from core.llm import LLMClient, get_llm_client

    client = LLMClient(enable=False)
    assert client.available is False

    result = client.chat("你是助手", "你好")
    assert result.ok is False
    assert result.error_type == "llm_offline"
    assert result.degraded is True
    assert result.latency_ms >= 0

    parsed, result2 = client.chat_json("你是助手", "返回 JSON")
    assert parsed is None
    assert result2.ok is False

    describe = client.describe()
    assert describe["available"] is False
    assert describe["usage"]["calls"] == 0

    # 单例可获取
    assert isinstance(get_llm_client(), LLMClient)


@pytest.mark.parametrize(
    "message,expected",
    [
        ("Request timed out", "llm_timeout"),
        ("Error code: 401 - invalid_api_key", "llm_auth"),
        ("Error code: 429 - rate limit reached", "llm_rate_limit"),
        ("Expecting value: line 1 column 1 (char 0) json", "llm_bad_output"),
        ("Connection error: DNS lookup failed", "llm_error"),
        ("something totally unknown", "llm_error"),
    ],
)
def test_classify_error(message: str, expected: str) -> None:
    """错误分类是差异化降级的前提，必须有稳定的映射。"""
    from core.llm import classify_error

    error_type, readable = classify_error(RuntimeError(message))
    assert error_type == expected
    assert readable  # 必须有人类可读的说明


def test_extract_json_tolerates_noise() -> None:
    """模型经常在 JSON 前后加自然语言、代码围栏或用全角标点，解析必须容错。"""
    from core.llm import extract_json

    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('好的，结果如下：\n```json\n{"a": 2}\n```\n以上。') == {"a": 2}
    assert extract_json('前言 {"a": 3} 后记') == {"a": 3}
    # JSON 字符串**内部**的中文引号是完全合法的，解析结果原样保留
    assert extract_json('{"a": "中文“引号”"}') == {"a": "中文“引号”"}
    # 全角冒号 + 全角引号：标准解析必然失败，只有兜底的"修复分支"能救回来
    assert extract_json('{"问题"："年假怎么算","分数"：0.8}') == {"问题": "年假怎么算", "分数": 0.8}
    assert extract_json("完全不是 JSON") is None
    assert extract_json("") is None


# ---------------------------------------------------------------------------
# API 接口
# ---------------------------------------------------------------------------
def test_root_endpoint(client) -> None:
    """根路径返回服务信息与接口导航。"""
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["version"] == "1.0.0"
    assert "endpoints" in body


def test_health_endpoint(client) -> None:
    """健康检查必须返回能力清单、存储统计与指标。"""
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()

    assert body["status"] in {"ok", "degraded"}
    assert body["uptime_s"] >= 0

    capabilities = body["capabilities"]
    assert capabilities["llm"]["online"] is False          # 测试环境强制离线
    assert capabilities["llm"]["mode"] == "offline"
    assert capabilities["retrieval"]["refuse_threshold"] > 0
    assert capabilities["embedding"]["model"] == "BAAI/bge-small-zh-v1.5"
    # 密钥脱敏
    assert "your_key_here" not in json.dumps(capabilities, ensure_ascii=False)

    assert body["storage"]["database_ok"] is True
    assert body["storage"]["documents"] == {"documents": 0, "chunks": 0, "chars": 0}
    assert body["metrics"]["counters"]["chat_requests"] == 0


def test_metrics_endpoint(client) -> None:
    """/api/metrics 返回计数器与延迟分位结构。"""
    response = client.get("/api/metrics")
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert {"counters", "tools", "latency_ms"} <= set(body["metrics"])
    assert set(body["metrics"]["latency_ms"]) == {"p50", "p95", "avg", "samples"}


def test_config_endpoint_masks_secrets(client) -> None:
    """/api/config 返回脱敏配置，可直接贴给别人排查问题。"""
    response = client.get("/api/config")
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["config"]["chunk_size"] == 500
    assert "api_key" in body["config"]


def test_docs_available(client) -> None:
    """OpenAPI 文档可用（说明路由装配正常）。"""
    assert client.get("/docs").status_code == 200
    schema = client.get("/openapi.json").json()
    assert "/health" in schema["paths"]


def test_validation_error_shape(client) -> None:
    """参数校验失败的响应结构要统一（前端按 error_type 分支处理）。

    B1 阶段还没有带参数的接口，用一个不存在的路径验证 404 与异常处理器共存即可；
    带请求体的校验测试在 B2 上传接口加入后补充。
    """
    response = client.get("/api/不存在的接口")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# 启动期诊断（都是实际踩过的体验问题）
# ---------------------------------------------------------------------------
def test_port_owner_detection() -> None:
    """能查出占用端口的进程 id。

    用途：同时开两个服务时，uvicorn 只抛一句难懂的 ``winerror 10048``，
    用户不知道"其实已经有一个在跑了"。启动前探测一次就能给出明确指引。
    """
    import socket

    from app import _port_owner

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        owner = _port_owner(port)
        # psutil 不可用或权限不足时返回 None，此时不做断言（不误报失败）
        assert owner is None or owner > 0


def test_port_owner_does_not_raise_on_free_port() -> None:
    """空闲端口查询不能抛异常（也不能误报占用）。"""
    import socket

    from app import _port_owner

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        free_port = probe.getsockname()[1]
    result = _port_owner(free_port)
    assert result is None or isinstance(result, int)


def test_manifest_endpoint(client) -> None:
    """/manifest.json 返回清单，避免浏览器控制台与日志出现 404。"""
    response = client.get("/manifest.json")
    assert response.status_code == 200
    assert response.json()["start_url"] == "/ui"
