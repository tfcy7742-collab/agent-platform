"""B3 测试：问答接口（POST /api/chat）与检索预览接口（GET /api/search）。

覆盖点：
1. 上传文档后能问答并返回来源（离线模式下答案由片段摘要生成，仍带引用）；
2. 知识库为空时返回标准拒答话术；
3. 参数校验（空问题、top_k 越界、threshold 越界）；
4. trace 落库：每次问答都能通过 trace_id 回看检索与拒答细节；
5. 会话消息落库（多轮对话与审计依赖它）；
6. /api/search 只检索不生成。
"""

from __future__ import annotations

import io

import pytest

from core.prompts import REFUSAL_MESSAGE

DOC_TEXT = "\n\n".join(
    [
        "第一条 年假规定：员工入职满一年后享有年假，工作满一年不满十年者每年五天，满十年不满二十年者每年十天。",
        "第二条 病假规定：员工因病需要休息的，凭二级以上医院开具的病假证明申请病假。",
        "第三条 报销标准：一线城市住宿标准为每晚六百元，其他城市为每晚四百元。",
    ]
)


def upload(client, name: str, content: str):
    """上传一份文本文档。"""
    return client.post(
        "/api/documents",
        files=[("files", (name, io.BytesIO(content.encode("utf-8")), "text/plain"))],
        data={"force": "false"},
    )


def ask(client, question: str, **kwargs):
    """调用问答接口。"""
    payload = {"question": question, **kwargs}
    return client.post("/api/chat", json=payload)


# ---------------------------------------------------------------------------
# 基础问答
# ---------------------------------------------------------------------------
def test_chat_with_empty_knowledge_base(client) -> None:
    """知识库为空：返回标准拒答话术，并说明原因。"""
    response = ask(client, "年假有几天")
    assert response.status_code == 200

    body = response.json()
    assert body["ok"] is True
    assert body["refused"] is True
    assert body["answer"] == REFUSAL_MESSAGE
    assert body["refuse_reason"] == "no_documents"
    assert body["sources"] == []
    assert body["retrieved"] == 0
    assert body["trace_id"].startswith("run_")
    assert body["session_id"].startswith("sess_")


def test_chat_after_upload_returns_source(client) -> None:
    """上传文档后问答：能返回答案与来源片段（离线模式下为片段摘要）。"""
    upload(client, "员工手册.txt", DOC_TEXT)

    response = ask(client, "年假有几天", threshold=0.0)
    assert response.status_code == 200
    body = response.json()

    assert body["ok"] is True
    assert body["refused"] is False
    assert body["answer"]
    assert body["sources"], "有资料时答案必须附来源"
    source = body["sources"][0]
    assert source["file_name"] == "员工手册.txt"
    assert source["chunk_id"]
    assert source["text"]
    assert source["score"] > 0

    # 离线模式下会标记 degraded，并说明原因（诚实标注，不伪装成大模型答案）
    assert body["llm"]["degraded"] is True
    assert "离线" in (body["llm"]["error"] or "")
    assert body["usage"]["total_tokens"] == 0


def test_chat_refuses_out_of_domain_question(client) -> None:
    """文档外问题：阈值为 0.5（模拟真实模型的安全阈值）时必须拒答。"""
    upload(client, "员工手册.txt", DOC_TEXT)

    body = ask(client, "今天北京的天气怎么样", threshold=0.5).json()
    assert body["refused"] is True
    assert body["answer"] == REFUSAL_MESSAGE
    assert body["refuse_reason"] == "below_threshold"
    assert body["usage"]["total_tokens"] == 0, "阈值短路不应调用大模型"


def test_chat_returns_retrieval_stats(client) -> None:
    """响应里要带检索统计（前端 trace 面板与调试都依赖它）。"""
    upload(client, "员工手册.txt", DOC_TEXT)

    body = ask(client, "报销标准是多少", threshold=0.0).json()
    retrieval = body["retrieval"]
    assert retrieval["vector_hits"] >= 1
    assert "bm25_hits" in retrieval
    assert retrieval["bm25_enabled"] is True
    assert retrieval["reranker"] if "reranker" in retrieval else True
    assert retrieval["latency_ms"] >= 0
    assert body["top_score"] > 0


# ---------------------------------------------------------------------------
# 参数校验
# ---------------------------------------------------------------------------
def test_chat_rejects_empty_question(client) -> None:
    """空问题 → 422（Pydantic 校验）。"""
    response = client.post("/api/chat", json={"question": ""})
    assert response.status_code == 422


def test_chat_rejects_too_long_question(client) -> None:
    """超长问题 → 422。"""
    response = client.post("/api/chat", json={"question": "问" * 1001})
    assert response.status_code == 422


@pytest.mark.parametrize("payload", [{"question": "年假", "top_k": 0}, {"question": "年假", "top_k": 99}])
def test_chat_rejects_bad_top_k(client, payload: dict) -> None:
    """top_k 越界 → 422。"""
    assert client.post("/api/chat", json=payload).status_code == 422


@pytest.mark.parametrize("threshold", [-0.1, 1.5])
def test_chat_rejects_bad_threshold(client, threshold: float) -> None:
    """threshold 越界 → 422。"""
    response = client.post("/api/chat", json={"question": "年假", "threshold": threshold})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# trace 与会话
# ---------------------------------------------------------------------------
def test_chat_writes_trace(client) -> None:
    """每次问答都会落 trace，包含计划步骤与回答/拒答步骤。"""
    upload(client, "员工手册.txt", DOC_TEXT)
    body = ask(client, "年假有几天", threshold=0.0).json()
    run_id = body["trace_id"]

    trace = client.get(f"/api/traces/{run_id}")
    # B4 才交付 /api/traces，此处允许 404，但要保证 run 已经落库
    if trace.status_code == 200:
        detail = trace.json()
        assert detail["run_id"] == run_id
        assert detail["question"] == "年假有几天"
    else:
        assert trace.status_code == 404

    # 直接查数据库确认落库（不依赖 B4 的接口）
    from infra import db

    stored = db.get_run(run_id)
    assert stored is not None
    assert stored["question"] == "年假有几天"
    assert stored["status"] in {"success", "degraded", "refused"}
    assert stored["steps_detail"], "必须记录分步明细"
    assert any(step["type"] == "plan" for step in stored["steps_detail"])


def test_chat_writes_session_messages(client) -> None:
    """问答会写入会话消息（user + assistant 各一条）。"""
    upload(client, "员工手册.txt", DOC_TEXT)

    session_id = "sess_test_001"
    ask(client, "年假有几天", session_id=session_id, threshold=0.0)
    ask(client, "那病假呢", session_id=session_id, threshold=0.5)

    from infra import db

    history = db.get_history(session_id)
    assert len(history) == 4
    assert [item["role"] for item in history] == ["user", "assistant", "user", "assistant"]
    assert history[0]["content"] == "年假有几天"


def test_chat_accepts_history_flag(client) -> None:
    """use_history=true 时不报错（B4 会用它做问题改写）。"""
    upload(client, "员工手册.txt", DOC_TEXT)
    response = client.post(
        "/api/chat",
        json={"question": "年假有几天", "session_id": "sess_hist", "use_history": True, "threshold": 0.0},
    )
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# 检索预览与拒答话术接口
# ---------------------------------------------------------------------------
def test_search_endpoint(client) -> None:
    """/api/search 只返回检索结果，不生成答案。"""
    upload(client, "员工手册.txt", DOC_TEXT)

    response = client.get("/api/search", params={"q": "年假", "k": 3})
    assert response.status_code == 200
    body = response.json()

    assert body["ok"] is True
    assert body["query"] == "年假"
    assert body["hits"]
    top = body["hits"][0]
    assert top["rank"] == 1
    assert top["file_name"] == "员工手册.txt"
    assert top["score"] > 0
    assert "text" in top
    assert body["stats"]["vector_hits"] >= 1


def test_search_requires_query(client) -> None:
    """/api/search 缺参数 → 422。"""
    assert client.get("/api/search").status_code == 422


def test_refusal_message_endpoint(client) -> None:
    """标准拒答话术接口必须与常量一致（前端与评测脚本都从这里取）。"""
    body = client.get("/api/refusal-message").json()
    assert body["ok"] is True
    assert body["refusal_message"] == REFUSAL_MESSAGE


def test_openapi_lists_chat_endpoints(client) -> None:
    """OpenAPI 文档包含问答相关接口。"""
    schema = client.get("/openapi.json").json()
    assert "/api/chat" in schema["paths"]
    assert "/api/search" in schema["paths"]
    assert "/api/refusal-message" in schema["paths"]
