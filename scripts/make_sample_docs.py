"""样例文档生成脚本。

生成 3 份中文样例文档，覆盖三种解析路径（PDF / DOCX / Markdown），
用于功能演示、端到端验收与 B6 的评测集构建：

===========  ==========================================  ==============
文件名        内容                                         解析路径
===========  ==========================================  ==============
员工手册     考勤/年假/病假/报销/保密/离职等制度条款         PyPDFLoader
产品需求     云笔记 V2.0 的功能范围、验收标准、上线计划       Docx2txtLoader
技术FAQ      部署、备份、故障排查、性能调优问答               TextLoader(.md)
===========  ==========================================  ==============

运行：
    .venv\\Scripts\\python.exe scripts\\make_sample_docs.py
输出目录：``data/sample_docs/``
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = PROJECT_ROOT / "data" / "sample_docs"


# ---------------------------------------------------------------------------
# 内容定义
# ---------------------------------------------------------------------------
HANDBOOK_SECTIONS: list[tuple[str, list[str]]] = [
    (
        "第一章 总则",
        [
            "本手册适用于与公司签订劳动合同的全体员工，自入职之日起生效。",
            "员工入职当日需提交身份证、学历证明、离职证明与体检报告，人事部在三个工作日内完成档案建立。",
            "试用期为三个月，试用期考核在期满前十个工作日由直属主管发起。",
        ],
    ),
    (
        "第二章 工作时间与考勤",
        [
            "公司实行标准工时制，工作时间为每周一至周五 9:00 至 18:00，午休 12:00 至 13:30。",
            "员工需在到岗与离岗时通过企业微信打卡，忘记打卡需在当日提交补卡申请，每月补卡次数不超过三次。",
            "迟到超过三十分钟视为迟到一次，当月迟到累计三次及以上扣发当月全勤奖。",
            "因公外出需提前在系统提交外出申请，由直属主管审批。",
        ],
    ),
    (
        "第三章 休假制度",
        [
            "年假：员工入职满一年后享有年假，工作满一年不满十年者每年五天，满十年不满二十年者每年十天，满二十年者每年十五天。",
            "年假以自然年为计算周期，当年未休完的年假最多结转三天至次年，结转部分须在次年三月三十一日前使用完毕。",
            "病假：员工因病需要休息的，凭二级以上医院开具的病假证明申请病假。病假期间工资按当地最低工资标准的百分之八十发放。",
            "事假：事假为无薪假，全年累计不超过十五天，需提前三个工作日申请。",
            "婚假为十天，产假按国家规定执行，陪产假为十五天，直系亲属丧假为三天。",
            "员工申请三天以内的假期由直属主管审批，三天以上需部门负责人审批，七天以上需人力资源总监审批。",
        ],
    ),
    (
        "第四章 薪酬与报销",
        [
            "公司于每月十五日发放上月工资，遇节假日提前至最近一个工作日发放。",
            "差旅报销标准：一线城市住宿标准为每晚六百元，其他城市为每晚四百元；市内交通凭票报销；餐补为每人每天一百元。",
            "报销单需在费用发生之日起三十日内提交，逾期不予受理。",
            "报销需附合规发票，发票抬头须为公司全称，个人消费发票不得报销。",
        ],
    ),
    (
        "第五章 保密与知识产权",
        [
            "员工在职期间接触到的技术资料、客户名单、经营数据均属公司商业秘密，不得对外泄露。",
            "保密义务在劳动合同解除或终止后继续有效，期限为三年。",
            "员工在职期间完成的与本职工作相关的发明创造、软件著作权归公司所有。",
        ],
    ),
    (
        "第六章 离职管理",
        [
            "员工主动离职需提前三十日以书面形式通知公司，试用期内提前三日通知。",
            "离职交接内容包括工作文档、代码仓库权限、办公设备与门禁卡，交接完成后方可办理离职证明。",
            "离职当月工资在次月发薪日正常发放，未休年假按日工资标准折算补偿。",
        ],
    ),
]

PRD_SECTIONS: list[tuple[str, list[str]]] = [
    (
        "1. 项目背景与目标",
        [
            "云笔记产品 V2.0 的核心目标是提升个人知识管理效率，重点解决 V1.0 中「检索不准、协作缺失、移动端体验差」三个问题。",
            "目标指标：日活提升百分之三十，笔记创建后的七日内留存提升至百分之四十五，检索成功率不低于百分之九十。",
        ],
    ),
    (
        "2. 功能范围",
        [
            "2.1 全文检索增强：支持按标题、正文、标签、创建时间组合筛选，检索结果按相关度排序，支持中文分词与同义词扩展。",
            "2.2 双向链接：在笔记中通过双方括号语法引用其他笔记，被引用笔记自动生成反向链接列表。",
            "2.3 协作空间：支持创建共享空间，成员分为管理员、编辑者、只读三种角色，权限粒度为空间级。",
            "2.4 移动端离线编辑：移动端支持离线创建与修改笔记，恢复网络后自动同步，冲突时以最后修改时间为准并保留冲突副本。",
            "2.5 模板市场：提供会议纪要、周报、读书笔记等模板，用户可一键套用并自定义。",
        ],
    ),
    (
        "3. 非功能需求",
        [
            "检索接口 P95 响应时间不超过三百毫秒，笔记保存接口 P95 不超过两百毫秒。",
            "支持单用户最多十万条笔记，单条笔记最大一兆字节。",
            "系统可用性不低于百分之九十九点九，数据每日全量备份、每小时增量备份。",
        ],
    ),
    (
        "4. 验收标准",
        [
            "检索增强功能：输入关键词后返回结果的前三条中至少包含一条与查询语义相关的笔记。",
            "协作空间：三种角色的权限边界必须通过自动化用例覆盖，越权访问返回 403。",
            "离线编辑：断网状态下可正常编辑，恢复网络后三十秒内完成同步。",
        ],
    ),
    (
        "5. 排期与里程碑",
        [
            "第一阶段（两周）：检索增强与中文分词，交付检索接口与评测集。",
            "第二阶段（三周）：双向链接与协作空间，交付权限模型与前端页面。",
            "第三阶段（两周）：移动端离线编辑与同步冲突处理。",
            "第四阶段（一周）：模板市场与灰度发布，灰度比例为百分之十。",
        ],
    ),
]

FAQ_SECTIONS: list[tuple[str, list[str]]] = [
    (
        "Q1：服务如何部署？",
        [
            "推荐使用 Docker Compose 部署。执行 docker compose up -d 后会启动 api、worker 与 postgres 三个容器。",
            "首次部署需要执行数据库迁移：docker compose exec api python -m app.migrate upgrade head。",
            "配置文件通过环境变量注入，生产环境必须设置 SECRET_KEY 与 DATABASE_URL，禁止使用默认值。",
        ],
    ),
    (
        "Q2：数据如何备份与恢复？",
        [
            "数据库每日凌晨两点执行全量备份，备份文件保留三十天；每小时执行一次增量备份，保留七天。",
            "恢复流程：先停止写入流量，再执行恢复脚本 restore.sh，最后校验行数与关键表校验和。",
            "对象存储中的附件采用跨区域复制，RPO 为十五分钟，RTO 为两小时。",
        ],
    ),
    (
        "Q3：接口返回 502 如何排查？",
        [
            "第一步检查反向代理日志，确认上游服务是否存活：docker compose ps。",
            "第二步查看应用日志中的启动异常，常见原因是数据库连接串错误或迁移未执行。",
            "第三步检查健康检查接口 /health，若返回 degraded 说明依赖（数据库或缓存）不可用。",
        ],
    ),
    (
        "Q4：如何做性能调优？",
        [
            "先定位瓶颈：使用压测工具对核心接口施压，观察 CPU、内存、数据库连接池与慢查询日志。",
            "常见优化顺序为：加索引、减少 N+1 查询、引入缓存、读写分离、最后才考虑分库分表。",
            "缓存策略建议采用 Cache-Aside，热点数据过期时间设置随机抖动，避免缓存雪崩。",
        ],
    ),
    (
        "Q5：日志与监控怎么接？",
        [
            "应用以 JSON 格式输出日志到标准输出，由日志采集器统一收集，禁止把日志写入容器内文件。",
            "关键指标包括请求量、错误率、P95 延迟、数据库连接数、队列积压量，均需配置告警阈值。",
            "每条请求需携带 trace_id，便于跨服务串联排查。",
        ],
    ),
]


# ---------------------------------------------------------------------------
# 生成器：Markdown
# ---------------------------------------------------------------------------
def write_markdown(path: Path, title: str, sections: list[tuple[str, list[str]]]) -> None:
    """把分节内容写成 Markdown 文件。"""
    lines = [f"# {title}", ""]
    for heading, paragraphs in sections:
        lines.append(f"## {heading}")
        lines.append("")
        for paragraph in paragraphs:
            lines.append(f"- {paragraph}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# 生成器：DOCX
# ---------------------------------------------------------------------------
def write_docx(path: Path, title: str, sections: list[tuple[str, list[str]]]) -> None:
    """用 python-docx 生成 DOCX。"""
    from docx import Document

    document = Document()
    document.add_heading(title, level=0)
    for heading, paragraphs in sections:
        document.add_heading(heading, level=1)
        for paragraph in paragraphs:
            document.add_paragraph(paragraph)
    document.save(str(path))


# ---------------------------------------------------------------------------
# 生成器：PDF（纯中文字体嵌入，无需额外依赖）
# ---------------------------------------------------------------------------
def _find_chinese_font() -> Path | None:
    """寻找可用于嵌入的系统中文 TrueType 字体。"""
    candidates = [
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\msyh.ttf"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\simsun.ttc"),
        Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/System/Library/Fonts/PingFang.ttc"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def write_pdf(path: Path, title: str, sections: list[tuple[str, list[str]]]) -> bool:
    """生成 PDF：优先用 reportlab 嵌入中文字体，失败则返回 False 由调用方降级。

    Returns:
        是否成功生成 PDF。
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
    except ImportError:
        return False

    font_path = _find_chinese_font()
    if font_path is None:
        return False

    font_name = "ChineseFont"
    try:
        # .ttc 是字体集合，reportlab 需要指定子字体索引
        if font_path.suffix.lower() == ".ttc":
            pdfmetrics.registerFont(TTFont(font_name, str(font_path), subfontIndex=0))
        else:
            pdfmetrics.registerFont(TTFont(font_name, str(font_path)))
    except Exception:  # noqa: BLE001 - 字体异常时交给调用方降级
        return False

    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title=title,
    )
    title_style = ParagraphStyle("TitleCN", fontName=font_name, fontSize=18, leading=26, spaceAfter=10)
    heading_style = ParagraphStyle("HeadingCN", fontName=font_name, fontSize=13, leading=20, spaceBefore=10, spaceAfter=6)
    body_style = ParagraphStyle("BodyCN", fontName=font_name, fontSize=10.5, leading=18)

    flow = [Paragraph(title, title_style), Spacer(1, 6)]
    for heading, paragraphs in sections:
        flow.append(Paragraph(heading, heading_style))
        for paragraph in paragraphs:
            flow.append(Paragraph(f"• {paragraph}", body_style))
        flow.append(Spacer(1, 4))
    doc.build(flow)
    return True


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> int:
    """生成所有样例文档。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    created: list[str] = []

    # 1) 员工手册 —— PDF（验收演示主文档）
    handbook_pdf = OUTPUT_DIR / "员工手册.pdf"
    pdf_ok = write_pdf(handbook_pdf, "员工手册（2025 版）", HANDBOOK_SECTIONS)
    if pdf_ok:
        created.append(f"{handbook_pdf.name}（PDF，{handbook_pdf.stat().st_size // 1024} KB）")
    else:
        # PDF 依赖缺失时降级：同时产出 md 版本，保证样例数据依然可用
        fallback = OUTPUT_DIR / "员工手册.md"
        write_markdown(fallback, "员工手册（2025 版）", HANDBOOK_SECTIONS)
        created.append(f"{fallback.name}（Markdown 降级，PDF 需要 reportlab + 系统中文 TTF 字体）")

    # 2) 产品需求文档 —— DOCX
    prd_docx = OUTPUT_DIR / "云笔记产品需求文档.docx"
    write_docx(prd_docx, "云笔记 V2.0 产品需求文档", PRD_SECTIONS)
    created.append(f"{prd_docx.name}（DOCX，{prd_docx.stat().st_size // 1024} KB）")

    # 3) 技术 FAQ —— Markdown
    faq_md = OUTPUT_DIR / "运维技术FAQ.md"
    write_markdown(faq_md, "运维技术 FAQ", FAQ_SECTIONS)
    created.append(f"{faq_md.name}（Markdown，{faq_md.stat().st_size // 1024} KB）")

    print("=" * 74)
    print(f"样例文档已生成到：{OUTPUT_DIR}")
    for item in created:
        print(f"  · {item}")
    print("=" * 74)
    print("下一步：启动服务后把该目录下的文件上传，或执行")
    print("  curl -X POST http://127.0.0.1:8000/api/documents -F \"files=@data/sample_docs/员工手册.pdf\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
