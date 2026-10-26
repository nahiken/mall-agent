"""CLI：构建 RAG 知识库集合（cs_kb / product_kb / policy_kb）。

流程：
1. 从 mall.sql / 预设文案生成三类知识源文档（写入 data/kb_src/）。
2. 调用 RAG-MCP-SERVER 的 ingest 命令为每个集合摄取文档。

v3 改动（改动一）：
- cs 知识补：联系渠道(400-888-6666/在线客服)、取消订单、确认收货；统一退款时效口径(3-5工作日)。
- policy 知识补：支付/发票/价保对齐 PQ 主题原文（供检索门禁与答案判答）。
- 新增 --embedding-provider 直通 RAG-MCP ingest（fake=离线确定性，默认 fake 供离线评测）。
- 打印生成的文档清单。

用法：
    python -m mall_agent.cli.ingest_kb            # 生成源文档 + 摄取全部集合(默认 fake)
    python -m mall_agent.cli.ingest_kb --list     # 仅列出集合（RAG-MCP）
    python -m mall_agent.cli.ingest_kb --gen-only # 仅生成源文档不摄取
"""
from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

from mall_agent.core.settings import settings

KB_SRC = Path(settings.get("data_dir", "data")) / "kb_src"

# 统一退款时效口径（售后政策与退款政策一致），供两处引用避免漂移
REFUND_TIMELINE = "退款原路返回，审核通过后一般3-5个工作日到账。"


def _gen_cs_knowledge() -> list[tuple[str, str]]:
    """生成客服知识文档（售后/政策/会员）。返回 [(文件名, 内容)]。

    v3：新增联系渠道/取消订单/确认收货；退款时效统一 3-5 工作日；补充 400 电话。
    v4（T9）：价格保护并入 cs（价保属售后/交易主集，与售后同集同搜）。
    """
    docs = [
        ("售后政策.txt",
         "本店支持7天无理由退货。收货后7天内商品未使用、包装完好可申请退货。"
         "退货需在订单中心提交申请，审核通过后处理退款。"
         "食品、定制商品不支持无理由退货。\n" + REFUND_TIMELINE),
        ("退款政策.txt",
         REFUND_TIMELINE + "使用优惠券支付的订单，退款时优惠券按比例返还。"
         "货到付款订单退款需提供银行账户信息。"),
        ("配送与物流.txt",
         "默认快递发货，全国包邮（偏远地区除外）。下单后48小时内发货，"
         "发货后可在订单详情查看物流轨迹。预售商品以页面标注时间为准。"),
        ("会员权益.txt",
         "注册即成为普通会员。购物可积累积分，积分可抵扣现金。"
         "生日月消费双倍积分。会员可参与专属优惠活动。"),
        ("优惠券规则.txt",
         "优惠券分为满减券和折扣券。满减券需满足最低消费金额才可使用。"
         "优惠券领取后30天内有效，过期作废。每笔订单限用一张优惠券。"),
        ("联系渠道.txt",
         "客服热线：400-888-6666；也可通过商城在线客服联系。"
         "工作时间9:00-21:00，售后问题可通过热线或在线客服提交。"),
        ("取消订单.txt",
         "未发货订单可直接取消，取消后款项原路退回。"
         "已发货订单不支持取消，需确认收货后走退货流程。"
         "取消订单可在订单详情页操作。\n" + REFUND_TIMELINE),
        ("确认收货.txt",
         "订单显示已发货/已送达后可确认收货：在订单详情页点击确认收货即可。"
         "若商品有问题，建议先联系售后（400-888-6666 或在线客服）再确认收货，"
         "避免影响后续维权。"),
        ("价格保护.txt",   # v4（T9）：从 policy 并入 cs，与售后同集同搜
         "商品在签收后7天内降价的，可申请价格保护，返还差价至账户余额。"),
    ]
    return docs


def _gen_product_knowledge() -> list[tuple[str, str]]:
    """从 mall.sql 生成商品知识文档。返回 [(文件名, 内容)]。"""
    from mall_agent.mall_api.mock_backend import MockMallBackend

    backend = MockMallBackend(settings.mall.get("sql_path"))
    # 同步加载（这里用事件循环跑一次）
    try:
        asyncio.run(backend.load())
    except RuntimeError:
        pass
    products = backend._tables.get("pms_product", [])
    docs = []
    for p in products[:80]:
        name = p.get("name", "商品")
        price = p.get("price", 0)
        subtitle = p.get("sub_title", "")
        cat = p.get("product_category_name", "")
        content = f"商品：{name}。价格：{price}元。分类：{cat}。简介：{subtitle}"
        docs.append((f"{p.get('id')}_{name}.txt", content))
    return docs


def _gen_policy_knowledge() -> list[tuple[str, str]]:
    """生成客服政策知识文档（支付/发票，与 PQ 主题原文对齐）。

    v4（T9 决策）：价格保护文档从 policy 并入 cs（价保属售后/交易主集，与售后
    同集同搜，消除跨 hub 分裂）；policy 仅保留支付/发票两篇（政策性、非售后）。
    """
    docs = [
        ("支付方式.txt",
         "支持支付宝、微信支付、银行卡支付。下单后请在30分钟内完成支付，超时订单自动关闭。"),
        ("发票说明.txt",
         "支持开具电子发票，可在订单详情页申请。发票抬头需与会员实名信息一致。"),
    ]
    return docs


def generate_sources() -> dict[str, list[str]]:
    """生成三类知识源文档，返回 {hub: [文档名]}。"""
    KB_SRC.mkdir(parents=True, exist_ok=True)
    names: dict[str, list[str]] = {}
    for hub, gen in [("cs", _gen_cs_knowledge), ("guide", _gen_product_knowledge), ("policy", _gen_policy_knowledge)]:
        dir_ = KB_SRC / hub
        dir_.mkdir(parents=True, exist_ok=True)
        files: list[str] = []
        for fname, content in gen():
            (dir_ / fname).write_text(content, encoding="utf-8")
            files.append(fname)
        names[hub] = files
        print(f"  生成 {hub}: {len(files)} 个文档：{', '.join(files)}")
    return names


def ingest_hub(hub: str, embedding_provider: str | None = None) -> None:
    """调用 RAG-MCP ingest 摄取一个集合。

    embedding_provider：None=读 RAG-MCP 配置（默认 ollama）；'fake'=确定性离线向量。
    """
    rag = settings.rag
    python = settings.env("RAG_MCP_PYTHON") or rag.get("python")
    if not python:
        print("  ⚠️ RAG_MCP_PYTHON 未配置，跳过 ingest")
        return
    hub_cfg = settings.rag.get("hubs", {}).get(hub, {})
    collection = hub_cfg.get("collection", f"{hub}_kb")
    # RAG-MCP 在它自己的 cwd(rag.cwd) 下解析相对路径 → 统一转绝对路径，避免路径解析歧义
    rag_cwd = Path(rag.get("cwd") or ".")
    src = (KB_SRC / hub).resolve()
    index_dir = (rag_cwd / hub_cfg.get("index_dir", "data/index")).resolve()
    cmd = [python, "-m", "rag_mcp_server", "ingest", str(src),
           "--collection", collection, "--index-dir", str(index_dir)]
    if embedding_provider:
        cmd += ["--embedding-provider", embedding_provider]
    print(f"  ingest {hub} -> {collection}（embedding={embedding_provider or 'config'}）...")
    import os
    result = subprocess.run(
        cmd,
        cwd=rag.get("cwd"),
        env={"PYTHONPATH": rag.get("src"), "PYTHONIOENCODING": "utf-8", **{k: v for k, v in os.environ.items()}},
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    print(result.stdout[-800:] if result.stdout else "")
    if result.stderr:
        print("  stderr:", result.stderr[-400:])
    if result.returncode != 0:
        print(f"  ⚠️ ingest {hub} 返回码 {result.returncode}")


def main() -> None:
    args = sys.argv[1:]
    if "--list" in args:
        from mall_agent.rag.mcp_client import get_rag_client
        asyncio.run(get_rag_client().list_collections())
        return
    # 提取 --embedding-provider <val>（若给出）；缺省用 fake（离线评测默认）
    provider = "fake"
    if "--embedding-provider" in args:
        i = args.index("--embedding-provider")
        provider = args[i + 1] if i + 1 < len(args) else "fake"
    print("生成知识源文档...")
    generate_sources()
    if "--gen-only" in args:
        print("✅ 仅生成知识源（未摄取）；目录:", KB_SRC)
        return
    print("摄取知识库集合...")
    for hub in ("cs", "guide", "policy"):
        ingest_hub(hub, provider)
    print("✅ 知识库构建完成")


if __name__ == "__main__":
    main()
