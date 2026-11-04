"""Prompt 模板。

输出格式要求模型显式标注来源编号，这样界面只展示答案真正引用的块，
而不是把 20 个召回结果全列出来（旧实现的做法，会误导用户）。
"""

from __future__ import annotations

REWRITE_PROMPT = """你是汽车仪表盘技术文档的检索助手。
请把用户问题改写成 2-3 个适合向量检索的子查询。

要求：
1. 保留原问题的核心意图，不要改变含义
2. 用文档里可能出现的专业术语替换口语表达
3. 每个子查询不超过 30 字，彼此差异明显
4. 只输出子查询，每行一个，不要编号、不要解释

用户问题：{question}
"""

ANSWER_PROMPT = """你是汽车仪表盘项目的资深工程师。请只依据下面编号的【参考资料】回答【问题】。

规则：
1. 答案必须 100% 来自参考资料，不得引入外部知识。
2. 如果资料没有直接给出答案，可以做交叉比对和逻辑推导，但要在推理过程中说明依据。
3. 如果资料确实缺失关键条件，直接回答"根据现有文档无法确定"，并指出缺什么。
4. 每个结论后面用 [编号] 标注来源，例如 [1][3]。
5. 不要编造编号，也不要把参考资料里没有的内容写成结论。

输出格式（Markdown）：
## 结论
1-2 句话说清答案。

## 关键依据
- 用 **粗体** 标出关键术语，并附上 [编号]

## 推理过程
1. 分步骤说明如何从资料推导出结论。

## 补充说明
（可选）资料未覆盖的点，或需要人工确认的地方。

【参考资料】
{context}

【问题】
{question}
"""


def build_context(hits) -> str:
    blocks = []
    for index, hit in enumerate(hits, start=1):
        chunk = hit.chunk
        where = chunk.file
        if chunk.page:
            where += f" 第{chunk.page}页"
        if chunk.header:
            where += f" / {chunk.header}"
        blocks.append(f"[{index}] 来源：{where}\n{chunk.text}")
    return "\n\n".join(blocks)
