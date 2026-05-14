from __future__ import annotations

from typing import Any


def _section(title: str, content: str) -> str:
    text = content.strip() if content.strip() else "（无）"
    return f"【{title}】\n{text}\n"


def _list_block(items: list[str] | None, empty_text: str = "（无）") -> str:
    normalized = [item.strip() for item in (items or []) if item and item.strip()]
    if not normalized:
        return empty_text
    return "\n".join(f"- {item}" for item in normalized)


def _step_record_block(step_record: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"步骤文本：{step_record.get('step_text', '')}",
            f"步骤意图：{step_record.get('step_intent', '')}",
            f"步骤状态：{step_record.get('step_status', '')}",
            f"判定理由：{step_record.get('judge_reason', '')}",
            "该步之前的学生数学事实：",
            _list_block(step_record.get("student_state_before", [])),
            "该步之前的正确数学事实：",
            _list_block(step_record.get("correct_state_before", [])),
        ]
    )


def build_parse_problem_prompt(question_text: str, student_answer_text: str) -> tuple[str, str]:
    system_prompt = (
        "你的任务是对数学解答题的题干原文与学生作答原文进行结构化拆分与对齐。"
        "提取时必须逐字保留所有的原始文本字符，绝不进行任何形式的删改、省略或概括。"
        "将题干与作答分别切分为独立的片段，剥离公共题干与独立子问题，并确保每一问的题干片段与其对应的作答片段形成严格的一一映射关系。"
    )
    user_prompt = (
        "根据输入内容输出纯 JSON 对象，不要输出任何额外的文本。输出的 JSON 结构与字段要求如下：\n"
        "{\n"
        "  \"shared_stem_text\": <str，多问题目共享的公共题干；若是单问题目，填入空字符串>,\n"
        "  \"subproblems\": <list[str]，按顺序存放子问题题干。若是多问，此处仅保留各小问自身的问法，不得重复公共题干；若是单问，必须返回包含完整题干的单元素数组>,\n"
        "  \"student_subanswers\": <list[str]，按顺序存放拆分后的作答片段。数组长度必须与 subproblems 绝对一致；若某问漏答，对应位置必须填入空字符串>,\n"
        "  \"is_multi_part\": <bool，表示题目是否包含多个子问题>\n"
        "}\n\n"
        "若学生作答连续且无显式序号，需依据题干的提问顺序推测并切分作答片段。\n\n"
        "例如当题干为“已知函数 f(x)=x^2-4。（1）求其零点。（2）求其极值。”，作答为“x=±2。所以极值为-4。”时，输出为：\n"
        "{\n"
        "  \"shared_stem_text\": \"已知函数 f(x)=x^2-4。\",\n"
        "  \"subproblems\": [\"（1）求其零点。\", \"（2）求其极值。\"],\n"
        "  \"student_subanswers\": [\"x=±2。\", \"所以极值为-4。\"],\n"
        "  \"is_multi_part\": true\n"
        "}\n\n"
        "输入数据：\n"
        f"{_section('题干原文', question_text)}\n"
        f"{_section('学生作答原文', student_answer_text)}\n"
        "请基于以上输入，直接输出 JSON 结果："
    )
    return system_prompt, user_prompt


def build_problem_review_prompt(
    review_source_text: str,
    subproblems: list[str],
    is_multi_part: bool,
) -> tuple[str, str]:
    system_prompt = (
        "你的任务是从数学解答题的题干中提取供后续推理使用的已知条件。"
        "过滤背景故事与说明性文字，仅保留数学条件。"
        "请勿进行任何计算或推导。"
    )
    if is_multi_part:
        system_prompt += "提取的条件必须对整道题目全局有效，请严格排除仅属于特定子问题的前提假设。"
        user_prompt = (
            "根据输入内容输出纯 JSON 对象，不要输出任何额外的文本。输出的 JSON 结构与字段要求如下：\n"
            "{\n"
            "  \"shared_problem_context\": <list[str]，存放提取出的全局已知条件。每一项作为独立的短句，不保留冗余的解释文本>\n"
            "}\n\n"
            "子问题列表仅用于辅助辨识局部边界，禁止从中提取任何内容。所有提取的条件均只能来源于全局题干。\n\n"
            "例如当全局题干为“已知等差数列 {a_n} 的前 n 项和为 S_n，且 a_3=5，S_5=15。”时，输出必须为：\n"
            "{\n"
            "  \"shared_problem_context\": [\"{a_n} 是等差数列\", \"{a_n} 的前 n 项和为 S_n\", \"a_3=5\", \"S_5=15\"]\n"
            "}\n\n"
            "输入数据：\n"
            f"{_section('全局题干', review_source_text)}\n"
            f"{_section('子问题列表', _list_block(subproblems))}\n"
            "请基于以上输入，直接输出 JSON 结果："
        )
    else:
        user_prompt = (
            "根据输入内容输出纯 JSON 对象，不要输出任何额外的文本。输出的 JSON 结构与字段要求如下：\n"
            "{\n"
            "  \"shared_problem_context\": <list[str]，存放提取出的已知条件。每一项作为独立的短句，不保留冗余的解释文本>\n"
            "}\n\n"
            "所有提取的条件均只能来源于全局题干。\n\n"
            "例如当题干原文为“已知椭圆 C: x^2/a^2 + y^2/b^2 = 1 (a>b>0) 的离心率为 1/2，且过点 (2, 0)。求椭圆方程。”时，输出必须为：\n"
            "{\n"
            "  \"shared_problem_context\": [\"椭圆 C: x^2/a^2 + y^2/b^2 = 1 (a>b>0)\", \"椭圆 C 的离心率为 1/2\", \"椭圆 C 过点 (2, 0)\"]\n"
            "}\n\n"
            "输入数据：\n"
            f"{_section('题干原文', review_source_text)}\n"
            "请基于以上输入，直接输出 JSON 结果："
        )
    return system_prompt, user_prompt


def build_step_split_prompt(
    shared_stem_text: str,
    subproblem_text: str,
    student_subanswer: str,
) -> tuple[str, str]:
    system_prompt = (
        "你的任务是将学生的作答文本切分为可独立验证的步骤序列。"
        "切分时必须逐字保留所有原始文本字符，绝不进行任何形式的删改、省略或概括。"
        "解释性文字应与其对应的数学公式保留在同一步中。保持合适的切分粒度，既不将作答过程切分得过于细碎，且不将多个处理过程整合到同一步中。"
    )
    if shared_stem_text.strip():
        user_prompt = (
            "根据输入内容输出纯 JSON 对象，不要输出任何额外的文本。输出的 JSON 结构与字段要求如下：\n"
            "{\n"
            "  \"student_steps\": <list[str]，按顺序存放切分后的作答片段。每个片段代表完整的一步数学推导>\n"
            "}\n\n"
            "题干信息仅作为上下文参考，只能对学生作答进行步骤切分。\n\n"
            "例如当作答原文为“由 x^2-4=0 解得 x=2 或 x=-2。因为题干要求 x>0，所以 x=2。”时，输出必须为：\n"
            "{\n"
            "  \"student_steps\": [\"由 x^2-4=0 解得 x=2 或 x=-2。\", \"因为题干要求 x>0，所以 x=2。\"]\n"
            "}\n\n"
            "输入数据：\n"
            f"{_section('公共题干原文', shared_stem_text)}\n"
            f"{_section('当前问题题干', subproblem_text)}\n"
            f"{_section('学生作答原文', student_subanswer)}\n"
            "请基于以上输入，直接输出 JSON 结果："
        )
    else:
        user_prompt = (
            "根据输入内容输出纯 JSON 对象，不要输出任何额外的文本。输出的 JSON 结构与字段要求如下：\n"
            "{\n"
            "  \"student_steps\": <list[str]，按顺序存放切分后的作答片段。每个片段代表完整的一步数学推导>\n"
            "}\n\n"
            "题干信息仅作为上下文参考，只能对学生作答进行步骤切分。\n\n"
            "例如当作答原文为“由 x^2-4=0 解得 x=2 或 x=-2。因为题干要求 x>0，所以 x=2。”时，输出必须为：\n"
            "{\n"
            "  \"student_steps\": [\"由 x^2-4=0 解得 x=2 或 x=-2。\", \"因为题干要求 x>0，所以 x=2。\"]\n"
            "}\n\n"
            "输入数据：\n"
            f"{_section('当前问题题干', subproblem_text)}\n"
            f"{_section('学生作答原文', student_subanswer)}\n"
            "请基于以上输入，直接输出 JSON 结果："
        )
    return system_prompt, user_prompt


def build_step_analysis_prompt(
    shared_stem_text: str,
    shared_problem_context: list[str],
    current_subproblem_text: str,
    current_step_text: str,
) -> tuple[str, str]:
    system_prompt = (
        "你是一名严谨客观的数学专家。"
        "你的任务是识别学生在当前步骤中的解题意图，并视需要提供数学工具的执行参数，提供结构化的数据。"
        "通常只有在涉及到数学计算的步骤时才需调用数学工具。"
    )
    common_blocks = ""
    if shared_stem_text.strip():
        common_blocks += f"{_section('公共题干原文', shared_stem_text)}\n"

    user_prompt = (
        "请根据输入内容进行分析，直接输出纯 JSON 对象，不要输出任何额外的文字。JSON 结构与字段要求如下：\n"
        "{\n"
        "  \"step_intent\": <str，简要陈述当前步骤的解题意图，如“对二次三项式因式分解”或“根据边相等推导角相等”>,\n"
        "  \"tool_needed\": <bool，表示是否需要调用数学工具>,\n"
        "  \"math_tool_request\": <dict | null，若 tool_needed 为 false，返回 null；若为 true，必须提供包含 \"mode\" 和 \"payload\" 两个键的字典，详见下方接口规范>\n"
        "}\n\n"
        "【工具接口规范】\n"
        "- check_equivalence（检查两式是否等价） -> 构造：{\"mode\": \"check_equivalence\", \"payload\": {\"expr1\": \"表达式1\", \"expr2\": \"表达式2\"}}\n"
        "- solve（求解单变量方程） -> 构造：{\"mode\": \"solve\", \"payload\": {\"equation\": \"完整方程\", \"variable\": \"目标变量\"}}\n"
        "- simplify（化简数学表达式） -> 构造：{\"mode\": \"simplify\", \"payload\": {\"expression\": \"待化简表达式\"}}\n"
        "- substitute（代入具体数值求值） -> 构造：{\"mode\": \"substitute\", \"payload\": {\"expression\": \"原式\", \"substitutions\": {\"变量名\": 数值}}}\n"
        "- numeric_check（对表达式进行数值近似检查） -> 构造：{\"mode\": \"numeric_check\", \"payload\": {\"expression\": \"待检查表达式\"}}\n\n"
        "【输出参考示例】\n"
        "示例 1：若步骤为“x^2-5x+6=(x-2)(x-3)”，输出：\n"
        "{\"step_intent\": \"对二次三项式因式分解\", \"tool_needed\": true, \"math_tool_request\": {\"mode\": \"check_equivalence\", \"payload\": {\"expr1\": \"x^2-5x+6\", \"expr2\": \"(x-2)(x-3)\"}}}\n\n"
        "示例 2：若步骤为“所以顶点为(2,-1)”，输出：\n"
        "{\"step_intent\": \"基于标准式读出顶点坐标结论\", \"tool_needed\": false, \"math_tool_request\": null}\n\n"
        "输入数据：\n"
        f"{common_blocks}"
        f"{_section('题干已知信息', _list_block(shared_problem_context))}\n"
        f"{_section('当前题目', current_subproblem_text)}\n"
        f"{_section('当前步骤', current_step_text)}\n"
        "请直接输出 JSON 结果："
    )
    return system_prompt, user_prompt


def build_step_judge_prompt(
    shared_stem_text: str,
    shared_problem_context: list[str],
    current_subproblem_text: str,
    current_step_text: str,
    step_intent: str,
    history_has_error: bool,
    current_student_state: list[str],
    current_correct_state: list[str],
    math_tool_result: dict[str, Any] | None,
) -> tuple[str, str]:
    system_prompt = (
        "你是一名客观严谨的数学专家。"
        "你的任务是基于客观的数学规律与提供的前置解题状态，输出当前步骤获取的数学事实，并对学生当前的解题步骤进行批改。"
    )
    math_tool_result_text = "未调用数学工具" if not math_tool_result else str(math_tool_result)
    common_blocks = ""
    if shared_stem_text.strip():
        common_blocks += f"{_section('公共题干原文', shared_stem_text)}\n"
    user_prompt = (
        "请根据输入数据对学生当前步骤进行分析，并输出纯 JSON 对象。\n\n"
        "【输入状态说明】\n"
        "- 学生当前已得到的数学事实：学生推导出的结论集合。若“截至当前是否已发生错误”为“是”，则此集合已包含错误的数学前提。\n"
        "- 当前正确的数学事实：截至当前步骤发生前，客观正确的数学结果（此为验证当前步骤正确性的起点，并非本题最终答案）。\n\n"
        "【执行验证流程】\n"
        "步骤 1（事实提取）：审视学生当前步骤，提取其得出的所有阶段性结论。无论该结论客观上是否正确，必须如实提取。\n"
        "步骤 2（基准测试）：以“当前正确的数学事实”为前提推演当前步骤。\n"
        "如果推演合法且结果与学生一致，则 step_status 为 correct。\n"
        "如果结果不一致，进入步骤 3。\n"
        "步骤 3（前提代入与分支判定）：将“学生当前已得到的数学事实”视为前提，进行当前步骤的推导。\n"
        "若能得出与学生一致的结果，说明符合数学规律，但因使用了错误前提导致结果偏差，step_status 为 inherited_error。\n"
        "若结果与学生依然不符，说明当前步骤的动作本身存在错误，step_status 为 autonomous_error。\n\n"
        "【输出 JSON 字段要求】\n"
        "{\n"
        "  \"student_facts_update\": <list[str]，提取当前步骤中学生得出的结论。只要是推进了解题进度的客观论断，无论其本身对错，都必须如实提取。请排除解释性纯文本；若无实质新增请返回 []>,\n"
        "  \"judge_reason\": <str，判定依据：请依据【执行验证流程】，使用精简的书面化语言记录你的推演过程。>,\n"
        "  \"step_status\": <str，状态结果：基于上述推理得出此步骤是否正确，仅限填写 \"correct\"、\"autonomous_error\" 或 \"inherited_error\"> \n"
        "}\n\n"
        "输入数据：\n"
        f"{common_blocks}"
        f"{_section('题干已知信息', _list_block(shared_problem_context))}\n"
        f"{_section('当前题目', current_subproblem_text)}\n"
        f"{_section('截至当前是否已发生错误', '是' if history_has_error else '否')}\n"
        f"{_section('学生当前已得到的数学事实', _list_block(current_student_state))}\n"
        f"{_section('当前正确的数学事实', _list_block(current_correct_state))}\n"
        f"{_section('当前步骤意图', step_intent or '（未给出）')}\n"
        f"{_section('当前步骤', current_step_text)}\n"
        f"{_section('数学工具返回结果', math_tool_result_text)}\n"
        "请直接输出 JSON 结果："
    )
    return system_prompt, user_prompt


def build_completion_prompt(
    shared_stem_text: str,
    shared_problem_context: list[str],
    current_subproblem_text: str,
    current_student_steps: list[str],
    current_student_state: list[str],
) -> tuple[str, str]:
    system_prompt = (
        "你负责执行数学解答完整性的逻辑校验任务。"
        "需通过比对试题的最终求解目标与学生的实际推导状态，客观判定当前解答是否已形成逻辑闭合或得出最终结论。"
        "输出语言需保持绝对的学术严谨与客观。"
    )
    common_blocks = ""
    if shared_stem_text.strip():
        common_blocks += f"{_section('公共题干原文', shared_stem_text)}\n"
    user_prompt = (
        "请依据输入数据执行完整性校验，并输出标准 JSON 格式，不要输出任何额外的文本。\n\n"
        "【输入状态定义】\n"
        "- 题干已知信息：全局的已知条件与数学前提。\n"
        "- 当前题目：界定了本问的最终核心答题要求（如：特定的数值解、取值范围、函数解析式或几何证明终点）。\n"
        "- 当前问题的学生原始作答过程：学生为解答本问写出的推导文本。\n"
        "- 学生当前已得到的数学事实：从学生作答中提取出的实质性数学结论集合。\n\n"
        "【校验执行序列】\n"
        "步骤 1：分析“当前题目”，理解其最终的求解目标或证明要求。\n"
        "步骤 2：通过学生的“原始作答过程”与“已得到的数学事实”，核查其是否已完成上述最终目标。\n"
        "步骤 3：\n"
        "若学生推导中已包含最终定论、得出所求结果或实现证明逻辑闭合，判定为已完成。\n"
        "若推导仅停留在中间过渡阶段，缺少最终结论语句或目标数值，判定为未完成。\n\n"
        "【输出 JSON 字段规范】\n"
        "{\n"
        "  \"issue_reason\": <str，缺失环节摘要。若判定为未完成，精简且精确地指出学生当前作答与最终目标之间的缺失；若判定为已完成，返回空字符串 \"\">,\n"
        "  \"is_complete\": <bool，基于分析过程得出的最终校验状态：true 或 false>\n"
        "}\n\n"
        "输入数据：\n"
        f"{common_blocks}"
        f"{_section('题干已知信息', _list_block(shared_problem_context))}\n"
        f"{_section('当前题目', current_subproblem_text)}\n"
        f"{_section('当前问题的学生原始作答过程', _list_block(current_student_steps))}\n"
        f"{_section('学生当前已得到的数学事实', _list_block(current_student_state))}\n"
        "请直接输出 JSON 结果："
    )
    return system_prompt, user_prompt


def build_correct_solution_prompt(
    shared_stem_text: str,
    shared_problem_context: list[str],
    current_subproblem_text: str,
    current_correct_state: list[str],
) -> tuple[str, str]:
    system_prompt = (
        "你是一名客观严谨的数学专家。"
        "你的任务是基于当前已确立的客观数学基准，展开严密的逻辑推演，直至得出当前问题的最终答案，完成剩余的全部解答过程。"
        "输出文本必须保持绝对的严谨性，严格采用标准教科书解析风格，杜绝任何口语化表述。"
    )
    common_blocks = ""
    if shared_stem_text.strip():
        common_blocks += f"{_section('公共题干原文', shared_stem_text)}\n"
    user_prompt = (
        "请依据输入数据，在当前正确的数学事实的基础之上继续展开严密的数学推导，直至该问题完全解答完毕，并输出标准 JSON 格式，不要生成任何额外文本。\n\n"
        "【输入状态定义】\n"
        "- 公共题干原文 / 题干已知信息：全局的已知条件与数学前提。\n"
        "- 当前题目：界定了本问的最终求解目标。\n"
        "- 当前正确的数学事实：推导的客观起点。此为截至目前已验证无误的数学结论，你要在此状态继续向下推演。\n\n"
        "【推演执行序列】\n"
        "步骤 1（按步书写解析）：从“当前正确的数学事实”接着向下推导，直到得出最终答案。请将推导过程按合理的解题节奏拆分成列表元素。列表的每一项就是解析里的一小步（包含简单的解释性文字和数学推理，例如：“由题意可得 x=3” 或 “代入化简得 y=9”）。要求：一步一项，不要把多步复杂的计算全挤在一个元素里。\n"
        "输出表述规范：必须采用标准、客观的数学书面解析体例。请勿使用第一人称、主观教导性话语或任何口语化词汇。\n"
        "步骤 2（数学事实提纯）：与步骤 1 生成的序列严格对齐，逐项剥离其中的解释性与过渡性文本，仅提取切实推进解题进度的“实质性数学事实”（如：方程式、不等式解集、等价变形结果、几何断言等）。\n\n"
        "【输出 JSON 字段规范】\n"
        "{\n"
        "  \"correct_solution_process\": <list[str]，执行步骤 1 的解答序列。每一项对应一个结构完整的推导动作，整体需串联成一篇严谨、标准的书面解析>,\n"
        "  \"correct_solution_steps\": <list[str]，执行步骤 2 的提纯事实序列。按推演顺序排列的纯粹客观数学结论。列表第一项是继“当前正确的数学事实”之后的新增事实。务必保持精简，严格剔除解释性文字>\n"
        "}\n\n"
        "输入数据：\n"
        f"{common_blocks}"
        f"{_section('题干已知信息', _list_block(shared_problem_context))}\n"
        f"{_section('当前题目', current_subproblem_text)}\n"
        f"{_section('当前正确的数学事实', _list_block(current_correct_state))}\n"
        "请直接输出 JSON 结果："
    )
    return system_prompt, user_prompt


def build_correct_alignment_prompt(
    shared_stem_text: str,
    current_subproblem_text: str,
    current_step_text: str,
    current_step_intent: str,
    next_step_text: str | None,
    current_correct_state: list[str],
    remaining_correct_solution_steps: list[str],
) -> tuple[str, str]:
    system_prompt = (
        "你的任务是根据当前解题进度，从剩余正确事实序列中确定应当在此步骤确认的事实。"
        "请仅输出符合要求的 JSON 结构。"
    )
    common_blocks = ""
    if shared_stem_text.strip():
        common_blocks += f"{_section('公共题干原文', shared_stem_text)}\n"
    user_prompt = (
        "请根据以下信息判断，在当前步骤的解题进度下，剩余正确事实序列中的前若干条事实是否应当被确认。\n\n"
        "【任务说明】\n"
        "正确解题过程的推进应与学生的步骤保持同步。当前正在处理的是解题过程中的某一个步骤，"
        "你需要依据正常的解题逻辑，从“剩余正确事实序列”的开头开始，依次判断每一条事实是否属于当前步骤应当完成的内容。"
        "如果属于，则将该事实纳入本次应当确认的事实列表；如果不属于，则停止判断，后续事实不在本步确认。\n\n"
        "【判断依据】\n"
        "1. 确认必须从剩余正确事实序列的开头连续进行，按顺序选取。\n"
        "2. 判断一条事实是否应当在本步确认，取决于该事实在正常解题逻辑中是否与当前步骤所处的阶段对应。\n"
        "3. “下一步内容”字段展示了学生下一步的作答内容，可用于界定当前步骤的边界：如果某事实明显属于下一步才会得出的内容，则不应在本步确认。\n"
        "4. 输出的确认事实必须原样取自“剩余正确事实序列”中的条目，不得修改表述。\n"
        "5. 若当前步骤所处阶段尚未对应任何新的事实，则返回空列表。\n\n"
        "【输出格式】\n"
        "请输出一个 JSON 对象，格式为：\n"
        "{\"aligned_correct_facts\": [\"事实内容1\", \"事实内容2\"]}\n"
        "不要添加任何解释、说明或额外的文本。\n\n"
        f"{common_blocks}"
        f"{_section('当前子问题题干', current_subproblem_text)}\n"
        f"{_section('当前步骤意图', current_step_intent)}\n"
        f"{_section('当前步骤内容', current_step_text)}\n"
        f"{_section('下一步内容', next_step_text or '（当前已是本问最后一步）')}\n"
        f"{_section('当前已确认的正确事实（截至本步之前）', _list_block(current_correct_state))}\n"
        f"{_section('剩余正确事实序列', _list_block(remaining_correct_solution_steps))}\n\n"
        "请直接输出 JSON 结果。"
    )
    return system_prompt, user_prompt


def build_attribution_prompt(
    shared_stem_text: str,
    shared_problem_context: list[str],
    current_subproblem_text: str,
    step_record: dict[str, Any],
    correct_next_step: str,
) -> tuple[str, str]:
    system_prompt = (
        "你是一位经验丰富的数学教师，善于从学生的作答细节中发现其知识漏洞或思维误区。"
        "你的任务是对学生当前这一步的自主错误进行分析，并生成直接面向学生的反馈。"
    )
    user_prompt = (
        "请对当前这一步自主错误进行分析，并撰写反馈。\n\n"
        "【错误类型标签】\n"
        "请从以下五种标准类型中选择一个或多个最符合当前错误特征的标签。"
        "若错误特征确与所有类别均不相符，使用「其他未明确错误」作为标签。\n"
        "1. 公式与法则误用：对定理、公式或运算法则的调用存在事实性偏差，例如符号颠倒、法则适用条件错误。\n"
        "2. 基础计算失误：解题逻辑正确，但在算术运算、代数化简、移项合并等执行环节出现局部计算差错。\n"
        "3. 审题遗漏与条件忽视：推导过程本身成立，但未能兼顾题干给出的显性条件、隐含约束或边界限制，例如未舍去增根、忽略定义域。\n"
        "4. 概念混淆与理解偏差：对核心数学概念的定义、内涵或外延存在认知混淆，例如混淆充分条件与必要条件。\n"
        "5. 逻辑推断失误：在多步复合推理中出现分类讨论缺失、等价转化不当或因果倒置。\n\n"
        "【输出字段说明】\n"
        "- primary_reason_type：字符串，表示本步错误中最主要、最核心的错误类型标签。只能填写一个标签。\n"
        "- attribution_reason：对本次错误的详细分析概括。请依次涵盖：学生错误前已掌握的事实、本步操作与正确逻辑的具体偏离点、结合当前步骤正确的做法说明本应如何推进。用精炼的书面语表达，限 300 字以内。\n"
        "- reason_type：字符串列表，包含一个或多个错误类型标签；其中必须包含 primary_reason_type。\n"
        "- feedback_text：面向学生的反馈内容，使用教师口吻，语气温和而坚定。\n\n"
        "【反馈撰写要求】\n"
        "1. 反馈内容应直接面对学生，使用「你」作为称呼，可适当包含鼓励性话语，但必须明确指出问题所在。\n"
        "2. 结合学生错误前的状态、错误步骤内容以及当前步骤正确的做法，清晰地说明：\n"
        "   - 哪里出现了偏差；\n"
        "   - 为什么会出现这个偏差（从知识或思维角度简要说明）；\n"
        "   - 正确的做法或思考方向是什么。\n"
        "3. 避免使用过于严厉的批评口吻，保持专业且具有建设性。\n"
        "4. 反馈内容仅针对当前这一步错误，不涉及其他步骤。\n\n"
        "【分析流程】\n"
        "在撰写 attribution_reason 前，请先完成以下分析，并将其凝练至该字段中：\n"
        "1. 确认学生在错误发生前已掌握的事实（包括可能已存在的错误前提）。\n"
        "2. 指出本步操作与正确逻辑之间的具体偏离点。\n"
        "3. 结合当前步骤正确的做法，说明本应如何推进。\n"
        "4. 基于以上分析，先确定一个最主要的错误类型标签作为 primary_reason_type，再补充 reason_type 列表。\n\n"
        f"{_section('公共题干原文', shared_stem_text) if shared_stem_text.strip() else ''}"
        f"{_section('题干已知信息', _list_block(shared_problem_context))}\n"
        f"{_section('当前子问题题干', current_subproblem_text)}\n"
        f"{_section('当前错误步骤记录', _step_record_block(step_record))}\n"
        f"{_section('当前步骤正确的做法', correct_next_step or '（暂无）')}\n\n"
        "请直接输出符合以下格式的 JSON 对象，不要添加任何额外说明：\n"
        "{\"primary_reason_type\": \"最主要标签\", \"attribution_reason\": \"归因概括内容\", \"reason_type\": [\"标签1\", \"标签2\"], \"feedback_text\": \"面向学生的反馈内容\"}"
    )
    return system_prompt, user_prompt
