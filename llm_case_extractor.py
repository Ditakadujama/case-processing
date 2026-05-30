"""
LLM 病例卡抽取器 — 从原始病历中提取结构化病例卡。

流程：
  1. extract_llm_context(text) → 从长病历中压缩提取 LLM 输入上下文
  2. build_case_card_prompt(context, record_id) → 构建 prompt
  3. LLMCaseExtractor.extract(text, record_id) → 调用 LLM → 解析 JSON → 校验
  4. validate_case_card(card, source_text) → 证据校验、合规检查
"""

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════════

# LLMConfig 已迁移到项目根目录 config.py，此处保留别名以兼容旧导入路径
from config import LLMConfig  # noqa: F401  # 向后兼容，新代码请用 from config import LLMConfig


# ═══════════════════════════════════════════════════════════════════
# LLM 输入上下文提取
# ═══════════════════════════════════════════════════════════════════

# 优先抽取的章节（按优先级排序）
PRIORITY_SECTIONS = [
    "patient_info", "chief_complaint", "主诉",
    "history_illness", "现病史",
    "inspection_visit", "查房记录",
    "surgery_record", "手术记录",
    "operation_record",
    "examine", "检查",
]

# 从 checkout 中提取的关键检验指标
KEY_LAB_INDICATORS = [
    "pH", "Lac", "肌酐", "尿素", "超敏C反应蛋白", "白细胞计数", "血小板计数",
    "血红蛋白", "D-二聚体", "纤维蛋白原", "降钙素原", "白蛋白", "胆红素",
    "ALT", "AST", "肌钙蛋白", "BNP", "NT-proBNP", "氧合指数", "P/F",
]

# 从 doctor_advice 中提取的关键治疗/药物
KEY_TREATMENTS = [
    "机械通气", "气管插管", "呼吸机", "CRRT", "血滤", "透析",
    "IABP", "主动脉内球囊反搏", "ECMO", "PICCO",
    "去甲肾上腺素", "肾上腺素", "多巴胺", "多巴酚丁胺",
    "丙泊酚", "瑞芬太尼", "咪达唑仑",
    "冰冻血浆", "冷沉淀", "血小板", "人纤维蛋白原",
    "抗感染", "抗生素", "美罗培南", "万古霉素", "替加环素",
]

# 上下文关键字（在这些关键字附近抓取上下文）
CONTEXT_KEYWORDS = [
    "入院诊断", "目前诊断", "出院诊断", "术后诊断", "修正诊断",
    "手术名称", "手术日期",
    "机械通气", "气管插管", "气管切开", "呼吸机",
    "CRRT", "血滤", "透析", "IABP", "ECMO", "PICCO",
    "去甲肾上腺素", "升压", "血管活性",
    "休克", "心衰", "呼吸衰竭", "肾功能不全", "肾衰竭",
    "感染", "肺部感染", "血栓", "出血",
    "死亡", "转出", "转入", "转科",
    "并发症", "合并", "出现",
]


def remove_non_clinical_consent_text(text: str) -> str:
    """
    多层次过滤知情同意书、高值耗材、医保药品、签名等非临床段落（P1-5）。

    第一层：按 ### 分章节，移除标题匹配知情同意/风险告知模式的整个章节。
    第二层：段落级过滤 — 对章节内部（如 operation_record）的
            「知情同意书-xxx」「高值耗材」「患者签名」等标记做段落切除，
            这些标记不是 ### 章节，而是普通文本中的段落标签。

    保留真正的病程记录、会诊记录、手术记录、转出记录。
    """
    if not text:
        return text

    # ═══════════════════════════════════════════════════════════
    # 第一层：按 ### 章节级过滤
    # ═══════════════════════════════════════════════════════════
    sections = re.split(r"(###[^\n]*)", text)
    kept_parts = []
    skip_next = False

    if len(sections) > 1:
        for part in sections:
            title_match = re.match(r"###([^：:\n]+)", part)
            if title_match:
                title = title_match.group(1).strip()
                should_skip = False
                for skip_pat in INFORMED_CONSENT_SECTION_TITLES:
                    if skip_pat in title:
                        should_skip = True
                        break
                if any(p in title for p in ["知情同意", "风险告知", "授权委托", "自付比例"]):
                    should_skip = True

                if should_skip:
                    skip_next = True
                    kept_parts.append(f"###{title}：[本节已过滤，非临床风险告知/同意内容]")
                    continue
                else:
                    skip_next = False
                    kept_parts.append(part)
            else:
                if skip_next:
                    skip_next = False
                    continue
                kept_parts.append(part)
        text = "".join(kept_parts)

    # ═══════════════════════════════════════════════════════════
    # 第二层：段落级过滤（针对章节内部的内联非临床标签）
    # ═══════════════════════════════════════════════════════════
    text = _filter_non_clinical_paragraphs(text)

    return text


def _filter_non_clinical_paragraphs(text: str) -> str:
    """
    段落级非临床内容切除 — 按内联文书标签分段，只切非临床段，保留其后的临床段。

    operation_record 内部包含多个内联文书（会诊记录、病程记录、知情同意书等），
    以行首标签（如「病程记录-」「知情同意书-」「转入记录」）作为段落边界。

    此函数：
    1. 按主要文书标签分段
    2. 保留临床段（会诊/病程/转入/转出/抢救/手术/出院/入院/死亡）
    3. 切除非临床段（知情同意书/高值耗材/医保药品/授权委托/医患沟通/根据医保协议规定）
    4. 保留段内再逐行过滤签名/费用等高密度非临床行
    """
    if not text:
        return text

    # ── 主要文书标签（行首出现时标志新段开始）──
    _CLINICAL_LABELS = {
        "手术相关记录", "病程记录", "会诊记录", "转入记录", "转出记录",
        "抢救记录", "出院记录", "入院记录", "死亡记录",
        "日常病程记录", "首次病程记录", "术后首次病程记录",
        "疾病诊断", "病情变化", "治疗计划", "注意事项",
    }
    _NON_CLINICAL_LABELS = {
        "知情同意书", "高值耗材", "医保药品", "授权委托",
        "根据医保协议规定", "医患沟通记录",
    }
    _ALL_LABELS = _CLINICAL_LABELS | _NON_CLINICAL_LABELS

    # 构建正则：匹配行首（文本开头可带前导空白、或 \n 后）紧跟任一标签
    _LABEL_PATTERN = re.compile(
        r'(?:^[ \t]*|\n)(?=(' + '|'.join(re.escape(l) for l in _ALL_LABELS) + r'))'
    )

    # 找到所有标签位置
    matches = list(_LABEL_PATTERN.finditer(text))
    if not matches:
        # 无主要标签，仅做行级过滤
        return _filter_non_clinical_lines(text)

    # 按标签位置分段：每段从当前标签之前到下一标签（或文末）
    kept_segments = []
    for i, m in enumerate(matches):
        label_text = m.group(1)
        # 段边界：从当前匹配位置（标签前的 \n 或文本开头）到下一标签（或文末）
        seg_start = m.start()
        next_start = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        segment = text[seg_start:next_start]

        # 首个标签前可能有无标签前导文本
        if i == 0 and seg_start > 0:
            kept_segments.append(text[:seg_start])

        if label_text in _CLINICAL_LABELS:
            kept_segments.append(segment)
        # 非临床标签 → 整段丢弃

    if not kept_segments:
        # 所有段都是非临床的，保留开头无标签部分（如果有）
        if matches and matches[0].start() > 0:
            return _filter_non_clinical_lines(text[:matches[0].start()])
        return ""

    result = "".join(kept_segments)

    # 最后一遍：行级过滤（签名/费用等）
    return _filter_non_clinical_lines(result)


def _filter_non_clinical_lines(text: str) -> str:
    """逐行过滤签名、费用等高密度非临床行。"""
    lines = text.split("\n")
    filtered = []

    NON_CLINICAL_LINE_PATTERNS = [
        "患者签名:", "家属签名:", "患者签名日期:", "家属签名日期:",
        "受委托人/家属签名:", "受委托人/家属备注:",
        "患者联系方式:", "家属联系方式:",
        "被授权人/家属签名:", "被授权人/家属备注:",
        "双击此处选择", "文本框",
        "自付比例",
        "医师签名：", "医师签名:",
    ]

    for line in lines:
        stripped = line.strip()

        if any(stripped.startswith(m) for m in ["高值耗材", "医保药品"]):
            continue

        if any(kw in stripped for kw in NON_CLINICAL_LINE_PATTERNS):
            continue

        filtered.append(line)

    return "\n".join(filtered)


def extract_llm_context(text: str, max_chars: int = 15000) -> str:
    """
    从长病历中提取 LLM 输入上下文。

    策略：
    1. 优先提取关键章节全文
    2. 从 checkout 中只提取关键检验指标行
    3. 从 doctor_advice 中只提取关键治疗/药物行
    4. 在关键字附近抓取上下文窗口
    5. 如果结果为空，回退到纯截断

    Args:
        text: 原始病历全文
        max_chars: 上下文最大字符数

    Returns:
        压缩后的 LLM 输入文本
    """
    chunks: List[str] = []

    # ── 1. 提取优先章节 ──
    section_set = set(PRIORITY_SECTIONS)
    # 匹配 ###section_name: content 格式
    for section_name, content in re.findall(
        r"###([^：:\n]+)[：:](.*?)(?=###[^\n]|\Z)", text, flags=re.DOTALL
    ):
        name = section_name.strip()
        if name in section_set and content.strip():
            # P1-5: operation_record 先过滤知情同意书/风险告知段落
            if name == "operation_record":
                content = remove_non_clinical_consent_text(content)
            # 每个章节最多保留 2000 字符
            trimmed = content.strip()[:2000]
            chunks.append(f"###{name}:\n{trimmed}")

    # ── 2. 从 checkout 提取关键检验指标 ──
    checkout_match = re.search(r"###checkout[：:](.*?)(?=###[a-zA-Z_]|\Z)", text, flags=re.DOTALL)
    if checkout_match:
        checkout_text = checkout_match.group(1)
        key_lines: List[str] = []
        for indicator in KEY_LAB_INDICATORS:
            for match in re.finditer(
                rf"[^;]*{re.escape(indicator)}[^;]*",
                checkout_text, flags=re.IGNORECASE
            ):
                line = match.group().strip()
                if len(line) > 10:  # 过滤掉太短的片段
                    key_lines.append(line)
        if key_lines:
            # 去重并限制长度
            unique_lines = list(dict.fromkeys(key_lines))[:30]
            chunks.append(f"###checkout(关键指标):\n{';\n'.join(unique_lines)}")

    # ── 3. 从 doctor_advice 提取关键治疗 ──
    advice_match = re.search(r"###doctor_advice[：:](.*?)(?=###[a-zA-Z_]|\Z)", text, flags=re.DOTALL)
    if advice_match:
        advice_text = advice_match.group(1)
        key_advice: List[str] = []
        for treatment in KEY_TREATMENTS:
            for match in re.finditer(
                rf"[^;]*{re.escape(treatment)}[^;]*",
                advice_text, flags=re.IGNORECASE
            ):
                line = match.group().strip()
                if len(line) > 5:
                    key_advice.append(line)
        if key_advice:
            unique_advice = list(dict.fromkeys(key_advice))[:20]
            chunks.append(f"###doctor_advice(关键治疗):\n{';\n'.join(unique_advice)}")

    # ── 4. 关键字附近上下文（从已过滤的干净文本中抓取，避免知情同意书噪声）──
    # 拼接步骤 1-3 已过滤的文本作为干净语料
    clean_corpus = "\n".join(chunks) if chunks else text
    kw_chunks: List[str] = []
    for kw in CONTEXT_KEYWORDS:
        start = 0
        hits = 0
        while hits < 3:  # 每个关键字最多抓 3 处
            idx = clean_corpus.find(kw, start)
            if idx < 0:
                break
            # 取前后各 120 字符作为上下文窗口
            window_start = max(0, idx - 120)
            window_end = min(len(clean_corpus), idx + len(kw) + 200)
            window = clean_corpus[window_start:window_end].strip()
            if len(window) > 20:
                kw_chunks.append(window)
            start = idx + len(kw)
            hits += 1

    if kw_chunks:
        # 去重（按内容）
        seen = set()
        unique_kw_chunks = []
        for c in kw_chunks:
            if c not in seen:
                seen.add(c)
                unique_kw_chunks.append(c)
        chunks.append("###关键词上下文:\n" + "\n---\n".join(unique_kw_chunks[:40]))

    # ── 5. 组装结果 ──
    if not chunks:
        # 回退：直接截取前 max_chars 字符
        return text[:max_chars]

    result = "\n\n".join(chunks)
    # 清理多余空白
    result = re.sub(r"\n{3,}", "\n\n", result)
    result = re.sub(r"[ \t]{2,}", " ", result)

    if len(result) > max_chars:
        # 按优先级截断：优先保留前面的章节
        result = result[:max_chars]

    return result


# ═══════════════════════════════════════════════════════════════════
# Prompt 构建
# ═══════════════════════════════════════════════════════════════════

# 病例卡 JSON Schema（嵌入 prompt 中）
CASE_CARD_SCHEMA = """
{
  "record_id": "病例ID",
  "patient_profile": {
    "age": 整数,
    "gender": "男/女"
  },
  "chief_problem": "主要临床问题，概括主病情的一句话",
  "icu_reason": "进入ICU/重症监护的核心原因",
  "primary_diagnoses": ["主诊断1", "主诊断2"],
  "secondary_diagnoses": ["基础病或次要诊断"],
  "surgery_or_operations": [
    {
      "name": "手术/操作名称",
      "normalized_name": "标准化名称",
      "time": "日期 YYYY-MM-DD 或 null",
      "category": "cardiac_support/neuro_surgery/general_surgery/ortho/other",
      "evidence": "原文依据"
    }
  ],
  "key_interventions": [
    {
      "name": "干预名称（如机械通气、CRRT、IABP、升压药）",
      "time": "日期 或 null",
      "evidence": "原文依据"
    }
  ],
  "organ_failures": [
    {
      "name": "器官功能问题（如循环衰竭、呼吸衰竭、肾功能障碍、肝功能障碍、凝血功能障碍、神经系统问题）",
      "evidence": "原文依据"
    }
  ],
  "complications": [
    {
      "name": "并发症名称",
      "status": "confirmed/suspected",
      "evidence": "原文依据"
    }
  ],
  "clinical_course": [
    {
      "stage": "阶段名称（如入ICU、高级生命支持、病情稳定、并发症处理、转出/死亡）",
      "time": "日期 或 null",
      "summary": "阶段摘要"
    }
  ],
  "outcome": {
    "status": "好转/转出/死亡/自动出院/unknown",
    "evidence": "原文依据或空字符串"
  },
  "severity_level": "critical/severe/moderate/mild/unknown",
  "summary_for_embedding": "用于语义检索的事实摘要，100-300字，包含：人口学信息、主要诊断、关键干预、器官功能问题、并发症、转归。不得加入推测。"
}
"""

EXTRACTION_SYSTEM_PROMPT = """你是 ICU 临床病历信息抽取专家。请只根据给定的病历文本抽取结构化病例卡。

核心原则：
1. 只允许基于原文抽取，禁止根据医学常识补全原文没有的信息。
2. 如果信息不明确，输出 null、unknown 或空数组 []。
3. 每个诊断、手术/操作、关键干预、器官功能问题、并发症、转归必须提供 evidence 字段，引用原文片段。
4. 严格区分"已经发生"与"风险告知/知情同意"：知情同意书中列出的潜在并发症风险绝不能当作已发生的并发症，除非病程记录、检查报告或诊断明确写明已经发生。
5. summary_for_embedding 必须是纯事实摘要，不得加入任何推测或模型自己的医学判断。
6. 输出必须是严格的 JSON，不要包含 markdown 代码块标记（如 ```json），不要加任何解释性文字。直接输出 JSON 对象。

ICU 病历常见模式提示：
- 关键干预优先识别：机械通气（气管插管/呼吸机）、CRRT/血滤/透析、IABP/主动脉内球囊反搏、ECMO、升压药（去甲肾上腺素/多巴胺等）、PICCO监测
- 器官功能问题按系统识别：循环（休克/心衰/需升压药）、呼吸（呼衰/ARDS/需机械通气）、肾脏（AKI/肌酐升高/需CRRT）、肝脏（肝功能异常/黄疸）、凝血（DIC/血小板减少/出血）、神经（昏迷/GCS下降）
- 并发症注意区分：已确诊的感染（肺部感染/血流感染等）、血栓/栓塞、出血事件等
- 手术/操作请从手术记录或操作记录中提取，注意区分手术和床边操作
- 如果病历中记录了患者死亡，outcome.status 应为 "死亡"；如果转出ICU，应为 "转出"
"""


def build_case_card_prompt(context: str, record_id: str = "") -> str:
    """
    构建 LLM case card 抽取的完整 prompt。

    Args:
        context: 经过 extract_llm_context 压缩的病历文本
        record_id: 病例 ID

    Returns:
        完整的 user prompt 字符串
    """
    parts = [
        "请从以下 ICU 病历中抽取结构化病例卡。",
        "",
        "输出 JSON Schema：",
        CASE_CARD_SCHEMA,
    ]
    if record_id:
        parts.append(f"\n病历 ID：{record_id}")

    parts.append(f"\n病历文本：\n{context}")

    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════
# LLM 调用
# ═══════════════════════════════════════════════════════════════════

class LLMCaseExtractor:
    """
    LLM 病例卡抽取器。

    用法:
        config = LLMConfig()
        extractor = LLMCaseExtractor(config)
        card = extractor.extract(raw_text, record_id="ZY...")
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self._request_count = 0
        self._error_count = 0
        self._total_tokens = 0

    @property
    def is_available(self) -> bool:
        return self.config.is_configured

    def extract(self, text: str, record_id: str = "") -> Optional[dict]:
        """
        从原始病历文本抽取病例卡。

        Args:
            text: 原始病历全文
            record_id: 病例 ID

        Returns:
            校验后的病例卡 dict，失败时返回 None
        """
        if not self.is_available:
            logger.warning("LLM 服务未配置，跳过病例卡抽取")
            return None

        # 1. 提取上下文
        context = extract_llm_context(text)
        if not context.strip():
            logger.warning(f"[{record_id}] LLM 上下文为空，跳过")
            return None

        # 2. 构建 prompt
        user_prompt = build_case_card_prompt(context, record_id)

        # 3. 调用 LLM
        raw_response = self._call_llm(user_prompt)

        # 4. 解析 JSON
        card = self._parse_json_response(raw_response, record_id)
        if card is None:
            return None

        # 5. 校验
        card["record_id"] = record_id
        card = validate_case_card(card, text)

        return card

    def _call_llm(self, user_prompt: str) -> str:
        """调用 OpenAI-compatible Chat Completions API"""
        import urllib.request
        import urllib.error

        url = f"{self.config.api_base.rstrip('/')}/chat/completions"

        payload = json.dumps({
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }).encode("utf-8")

        for attempt in range(self.config.max_retries):
            try:
                req = urllib.request.Request(url, data=payload, method="POST")
                req.add_header("Content-Type", "application/json")
                req.add_header("Authorization", f"Bearer {self.config.api_key}")

                with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:
                    body = json.loads(resp.read().decode("utf-8"))

                usage = body.get("usage", {})
                self._total_tokens += usage.get("total_tokens", 0)
                self._request_count += 1

                content = body["choices"][0]["message"]["content"]
                return content.strip()

            except urllib.error.HTTPError as e:
                error_body = ""
                try:
                    error_body = e.read().decode("utf-8")[:500]
                except Exception:
                    pass
                logger.warning(
                    f"LLM API HTTP {e.code} (attempt {attempt+1}/{self.config.max_retries}): {error_body}"
                )
                if e.code == 429:  # Rate limit
                    time.sleep(min(10 * (2 ** attempt), 60))
                elif e.code >= 500:
                    time.sleep(2 ** attempt)
                else:
                    # 4xx errors (except 429) are not retryable
                    self._error_count += 1
                    logger.error(f"LLM API 请求失败 (HTTP {e.code}): {error_body}")
                    raise

            except Exception as e:
                logger.warning(f"LLM API 请求异常 (attempt {attempt+1}/{self.config.max_retries}): {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    self._error_count += 1
                    raise RuntimeError(f"LLM API 请求失败（已重试{self.config.max_retries}次）: {e}")

        self._error_count += 1
        return ""

    def _parse_json_response(self, raw: str, record_id: str) -> Optional[dict]:
        """解析 LLM 返回的 JSON 字符串，处理常见格式问题"""
        if not raw:
            logger.warning(f"[{record_id}] LLM 返回空响应")
            return None

        # 尝试直接解析
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # 尝试移除 markdown 代码块标记
        cleaned = raw
        # 移除 ```json ... ``` 或 ``` ... ```
        cleaned = re.sub(r'^```(?:json)?\s*\n?', '', cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r'\n?```\s*$', '', cleaned, flags=re.MULTILINE)

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # 尝试提取第一个 { 到最后一个 } 之间的内容
        first_brace = cleaned.find('{')
        last_brace = cleaned.rfind('}')
        if first_brace >= 0 and last_brace > first_brace:
            try:
                return json.loads(cleaned[first_brace:last_brace + 1])
            except json.JSONDecodeError:
                pass

        logger.warning(f"[{record_id}] JSON 解析失败，原始响应前 200 字符: {raw[:200]}")
        return None

    def get_stats(self) -> dict:
        """返回抽取器统计信息"""
        return {
            "available": self.is_available,
            "request_count": self._request_count,
            "error_count": self._error_count,
            "total_tokens": self._total_tokens,
            "model": self.config.model,
        }


# ═══════════════════════════════════════════════════════════════════
# 病例卡校验
# ═══════════════════════════════════════════════════════════════════

# 知情同意/风险告知相关的上下文关键词
INFORMED_CONSENT_PATTERNS = [
    "知情同意", "知情权", "风险告知", "可能出现", "可能发生",
    "并发症或风险", "潜在风险", "有风险", "风险包括", "风险如下",
    "可能出现以下并发症", "手术风险", "麻醉风险",
]

# 知情同意书/非临床章节标题模式（用于段落级判断和上下文过滤）
INFORMED_CONSENT_SECTION_TITLES = [
    "知情同意书", "手术知情同意", "麻醉知情同意", "输血知情同意",
    "有创操作知情", "高值耗材", "医保药品", "自付比例",
    "患者签名", "家属签名", "授权委托书", "医患沟通记录",
    "风险告知书",
]


def _fuzzy_find_in_text(evidence: str, source_text: str) -> bool:
    """
    模糊匹配：检查 evidence 是否在 source_text 中。
    对 evidence 做清理后，先精确匹配，再逐句匹配。
    """
    if not evidence or len(evidence.strip()) < 2:
        return False

    ev = evidence.strip()
    # 移除可能的前后缀标点
    ev = re.sub(r'^[，。；：、\s]+|[，。；：、\s]+$', '', ev)

    if ev in source_text:
        return True

    # 取 evidence 中最长的一段（移除诊断编号等前缀）
    core = re.sub(r'^\d+[、.]?\s*', '', ev)
    if len(core) >= 3 and core in source_text:
        return True

    # 逐句匹配（按中文标点分句）
    sentences = re.split(r'[。；\n]', source_text)
    for sent in sentences:
        if core in sent:
            return True

    return False


def _is_from_informed_consent(evidence: str, source_text: str) -> bool:
    """
    判断 evidence 是否来自知情同意书/风险告知等章节（P1-4: 段落级/章节级判断）。

    原来用 ±500 字符窗口粗暴判断，容易误伤真实并发症。
    现在改为：
    1. 找到 evidence 所在位置
    2. 向前找到最近的 ### 章节标题
    3. 判断章节标题是否属于知情同意/风险告知
    4. 再检查 evidence 所在句子是否包含知情同意关键词
    """
    if not evidence:
        return False

    # 查找 evidence 在原文中的位置
    idx = source_text.find(evidence.strip())
    if idx < 0:
        return False

    # ── 1. 章节级判断：找最近章节起点 ──
    section_start = source_text.rfind("###", 0, idx)
    if section_start < 0:
        section_start = max(0, idx - 120)

    # 找最近章节终点（也是下一章节的起点）
    section_end = source_text.find("###", idx)
    if section_end < 0:
        section_end = min(len(source_text), idx + 300)

    section_text = source_text[section_start:section_end]

    # 提取章节标题
    section_title_match = re.match(r"###([^：:\n]+)", section_text)
    section_title = section_title_match.group(1).strip() if section_title_match else ""

    # 如果章节标题明显是知情同意书，判为 risk
    for title_pat in INFORMED_CONSENT_SECTION_TITLES:
        if title_pat in section_title:
            return True

    # 章节标题本身包含风险相关词
    if any(p in section_title for p in ["知情同意", "风险告知", "授权委托"]):
        return True

    # ── 2. 句子级判断：检查 evidence 所在句子 ──
    sentence = _extract_sentence_around(source_text, idx)
    if any(p in sentence for p in INFORMED_CONSENT_PATTERNS):
        return True

    return False


def _extract_sentence_around(text: str, pos: int) -> str:
    """
    提取 pos 位置所在的中文句子。

    按中文标点（。；！？\n）分句，向前找到句首，向后找到句尾。
    """
    # 向前找句首
    sentence_start = pos
    for i in range(pos - 1, max(0, pos - 500) - 1, -1):
        if text[i] in "。；！？\n":
            sentence_start = i + 1
            break
    else:
        sentence_start = max(0, pos - 200)

    # 向后找句尾
    sentence_end = pos
    for i in range(pos, min(len(text), pos + 500)):
        if text[i] in "。；！？\n":
            sentence_end = i + 1
            break
    else:
        sentence_end = min(len(text), pos + 200)

    return text[sentence_start:sentence_end]


def validate_case_card(card: dict, source_text: str) -> dict:
    """
    校验并清洗 LLM 输出的病例卡。

    校验规则：
    1. 必填字段必须存在
    2. summary_for_embedding 不能为空
    3. evidence 必须在原文中（模糊匹配），不在则标记 low_confidence 或删除
    4. 并发症若 evidence 来自知情同意/风险告知 → 不能为 confirmed
    5. 标签标准化

    Args:
        card: LLM 输出的病例卡 dict
        source_text: 原始病历全文

    Returns:
        添加了 confidence 标注的病例卡 dict
    """
    # ── 1. 必填字段检查 ──
    if not isinstance(card, dict):
        logger.warning("病例卡校验失败：不是有效的 dict")
        return {"error": "invalid_type", "raw": str(card)}

    card.setdefault("record_id", "")
    card.setdefault("chief_problem", "")
    card.setdefault("icu_reason", "")
    card.setdefault("primary_diagnoses", [])
    card.setdefault("secondary_diagnoses", [])
    card.setdefault("surgery_or_operations", [])
    card.setdefault("key_interventions", [])
    card.setdefault("organ_failures", [])
    card.setdefault("complications", [])
    card.setdefault("clinical_course", [])
    card.setdefault("outcome", {"status": "unknown", "evidence": ""})
    card.setdefault("severity_level", "unknown")
    card.setdefault("summary_for_embedding", "")

    # ── 2. summary_for_embedding 非空检查（P2-8: 改进兜底质量）──
    summary = card.get("summary_for_embedding", "")
    if not summary or len(summary.strip()) < 20:
        logger.warning(f"[{card.get('record_id', '?')}] summary_for_embedding 为空或过短，生成兜底摘要")
        # 从病例卡各字段组合更完整的摘要
        summary_parts = []
        chief = card.get("chief_problem", "")
        if chief:
            summary_parts.append(chief)
        icu = card.get("icu_reason", "")
        if icu:
            summary_parts.append(f"进入ICU原因：{icu}")

        primary = card.get("primary_diagnoses", [])
        if primary:
            diag_names = [d if isinstance(d, str) else d.get("name", "") for d in primary]
            diag_names = [n for n in diag_names if n]
            if diag_names:
                summary_parts.append("主要诊断：" + "、".join(diag_names))

        # 关键干预
        interventions = card.get("key_interventions", [])
        surgeries = card.get("surgery_or_operations", [])
        all_ops = interventions + surgeries
        if all_ops:
            op_names = []
            for op in all_ops:
                if isinstance(op, dict):
                    n = op.get("normalized_name", "") or op.get("name", "")
                else:
                    n = str(op)
                if n:
                    op_names.append(n)
            if op_names:
                summary_parts.append("关键干预：" + "、".join(op_names))

        # 器官功能问题
        organs = card.get("organ_failures", [])
        if organs:
            org_names = [o.get("name", "") if isinstance(o, dict) else str(o) for o in organs]
            org_names = [n for n in org_names if n]
            if org_names:
                summary_parts.append("器官功能问题：" + "、".join(org_names))

        # 转归
        outcome = card.get("outcome", {})
        if isinstance(outcome, dict) and outcome.get("status"):
            summary_parts.append(f"转归：{outcome['status']}")

        fallback = "。".join([p for p in summary_parts if p])
        if len(fallback.strip()) >= 10:
            card["summary_for_embedding"] = fallback
            card["_summary_fallback"] = True
        elif chief:
            card["summary_for_embedding"] = chief
            card["_summary_fallback"] = True
            card["_summary_low_quality"] = True
        else:
            card["summary_for_embedding"] = "无有效摘要"
            card["_summary_low_quality"] = True

    # ── 3. evidence 校验 ──
    card = _validate_evidence_items(card, "surgery_or_operations", source_text)
    card = _validate_evidence_items(card, "key_interventions", source_text)
    card = _validate_evidence_items(card, "organ_failures", source_text)
    card = _validate_evidence_items(card, "complications", source_text, check_consent=True)

    # ── 4. 诊断列表转纯字符串 ──
    card["primary_diagnoses"] = _normalize_string_list(card.get("primary_diagnoses", []))
    card["secondary_diagnoses"] = _normalize_string_list(card.get("secondary_diagnoses", []))

    return card


def _validate_evidence_items(
    card: dict,
    field: str,
    source_text: str,
    check_consent: bool = False,
) -> dict:
    """
    校验列表中每个条目的 evidence 是否在原文中。

    不在原文中的条目标记 confidence="low"；
    若 check_consent=True 且 evidence 来自知情同意书，标记 status="risk_only"
    并降低 confidence 为 "low"。
    """
    items = card.get(field, [])
    if not items:
        return card

    validated = []
    for item in items:
        if not isinstance(item, dict):
            continue

        evidence = item.get("evidence", "")
        name = item.get("name", "")

        # 如果 name 非空但 evidence 为空，尝试用 name 做宽松匹配
        if not evidence and name:
            # 不需要严格校验，标记为 medium
            item["confidence"] = "medium"
            item["_evidence_validated"] = False
            validated.append(item)
            continue

        # 校验 evidence 是否在原文中
        if evidence:
            in_text = _fuzzy_find_in_text(evidence, source_text)
            item["_evidence_validated"] = in_text

            if not in_text:
                item["confidence"] = "low"
                logger.debug(f"evidence 未在原文中找到: {name} -> '{evidence[:60]}...'")
            else:
                item["confidence"] = "high"
        else:
            item["confidence"] = "medium"
            item["_evidence_validated"] = False

        # 检查是否来自知情同意书
        if check_consent and evidence and _is_from_informed_consent(evidence, source_text):
            current_status = item.get("status", "confirmed")
            if current_status in ("confirmed", "suspected"):
                item["status"] = "risk_only"
                item["confidence"] = "low"
                item["_from_informed_consent"] = True
                logger.info(f"并发症 {name} 的 evidence 来自知情同意/风险告知，标记为 risk_only")

        validated.append(item)

    card[field] = validated
    return card


def _normalize_string_list(items: List) -> List[str]:
    """将列表中的项转为纯字符串（处理 dict 和 str 混合的情况）"""
    result = []
    for item in (items or []):
        if isinstance(item, str):
            result.append(item.strip())
        elif isinstance(item, dict):
            result.append(item.get("name", "").strip())
    return [r for r in result if r]
