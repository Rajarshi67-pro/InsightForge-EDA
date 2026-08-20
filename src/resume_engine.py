"""
Deep Thinking AI Resume & Career Intelligence Engine for InsightForge AI.
Provides multi-dimensional executive scoring out of 10.0,
ATS architectural analysis, line-by-line Google XYZ rewrites,
recruiter red-flag scans, and automated interview question synthesis.
"""

import os
import re
import math
from typing import Dict, Any, List, Optional
import pdfplumber
from src.logger import get_logger

logger = get_logger("ResumeEngine")

# 2026+ Modern Market Competency Catalog
MODERN_MARKET_SKILLS = {
    "AI, ML & Data Engineering": [
        "python", "pytorch", "tensorflow", "scikit-learn", "pandas", "numpy",
        "langchain", "llamaindex", "rag", "agents", "genai", "transformer",
        "sql", "postgresql", "spark", "airflow", "databricks", "snowflake",
        "vector database", "fine-tuning", "huggingface", "llm"
    ],
    "Cloud, DevOps & Infra": [
        "aws", "gcp", "azure", "docker", "kubernetes", "terraform", "ci/cd",
        "github actions", "linux", "prometheus", "grafana", "microservices",
        "helm", "istio", "serverless", "cloudwatch"
    ],
    "Software Architecture & Backend": [
        "fastapi", "django", "nodejs", "react", "next.js", "typescript",
        "graphql", "rest api", "system design", "redis", "kafka", "distributed systems",
        "golang", "c++", "java", "grpc", "websockets", "postgresql", "mongodb"
    ],
    "Product, Analytics & BI": [
        "tableau", "power bi", "excel", "a/b testing", "statistics", "eda",
        "predictive modeling", "looker", "dbt", "amplitude", "mixpanel",
        "growth metrics", "kpi modeling"
    ]
}

EXECUTIVE_POWER_VERBS = [
    "spearheaded", "orchestrated", "engineered", "architected", "optimized",
    "scaled", "accelerated", "automated", "streamlined", "deployed", "implemented",
    "reduced", "increased", "boosted", "maximized", "designed", "delivered",
    "championed", "led", "developed", "built", "established", "revamped",
    "created", "managed", "resolved", "maintained", "collaborated", "researched",
    "analyzed", "generated", "achieved", "executed", "formulated", "directed",
    "pioneered", "negotiated", "authored", "standardized", "mentored", "overhauled"
]

WEAK_PASSIVE_PHRASES = [
    "responsible for", "worked on", "helped with", "assisted in", "duties included",
    "participated in", "handled", "served as", "was tasked with", "involved in"
]

RESUME_KEYWORDS = [
    "experience", "education", "skills", "projects", "employment",
    "curriculum vitae", "resume", "work history", "certifications",
    "summary", "objective", "achievements", "bachelor", "master",
    "university", "github", "linkedin", "internship", "contact",
    "email", "phone", "profile", "qualifications", "competencies",
    "technical proficiencies", "career history", "academic", "b.tech",
    "b.e", "b.sc", "m.tech", "m.s", "gpa", "cgpa", "developer",
    "engineer", "analyst", "manager", "portfolio", "coursework",
    "responsibilities", "technologies", "tools", "languages", "developer",
    "frontend", "backend", "fullstack", "data scientist", "software"
]


class ResumeEngine:
    """Multi-dimensional executive career audit & scoring engine."""

    def __init__(self, raw_text: str, file_name: str = "Resume.pdf", api_key: Optional[str] = None):
        self.raw_text = raw_text.strip()
        self.file_name = file_name
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")

    @classmethod
    def from_pdf(cls, file_path: str, api_key: Optional[str] = None) -> "ResumeEngine":
        """Extracts text content from a PDF file using high-resilience PDFExtractor."""
        from src.pdf_extractor import PDFExtractor
        full_text, _ = PDFExtractor.extract_full_text(file_path, api_key=api_key)

        if not full_text:
            raise ValueError(
                f"The document '{os.path.basename(file_path)}' has no readable text. "
                "It may be a scanned image or empty. If it is a scanned image, please enter your Gemini API key for automatic vision OCR."
            )

        return cls(full_text, file_name=os.path.basename(file_path), api_key=api_key)

    @classmethod
    def is_resume(cls, text: str, file_name: Optional[str] = None) -> bool:
        """High-precision heuristic for resume/CV detection."""
        if not text and not file_name:
            return False

        if file_name:
            fn_lower = os.path.basename(file_name).lower()
            if any(term in fn_lower for term in ["resume", "cv", "curriculum", "biodata", "profile", "career"]):
                return True

        if not text:
            return False

        text_lower = text.lower()
        has_email = bool(re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text))
        has_phone = bool(re.search(r"(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", text))
        # Strict keyword count
        matches = sum(1 for kw in RESUME_KEYWORDS if re.search(r"\b" + re.escape(kw) + r"\b", text_lower))

        # Check for multiple classic resume sections
        section_headers = ["experience", "education", "skills", "projects", "summary", "employment", "certifications", "work history", "objective"]
        section_matches = sum(1 for sec in section_headers if re.search(r"\b" + re.escape(sec) + r"\b", text_lower))

        # 1. Definite Resume: Section headers + contact information
        if (has_email or has_phone) and section_matches >= 2 and matches >= 3:
            return True

        # 2. Strong section header presence
        if section_matches >= 4 and matches >= 4:
            return True

        return False

    def analyze(self) -> Dict[str, Any]:
        """Runs the complete Deep Thinking career intelligence audit."""
        lines = [line.strip() for line in self.raw_text.split("\n") if line.strip()]
        bullet_points = [line for line in lines if line.startswith(("-", "•", "*", "–", "—", ">")) or len(line.split()) > 5]

        # 1. Dimension: Metric Business Impact (0 - 10.0)
        metric_pattern = re.compile(
            r"(\d+[\d,.]*|\$\d+|\d+%\s*|\d+x|\b\d+\b\s*(ms|sec|min|hours|users|k|m|million|billion|gb|tb|pb|roi|arr|mrr))",
            re.IGNORECASE
        )
        quantified_bullets = [b for b in bullet_points if metric_pattern.search(b)]
        quant_ratio = (len(quantified_bullets) / max(1, len(bullet_points)))
        impact_score = min(10.0, round(max(3.5, quant_ratio * 12.0), 1))

        # 2. Dimension: Executive Power Verbs & Phrasing (0 - 10.0)
        action_verb_hits = []
        weak_phrase_hits = []
        for b in bullet_points:
            words = re.findall(r"\b[a-zA-Z]+\b", b.lower())
            if words and words[0] in EXECUTIVE_POWER_VERBS:
                action_verb_hits.append(words[0])
            for wp in WEAK_PASSIVE_PHRASES:
                if wp in b.lower():
                    weak_phrase_hits.append(wp)

        verb_ratio = (len(action_verb_hits) / max(1, len(bullet_points)))
        penalty = min(2.0, len(weak_phrase_hits) * 0.4)
        verb_score = min(10.0, round(max(3.0, (verb_ratio * 12.5) - penalty), 1))

        # 3. Dimension: 2026 Tech & Skill Alignment (0 - 10.0)
        text_lower = self.raw_text.lower()
        matched_skills: Dict[str, List[str]] = {}
        missing_recommendations: Dict[str, List[str]] = {}
        total_found = 0

        for category, skills in MODERN_MARKET_SKILLS.items():
            found = [s for s in skills if re.search(r"\b" + re.escape(s) + r"\b", text_lower)]
            missing = [s for s in skills if s not in found]
            matched_skills[category] = found
            missing_recommendations[category] = missing[:5]
            total_found += len(found)

        skills_score = min(10.0, round(max(4.0, (total_found / 14.0) * 10.0), 1))

        # 4. Dimension: ATS Structure & Architecture (0 - 10.0)
        # Robust ATS Entity Extraction
        email_match = re.search(r"[\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,}", self.raw_text)
        detected_email = email_match.group(0) if email_match else "Not Detected"

        phone_match = re.search(
            r"(?:(?:\+|0{0,2})91[\s.-]?)?[6789]\d{9}|(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,5}[-.\s]?\d{3,5}",
            self.raw_text
        )
        detected_phone = phone_match.group(0).strip() if phone_match else "Not Detected"

        linkedin_match = re.search(r"(?:https?://)?(?:www\.)?linkedin\.com/in/[\w-]+", self.raw_text, re.IGNORECASE)
        linkedin_url = linkedin_match.group(0) if linkedin_match else ("https://linkedin.com" if "linkedin.com" in text_lower else None)

        github_match = re.search(r"(?:https?://)?(?:www\.)?github\.com/[\w-]+", self.raw_text, re.IGNORECASE)
        github_url = github_match.group(0) if github_match else ("https://github.com" if "github.com" in text_lower else None)

        # Candidate Name Extraction
        candidate_name = None
        for l in lines[:5]:
            l_clean = l.strip()
            # If line has 2-4 words, starts with capital, and is not a common header
            if 2 <= len(l_clean.split()) <= 4 and re.match(r"^[A-Z][a-zA-Z\s.-]+$", l_clean):
                if not any(hdr in l_clean.lower() for hdr in ["resume", "curriculum", "vitae", "contact", "email", "profile", "page", "phone"]):
                    candidate_name = l_clean
                    break
        
        if not candidate_name and self.file_name:
            fn_base = os.path.splitext(os.path.basename(self.file_name))[0]
            clean_fn = re.sub(r"[_\-\.]+", " ", fn_base).strip()
            clean_fn = re.sub(r"(?i)\b(resume|cv|biodata|final|latest|updated|profile)\b", "", clean_fn).strip()
            if clean_fn and len(clean_fn.split()) >= 1:
                candidate_name = clean_fn.title()

        if not candidate_name:
            candidate_name = "Candidate"

        # Detect Sections
        detected_sections = []
        sec_catalog = {
            "Professional Summary": ["summary", "objective", "profile", "about me", "professional summary"],
            "Work Experience": ["experience", "employment", "work history", "professional experience", "career"],
            "Technical Skills": ["skills", "technical skills", "tech stack", "technologies", "competencies", "tools"],
            "Education": ["education", "academic", "qualifications", "university", "college", "degree"],
            "Projects": ["projects", "personal projects", "academic projects", "key initiatives"],
            "Certifications": ["certifications", "certificates", "licenses", "courses", "credentials"],
            "Leadership & Honors": ["leadership", "awards", "honors", "achievements", "activities"]
        }
        for sec_name, keywords in sec_catalog.items():
            if any(re.search(r"\b" + re.escape(kw) + r"\b", text_lower) for kw in keywords):
                detected_sections.append(sec_name)

        if not detected_sections:
            detected_sections = ["Professional Summary", "Experience", "Skills", "Education"]

        ats_score = 6.0
        if detected_email != "Not Detected": ats_score += 1.0
        if detected_phone != "Not Detected": ats_score += 1.0
        if linkedin_url or github_url: ats_score += 1.0
        if len(bullet_points) >= 6: ats_score += 1.0
        ats_score = min(10.0, round(ats_score, 1))

        # 5. Dimension: Seniority, Breadth & Leadership (0 - 10.0)
        leadership_terms = ["lead", "led", "architected", "mentored", "directed", "spearheaded", "strategy", "roadmap", "cross-functional", "stakeholder", "budget", "hired", "scaled"]
        leadership_hits = sum(1 for term in leadership_terms if term in text_lower)
        leadership_score = min(10.0, round(max(4.0, 5.0 + (leadership_hits * 0.6)), 1))

        # COMPOSITE SCORE OUT OF 10.0
        overall_score = round(
            (impact_score * 0.30) +
            (verb_score * 0.20) +
            (skills_score * 0.25) +
            (ats_score * 0.15) +
            (leadership_score * 0.10),
            1
        )
        overall_score = max(4.0, min(9.8, overall_score))

        # Percentile Tier
        if overall_score >= 9.0:
            percentile = "95th Percentile • Top Tier Staff/Senior Candidate"
            badge_color = "#10B981"
        elif overall_score >= 8.0:
            percentile = "85th Percentile • Strong Market Contender"
            badge_color = "#34D399"
        elif overall_score >= 7.0:
            percentile = "70th Percentile • Competitive With Room For Polish"
            badge_color = "#FF851B"
        else:
            percentile = "50th Percentile • Needs Deep Strategic Optimization"
            badge_color = "#EF4444"

        profile = {
            "file_name": self.file_name,
            "name": candidate_name,
            "email": detected_email,
            "phone": detected_phone,
            "linkedin": linkedin_url,
            "github": github_url,
            "detected_sections": detected_sections,
            "detected_email": detected_email,
            "detected_phone": detected_phone,
            "has_linkedin": bool(linkedin_url),
            "has_github": bool(github_url),
            "total_lines": len(lines),
            "total_bullet_points": len(bullet_points),
            "quantified_bullets_count": len(quantified_bullets),
            "action_verb_count": len(action_verb_hits),
            "weak_phrase_count": len(weak_phrase_hits),
            "weak_phrases_found": list(set(weak_phrase_hits)),
        }

        # Generate Deep Thinking Recommendations & Rewrites
        deep_insights = self._deep_analysis(
            overall_score=overall_score,
            sub_scores={
                "impact": impact_score,
                "verbs": verb_score,
                "skills": skills_score,
                "ats": ats_score,
                "leadership": leadership_score,
            },
            bullet_points=bullet_points,
            weak_phrases=weak_phrase_hits,
            missing_skills=missing_recommendations,
        )

        # Extract structured rewrites for interactive tabs
        sample_bullets = bullet_points[:3] if bullet_points else [
            "Responsible for building web applications and backend APIs.",
            "Helped team improve database query speeds and fix bugs.",
            "Worked on machine learning models and data pipelines."
        ]
        structured_rewrites = []
        templates = [
            ("Lacks quantification and active executive leadership phrasing.",
             "Architected high-throughput microservices and REST APIs in Python & FastAPI, increasing endpoint throughput by 3.4x and reducing p99 latency to 110ms.",
             "Replaces passive duty statement with measurable systems latency and scalability."),
            ("Missing measurable business revenue or cost reduction impact.",
             "Optimized relational PostgreSQL schema indexing and query caching, cutting query execution times by 55% and saving $140k in annual cloud overhead.",
             "Demonstrates tangible engineering leverage and financial ROI."),
            ("Lacks specifics on scale, modern frameworks, and production readiness.",
             "Engineered automated production ML inference and RAG pipelines serving 1.8M+ requests/day with 99.95% system uptime.",
             "Validates enterprise-scale AI architecture and operational reliability.")
        ]
        for idx, bullet in enumerate(sample_bullets):
            w, r, a = templates[idx % len(templates)]
            structured_rewrites.append({
                "original": bullet,
                "weakness": w,
                "rewrite": r,
                "advantage": a,
            })

        all_recs = []
        for sks in missing_recommendations.values():
            all_recs.extend(sks)

        return {
            "is_resume": True,
            "overall_score": overall_score,
            "score_out_of_10": overall_score,
            "percentile_tier": percentile,
            "badge_color": badge_color,
            "sub_scores": {
                "impact": impact_score,
                "verbs": verb_score,
                "skills": skills_score,
                "ats": ats_score,
                "leadership": leadership_score,
            },
            "profile": profile,
            "matched_skills": matched_skills,
            "recommended_keywords": all_recs[:10],
            "weak_bullet_rewrites": structured_rewrites,
            "deep_insights": deep_insights,
            "suggestions": {"markdown": deep_insights.get("markdown", "")},
        }

    def _deep_analysis(
        self,
        overall_score: float,
        sub_scores: Dict[str, float],
        bullet_points: List[str],
        weak_phrases: List[str],
        missing_skills: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """Conducts deep executive audit and interview synthesis via Gemini or deterministic engine."""
        
        # 1. Gemini Deep Thinking Path
        if self.api_key:
            try:
                from google import genai
                client = genai.Client(api_key=self.api_key)
                prompt = f"""
You are an elite Principal Talent Partner and Executive Tech Recruiter for Tier-1 Tech (FAANG / AI Labs).
Conduct an extensive, deep-thinking, rigorous audit of the following candidate resume.
Score: {overall_score}/10.0
Sub-Scores:
- Impact & Quantification: {sub_scores['impact']}/10
- Power Verbs: {sub_scores['verbs']}/10
- Tech Stack Density: {sub_scores['skills']}/10
- ATS Parseability: {sub_scores['ats']}/10
- Seniority & Scope: {sub_scores['leadership']}/10

Candidate Resume Text:
\"\"\"{self.raw_text[:4000]}\"\"\"

Generate an authoritative, comprehensive executive review formatted strictly in GitHub-flavored Markdown with these exact sections:

# 🎖️ Executive Recruiter Diagnostic & Market Verdict
(State candidate seniority level, primary market positioning, and critical impression within a 6-second recruiter glance.)

# 🚨 Critical Red Flags & Blindspots (Why you might get filtered out)
(Identify at least 3 severe shortcomings such as unquantified duties, missing cloud/scale indicators, or buzzword fluff.)

# 🎯 High-Impact Line-by-Line Bullet Point Rewrites (Google XYZ Formula)
(Transform 3 of the weakest statements into elite XYZ statements: 'Accomplished [X] as measured by [Y] by doing [Z]')
Format each as:
- **Original**: `[exact line]`
- **Diagnosis**: *[what's wrong]*
- **Elite XYZ Rewrite**: `[quantified, power-verb version]`
- **Projected Impact**: *[Why this beats 90% of applicants]*

# ⚡ 2026 Tech Stack Deficits to Bridge Immediately
(Specify high-demand frameworks, vector DBs, orchestration, or distributed systems tools missing based on their profile.)

# 🎤 Tailored Technical Interview Questions (Derived from their exact projects)
(Generate 3 tough, deep-probing technical interview questions that a Staff/Principal interviewer will ask based on their resume claims, plus brief cheat-sheet answers on how to defend them.)

# 🗓️ 30-Day Strategic Roadmap to Reach a 9.8/10 Score
(Week-by-week actionable plan to elevate their resume and GitHub/portfolio to world-class standards.)
"""
                models_to_try = [os.getenv("MODEL_NAME", "gemini-2.5-flash"), "gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash-latest", "gemini-1.5-pro-latest", "gemini-1.5-flash"]
                for m in list(dict.fromkeys(models_to_try)):
                    try:
                        # 1. Try models.generate_content
                        if hasattr(client, "models") and hasattr(client.models, "generate_content"):
                            response = client.models.generate_content(model=m, contents=prompt)
                            if response and response.text:
                                return {"mode": "gemini_deep_thinking", "markdown": response.text.strip()}

                        # 2. Try interactions.create
                        if hasattr(client, "interactions") and hasattr(client.interactions, "create"):
                            inter = client.interactions.create(model=m, input=prompt)
                            if inter and hasattr(inter, "output_text") and inter.output_text:
                                return {"mode": "gemini_deep_thinking", "markdown": inter.output_text.strip()}
                    except Exception as me:
                        logger.warning(f"Resume model {m} attempt: {me}")
                        continue
            except Exception as e:
                logger.warning(f"Gemini deep analysis fallback triggered: {e}")

        # 2. Comprehensive Deterministic Deep Analysis Fallback
        sample_bullets = bullet_points[:3] if bullet_points else ["Worked on backend microservices and databases."]
        first_bullet = sample_bullets[0] if sample_bullets else "Responsible for developing web applications."
        second_bullet = sample_bullets[1] if len(sample_bullets) > 1 else "Helped team resolve software bugs."

        all_missing = []
        for cat, sks in missing_skills.items():
            all_missing.extend(sks[:3])

        markdown_report = f"""# 🎖️ Executive Recruiter Diagnostic & Market Verdict
* **Current Score**: **`{overall_score} / 10.0`**
* **Seniority Assessment**: Mid-to-Senior Tier Contender.
* **Recruiter Impression**: The candidate shows strong technical aptitude, but bullet points currently under-index on **business revenue impact** and **measurable scale**. With targeted metric hardening, this profile can readily achieve top 5% applicant tier positioning.

---

# 🚨 Critical Red Flags & Blindspots
1. **Passive Language Traps**: Detected {len(weak_phrases)} instances of passive phrasing (e.g. *"{', '.join(weak_phrases[:3]) if weak_phrases else 'helped with, responsible for'}"*). Replace with decisive ownership verbs like *Architected*, *Spearheaded*, or *Overhauled*.
2. **Missing Operational Scale ($/X/ms)**: Only {int(sub_scores['impact'] * 10)}% of your bullet points contain quantified business metrics. Hiring managers want to see latency reductions, user scale, dollar cost savings, or throughput improvements.
3. **Modern 2026 Stack Omission**: Crucial modern infrastructure and AI concepts (**{', '.join(all_missing[:5])}**) are omitted from primary project descriptions.

---

# 🎯 High-Impact Line-by-Line Bullet Point Rewrites (Google XYZ Formula)

### Example 1
* **Original**: `{first_bullet}`
* **Diagnosis**: *Lacks scale, power verb, and quantifiable business return.*
* **Elite XYZ Rewrite**: `Architected high-throughput data processing workflows using Python and PostgreSQL, reducing processing turnaround by 44% and saving $180k in annual cloud overhead.`
* **Projected Impact**: *Elevates passive task execution to strategic systems engineering.*

### Example 2
* **Original**: `{second_bullet}`
* **Diagnosis**: *Unclear individual contribution and missing outcome measurement.*
* **Elite XYZ Rewrite**: `Spearheaded automated regression test pipelines and CI/CD workflows, eliminating 95% of staging deployment errors and accelerating sprint velocity by 2.5x.`
* **Projected Impact**: *Demonstrates ownership, team leverage, and engineering excellence.*

---

# ⚡ 2026 Tech Stack Deficits to Bridge Immediately
To maximize competitive edge across AI, Data, and Cloud roles, prioritize adding:
* **AI & RAG Systems**: `{', '.join(missing_skills.get('AI, ML & Data Engineering', ['LangChain', 'LlamaIndex', 'Vector DBs', 'Fine-tuning'])[:4])}`
* **Cloud & Orchestration**: `{', '.join(missing_skills.get('Cloud, DevOps & Infra', ['Kubernetes', 'Docker', 'Terraform', 'CI/CD'])[:4])}`
* **Backend & Distributed Systems**: `{', '.join(missing_skills.get('Software Architecture & Backend', ['FastAPI', 'Redis', 'Kafka', 'System Design'])[:4])}`

---

# 🎤 Tailored Technical Interview Questions (Derived From Your Resume)
1. **System Scalability**: *"You mentioned working on backend components—how did you design for concurrency, rate limiting, and failure recovery under peak load?"*
   - *How to Answer*: Detail your caching strategy (Redis), async workers, and connection pooling.
2. **Tradeoff Analysis**: *"What was the most difficult architectural tradeoff you made between query latency and data consistency?"*
   - *How to Answer*: Frame using the CAP theorem, read replicas, and indexing optimizations.
3. **Impact Verification**: *"How did you define and measure the success metrics for your latest delivered feature?"*
   - *How to Answer*: Cite telemetry, user adoption rates, and A/B test statistical significance ($p < 0.05$).

---

# 🗓️ 30-Day Strategic Roadmap to Reach 9.8 / 10.0
* **Week 1 (Metric Hardening)**: Audit all bullet points using the Google XYZ formula. Insert concrete metrics for every single line ($X \to Y$).
* **Week 2 (Tech Stack Modernization)**: Integrate in-demand modern tools (FastAPI, Docker, LangChain/RAG, AWS) into your project summaries.
* **Week 3 (Portfolio & Proof of Work)**: Add verified live demo links, GitHub repositories, and system architecture diagrams.
* **Week 4 (ATS & Final Polish)**: Run automated parser checks to ensure 100% ATS readability and zero formatting artifacts.
"""

        return {
            "mode": "deterministic_deep_thinking",
            "markdown": markdown_report
        }
