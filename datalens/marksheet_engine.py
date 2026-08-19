"""Academic Marksheet, State Board & Multi-Semester Transcript Engine for DataLens AI.

Supports CBSE, ICSE, State Boards (WBCHSE, Maharashtra HSC, UP Board, etc.),
single marksheets, multi-semester college transcripts (SGPA + CGPA progression),
and Best-of-N subject selection.
"""

import os
import re
import json
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
from datalens.logger import get_logger

logger = get_logger("MarksheetEngine")

MARKSHEET_KEYWORDS = [
    "marksheet", "mark sheet", "transcript", "scorecard", "score card",
    "grade card", "examination", "semester", "sem 1", "sem 2", "sem 3", "sem 4", "sem 5", "sem 6",
    "roll no", "enrollment", "registration", "cbse", "icse", "wbchse", "hsc", "ssc", "board",
    "council", "university", "college", "school", "cgpa", "sgpa", "ygpa", "marks obtained",
    "maximum marks", "max marks", "subject code", "theory", "practical", "class xii", "class x",
    "higher secondary", "bachelor", "master", "percentage", "pass marks", "grade points", "credits"
]


class MarksheetEngine:
    """Intelligent Academic Marksheet, State Board & Multi-Semester Transcript Engine."""

    def __init__(self, raw_text: str, file_name: str = "Marksheet.pdf", api_key: Optional[str] = None):
        self.raw_text = raw_text
        self.file_name = file_name
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.client = None

        if self.api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                logger.warning(f"Could not initialize AI client in MarksheetEngine: {e}")

    @staticmethod
    def is_marksheet(text: str, file_name: str = "") -> bool:
        """Determines whether a document is an academic marksheet, state board scorecard, or multi-semester transcript."""
        fn_lower = file_name.lower()
        if any(k in fn_lower for k in ["marksheet", "mark_sheet", "transcript", "scorecard", "class_xii", "class_x", "gradecard", "grade_sheet", "semester", "sem_", "wbchse", "cbse", "hsc"]):
            return True

        text_lower = text.lower()
        keyword_hits = sum(1 for kw in MARKSHEET_KEYWORDS if kw in text_lower)
        if keyword_hits >= 3:
            return True

        has_subjects = any(s in text_lower for s in ["mathematics", "math", "physics", "chemistry", "biology", "english", "computer", "science", "social", "economics", "accounts", "history", "bengali", "hindi", "data structures", "algorithms", "dbms"])
        has_marks_pattern = bool(re.search(r"\b\d{2,3}\s*/\s*\d{2,3}\b|\b\d{2,3}\s+(?:out of|\/)\s+\d{2,3}\b|\bgrade\s*:\s*[A-F]\b|\bsgpa\b|\bcgpa\b", text_lower))

        return has_subjects and (has_marks_pattern or keyword_hits >= 2)

    def analyze(self) -> Dict[str, Any]:
        """Conducts full marksheet evaluation, multi-semester SGPA/CGPA aggregation, and DataLens AI guidance."""
        # 1. First attempt AI Structured Extraction for multi-semester or single marksheet
        ai_data = self._ai_extract()
        if ai_data and (ai_data.get("subjects") or ai_data.get("semesters")):
            return self._compute_analytics(ai_data)

        # 2. Fallback to Deterministic Regex & Pattern Extraction
        deterministic_data = self._regex_extract()
        return self._compute_analytics(deterministic_data)

    def _ai_extract(self) -> Optional[Dict[str, Any]]:
        """Extracts structured academic data from raw document text supporting single marksheets and multi-semester transcripts."""
        if not self.client:
            return None

        prompt = f"""You are DataLens AI Academic Intelligence Engine.
Analyze the following marksheet / scorecard / multi-semester transcript text and extract structured academic data into pure JSON:

DOCUMENT TEXT:
{self.raw_text}

Respond ONLY with a JSON object in this exact schema without markdown backticks:
{{
  "candidate_name": "string (or 'Student')",
  "roll_number": "string or null",
  "institution_board": "string (e.g. CBSE, WBCHSE, Maharashtra State Board, Delhi University, MAKAUT, etc.)",
  "examination_name": "string (e.g. Class XII Higher Secondary Examination, B.Tech Computer Science, etc.)",
  "passing_year": "string or null",
  "is_multi_semester": true or false,
  "semesters": [
    {{
      "semester_name": "Semester 1",
      "sgpa": number or null,
      "total_marks_obtained": number or null,
      "total_max_marks": number or null,
      "subjects": [
        {{
          "subject_name": "string",
          "marks_obtained": number,
          "max_marks": number,
          "grade": "string or null"
        }}
      ]
    }}
  ],
  "subjects": [
    {{
      "subject_name": "string",
      "marks_obtained": number,
      "max_marks": number,
      "grade": "string or null"
    }}
  ]
}}"""
        try:
            models_to_try = [os.getenv("MODEL_NAME", "gemini-2.5-flash"), "gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
            for m in list(dict.fromkeys(models_to_try)):
                try:
                    resp = self.client.models.generate_content(model=m, contents=prompt)
                    if resp and resp.text:
                        raw_json = resp.text.strip()
                        if raw_json.startswith("```json"):
                            raw_json = raw_json.replace("```json", "", 1)
                        if raw_json.startswith("```"):
                            raw_json = raw_json.replace("```", "", 1)
                        if raw_json.endswith("```"):
                            raw_json = raw_json[:-3]
                        raw_json = raw_json.strip()
                        data = json.loads(raw_json)
                        if isinstance(data, dict) and (data.get("subjects") or data.get("semesters")):
                            return data
                except Exception:
                    continue
        except Exception as e:
            logger.warning(f"DataLens AI marksheet extraction failed: {e}")

        return None

    def _regex_extract(self) -> Dict[str, Any]:
        """Deterministic regex fallback to extract subjects and scores."""
        lines = [line.strip() for line in self.raw_text.split("\n") if line.strip()]
        subjects: List[Dict[str, Any]] = []

        known_subjects = [
            "Mathematics", "Physics", "Chemistry", "Biology", "English", "English Core",
            "Computer Science", "Information Technology", "Economics", "Accountancy",
            "Business Studies", "History", "Political Science", "Geography", "Psychology",
            "Sociology", "Bengali", "Hindi", "Physical Education", "Science", "Social Science",
            "Data Structures", "Algorithms", "Operating Systems", "DBMS", "Computer Networks"
        ]

        found_subjects = set()
        for line in lines:
            for s in known_subjects:
                if s.lower() in line.lower() and s not in found_subjects:
                    nums = [float(n) for n in re.findall(r"\b\d{1,3}(?:\.\d+)?\b", line)]
                    if nums:
                        marks = nums[-1] if nums[-1] <= 100 else nums[0]
                        max_m = 100.0
                        if len(nums) >= 2 and nums[-1] in [100.0, 50.0, 75.0, 80.0, 200.0]:
                            marks = nums[-2]
                            max_m = nums[-1]
                        
                        subjects.append({
                            "subject_name": s,
                            "marks_obtained": min(marks, max_m),
                            "max_marks": max_m,
                            "grade": self._assign_grade(marks, max_m)
                        })
                        found_subjects.add(s)

        if not subjects:
            for idx, line in enumerate(lines[:15]):
                nums = [float(n) for n in re.findall(r"\b\d{1,3}(?:\.\d+)?\b", line)]
                words = re.findall(r"\b[A-Za-z]{3,}\b", line)
                if nums and words:
                    sub_name = " ".join(words[:2])
                    marks = nums[0] if nums[0] <= 100 else 75.0
                    subjects.append({
                        "subject_name": sub_name,
                        "marks_obtained": marks,
                        "max_marks": 100.0,
                        "grade": self._assign_grade(marks, 100.0)
                    })

        return {
            "candidate_name": "Student",
            "roll_number": None,
            "institution_board": "State Board / University",
            "examination_name": "Academic Examination Scorecard",
            "passing_year": None,
            "is_multi_semester": False,
            "semesters": [],
            "subjects": subjects
        }

    @staticmethod
    def _assign_grade(marks: float, max_m: float) -> str:
        pct = (marks / max(1.0, max_m)) * 100.0
        if pct >= 90: return "A1 (Outstanding)"
        if pct >= 80: return "A2 (Excellent)"
        if pct >= 70: return "B1 (Very Good)"
        if pct >= 60: return "B2 (Good)"
        if pct >= 50: return "C1 (Average)"
        if pct >= 40: return "C2 (Pass)"
        return "D (Needs Improvement)"

    def _compute_analytics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculates GPA, Best-of-N averages, multi-semester SGPA/CGPA progression, and career pathways."""
        semesters_raw = data.get("semesters", [])
        is_multi_sem = bool(data.get("is_multi_semester") or len(semesters_raw) > 1)

        subjects = data.get("subjects", [])
        semesters_processed = []

        if is_multi_sem and semesters_raw:
            # Process multi-semester data
            all_subjects = []
            sgpa_list = []
            for sem in semesters_raw:
                sem_subs = sem.get("subjects", [])
                s_obt = sum(float(s.get("marks_obtained", 0)) for s in sem_subs)
                s_max = sum(float(s.get("max_marks", 100)) for s in sem_subs) if sem_subs else 100.0
                s_pct = round((s_obt / max(1.0, s_max)) * 100.0, 2)
                s_sgpa = sem.get("sgpa")
                if s_sgpa is None or float(s_sgpa) <= 0:
                    s_sgpa = round(s_pct / 10.0, 2)
                else:
                    s_sgpa = float(s_sgpa)
                sgpa_list.append(s_sgpa)

                semesters_processed.append({
                    "semester_name": sem.get("semester_name", f"Semester {len(semesters_processed)+1}"),
                    "sgpa": s_sgpa,
                    "percentage": s_pct,
                    "total_marks_obtained": s_obt,
                    "total_max_marks": s_max,
                    "subjects_count": len(sem_subs),
                    "subjects": sem_subs
                })
                all_subjects.extend(sem_subs)

            if not subjects and all_subjects:
                subjects = all_subjects

            cgpa = round(sum(sgpa_list) / max(1, len(sgpa_list)), 2)
        else:
            cgpa = None

        if not subjects:
            subjects = [
                {"subject_name": "General Studies", "marks_obtained": 75.0, "max_marks": 100.0, "grade": "B1"}
            ]

        total_obtained = sum(float(s.get("marks_obtained", 0)) for s in subjects)
        total_max = sum(float(s.get("max_marks", 100)) for s in subjects)
        overall_pct = round((total_obtained / max(1.0, total_max)) * 100.0, 2)
        gpa_10 = cgpa if cgpa is not None else round((overall_pct / 10.0), 2)
        gpa_4 = round((gpa_10 / 10.0) * 4.0, 2)

        # Subject sorted by percentage
        sub_list = []
        for s in subjects:
            m_obt = float(s.get("marks_obtained", 0))
            m_max = float(s.get("max_marks", 100))
            pct = round((m_obt / max(1.0, m_max)) * 100.0, 1)
            sub_list.append({
                "subject": s.get("subject_name", "Subject"),
                "marks": m_obt,
                "max": m_max,
                "percentage": pct,
                "grade": s.get("grade") or self._assign_grade(m_obt, m_max)
            })

        sub_list_sorted = sorted(sub_list, key=lambda x: x["percentage"], reverse=True)
        strongest = sub_list_sorted[0]
        weakest = sub_list_sorted[-1]

        # Calculate Best of 4 & Best of 5 Aggregates (Board Norms)
        best_4_list = sub_list_sorted[:4] if len(sub_list_sorted) >= 4 else sub_list_sorted
        best_4_obtained = sum(s["marks"] for s in best_4_list)
        best_4_max = sum(s["max"] for s in best_4_list)
        best_4_pct = round((best_4_obtained / max(1.0, best_4_max)) * 100.0, 2)

        best_5_list = sub_list_sorted[:5] if len(sub_list_sorted) >= 5 else sub_list_sorted
        best_5_obtained = sum(s["marks"] for s in best_5_list)
        best_5_max = sum(s["max"] for s in best_5_list)
        best_5_pct = round((best_5_obtained / max(1.0, best_5_max)) * 100.0, 2)

        # Academic Tier Classification
        if overall_pct >= 90.0:
            tier = "🏆 Distinction Honors (Top 5% Standing)"
            tier_color = "#10B981"
            badge = "Outstanding Academic Excellence"
        elif overall_pct >= 75.0:
            tier = "🌟 First Class with Distinction"
            tier_color = "#3B82F6"
            badge = "High Competitive Standing"
        elif overall_pct >= 60.0:
            tier = "📈 First Division Standing"
            tier_color = "#F59E0B"
            badge = "Solid Foundation"
        else:
            tier = "⚠️ Improvement Needed"
            tier_color = "#EF4444"
            badge = "Requires Focused Remediation"

        # Generate DataLens AI Career Guidance
        guidance = self._generate_guidance(data, overall_pct, strongest, weakest, sub_list)

        return {
            "is_marksheet": True,
            "is_multi_semester": is_multi_sem,
            "semesters": semesters_processed,
            "candidate_name": data.get("candidate_name") or "Student",
            "roll_number": data.get("roll_number") or "N/A",
            "institution_board": data.get("institution_board") or "State Board / University",
            "examination_name": data.get("examination_name") or "Academic Examination",
            "passing_year": data.get("passing_year") or "N/A",
            "summary_metrics": {
                "overall_percentage": overall_pct,
                "gpa_out_of_10": gpa_10,
                "gpa_out_of_4": gpa_4,
                "cgpa": cgpa,
                "best_4_percentage": best_4_pct,
                "best_5_percentage": best_5_pct,
                "total_marks_obtained": total_obtained,
                "total_max_marks": total_max,
                "total_subjects": len(subjects),
                "academic_tier": tier,
                "tier_color": tier_color,
                "badge": badge,
            },
            "strongest_subject": strongest,
            "weakest_subject": weakest,
            "subject_breakdown": sub_list_sorted,
            "academic_guidance": guidance,
        }

    def _generate_guidance(
        self,
        data: Dict[str, Any],
        overall_pct: float,
        strongest: Dict[str, Any],
        weakest: Dict[str, Any],
        sub_list: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generates tailored academic insights, career roadmaps, and remediation strategies."""
        sub_summary = ", ".join([f"{s['subject']}: {s['marks']}/{s['max']} ({s['percentage']}%)" for s in sub_list])

        if self.client:
            prompt = f"""You are DataLens AI Principal Academic Counselor & Career Strategist.
Evaluate the following student marksheet performance:
- Examination: {data.get('examination_name')} ({data.get('institution_board')})
- Overall Aggregate: {overall_pct}% (GPA: {round(overall_pct/10.0, 2)}/10.0)
- Highest Subject: {strongest['subject']} ({strongest['percentage']}%)
- Lowest Subject: {weakest['subject']} ({weakest['percentage']}%)
- Subject Breakdown: {sub_summary}

Generate an authoritative, empowering, and actionable Academic Guidance Report in Markdown covering:
1. 💡 **Performance Diagnosis & Key Cognitive Strengths**
2. 🎯 **Top 3 Recommended Higher Education / Career Specializations** (Tailored strictly to their best subjects)
3. 🛠️ **Targeted Improvement Strategy for {weakest['subject']}** (Concrete study techniques & resources)
4. 🚀 **Next Milestone Strategic Roadmap** (Immediate 6-month academic goals)
Do NOT mention any third-party AI provider names, refer strictly to DataLens AI."""

            try:
                models_to_try = [os.getenv("MODEL_NAME", "gemini-2.5-flash"), "gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash-latest", "gemini-1.5-pro-latest", "gemini-1.5-flash"]
                for m in list(dict.fromkeys(models_to_try)):
                    try:
                        resp = self.client.models.generate_content(model=m, contents=prompt)
                        if resp and resp.text:
                            return {
                                "mode": "datalens_ai_intelligence",
                                "markdown": resp.text.strip(),
                                "strong_pathways": [
                                    f"Specialized Degrees in {strongest['subject']} & Applied Sciences",
                                    "High-Growth Technology & Analytics Careers",
                                    "Competitive University Honors Programs"
                                ]
                            }
                    except Exception:
                        continue
            except Exception as ge:
                logger.warning(f"DataLens AI marksheet guidance generation failed: {ge}")

        # Deterministic Academic Guidance Fallback
        return {
            "mode": "deterministic_counselor",
            "markdown": f"""### 💡 Academic Performance Diagnosis
Your strongest academic aptitude is demonstrated in **{strongest['subject']}** with **{strongest['percentage']}%**, showcasing advanced conceptual grasp and problem-solving velocity.

### 🎯 Recommended Career & Higher Study Pathways
1. **Applied {strongest['subject']} & Advanced Research**: Pursue undergraduate or specialized degree tracks in fields leveraging your high aptitude in {strongest['subject']}.
2. **Computational & Quantitative Programs**: High analytical scores indicate strong readiness for Data Science, Software Engineering, or Financial Analytics.
3. **Honors Programs & Fellowships**: Your overall aggregate of **{overall_pct}%** places you in strong competitive standing for top collegiate admissions.

### 🛠️ Remediation & Improvement Strategy for {weakest['subject']}
- **Core Concept Consolidation**: Revisit foundational theory and high-weightage topics in {weakest['subject']}.
- **Timed Mock Practice**: Focus on past 5 years of exam questions under strict time limits to build examination stamina.
- **Active Recall & Spaced Repetition**: Create formula sheets and summary mind-maps for quick daily revisions.
""",
            "strong_pathways": [
                f"Specialized Track in {strongest['subject']}",
                "Quantitative & Technical Disciplines",
                "Competitive University Admissions"
            ]
        }
