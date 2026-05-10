"""
SHL Assessment Recommender — Test Suite

Coverage:
  TestRetriever        — catalog loading, TF-IDF search, type filters, name lookup
  TestAgentBehaviors   — behavior probes matching the evaluator's assertion list
  TestAPISchema        — Pydantic validation edge cases
  TestRecallEvaluation — Mean Recall@10 on known query-answer pairs

Run: python tests/test_all.py
"""

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

CATALOG_PATH = str(Path(__file__).parent.parent / "catalog" / "catalog.json")


# ─── helpers ──────────────────────────────────────────────────────────────────

def _mock_llm_response(reply: str, recs: list = None, end: bool = False) -> MagicMock:
    """Build a mock Gemini response object with embedded JSON block."""
    recs = recs or []
    payload = json.dumps({"recommendations": recs, "end_of_conversation": end})
    if recs:
        full_text = f"{reply}\n```json\n{payload}\n```"
    else:
        full_text = f"{reply}\n```json\n{payload}\n```"
    mock_resp = MagicMock()
    mock_resp.text = full_text
    return mock_resp


def _run_agent(messages, retriever, reply="Here are some assessments.", recs=None):
    """Run agent with the LLM mocked out."""
    from agent.agent import run_agent

    mock_resp = _mock_llm_response(reply, recs or [])
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_resp

    with patch("agent.agent.get_model", return_value=mock_client), \
         patch("agent.agent.get_client", return_value=mock_client):
        return run_agent(messages, retriever)


# ─── Retriever Tests ──────────────────────────────────────────────────────────

class TestRetriever(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from catalog.retriever import CatalogRetriever
        cls.r = CatalogRetriever(CATALOG_PATH)

    # ── catalog integrity ──────────────────────────────────────────────────

    def test_catalog_loaded_minimum_size(self):
        self.assertGreater(len(self.r.assessments), 50,
                           "Catalog must have at least 50 assessments")

    def test_all_assessments_have_required_fields(self):
        for a in self.r.assessments:
            for field in ("name", "url", "test_types"):
                self.assertIn(field, a, f"Missing field '{field}' in: {a}")

    def test_all_urls_start_with_shl_domain(self):
        for a in self.r.assessments:
            self.assertTrue(
                a["url"].startswith("https://www.shl.com/"),
                f"Non-SHL URL found: {a['url']}",
            )

    def test_no_duplicate_urls(self):
        urls = [a["url"] for a in self.r.assessments]
        self.assertEqual(len(urls), len(set(urls)), "Catalog contains duplicate URLs")

    # ── search correctness ─────────────────────────────────────────────────

    def test_search_java_developer_returns_java_assessment(self):
        results = self.r.search("Java developer technical skills", k=5)
        self.assertGreater(len(results), 0)
        names_lower = [r["name"].lower() for r in results]
        self.assertTrue(
            any("java" in n for n in names_lower),
            f"No Java assessment in top results: {[r['name'] for r in results]}",
        )

    def test_search_personality_returns_p_type(self):
        results = self.r.search("personality behaviour assessment", k=5)
        types = {t for r in results for t in r["test_types"]}
        self.assertIn("P", types, "Personality search must return P-type results")

    def test_search_type_filter_boosts_matching_types(self):
        results = self.r.search("developer skills", k=5, test_type_filter=["K"])
        # With type boost, at least 3 of top 5 should be K
        k_count = sum(1 for r in results if "K" in r["test_types"])
        self.assertGreaterEqual(k_count, 3,
                                f"Type filter should boost K results; got {k_count}/5")

    def test_search_respects_k_limit(self):
        results = self.r.search("assessment", k=10)
        self.assertLessEqual(len(results), 10)

    def test_search_k_1_returns_single_result(self):
        results = self.r.search("numerical reasoning", k=1)
        self.assertEqual(len(results), 1)

    # ── name lookup ────────────────────────────────────────────────────────

    def test_get_by_name_exact_match(self):
        result = self.r.get_by_name("OPQ32r")
        self.assertIsNotNone(result, "Should find OPQ32r by exact name")
        self.assertEqual(result["name"], "OPQ32r")

    def test_get_by_name_case_insensitive(self):
        result = self.r.get_by_name("opq32r")
        self.assertIsNotNone(result, "Name lookup should be case-insensitive")

    def test_get_by_name_partial_match(self):
        result = self.r.get_by_name("Numerical Reasoning")
        self.assertIsNotNone(result, "Partial name lookup should succeed")

    def test_get_by_name_unknown_returns_none(self):
        result = self.r.get_by_name("This Assessment Does Not Exist XYZ123")
        self.assertIsNone(result)

    def test_get_all_returns_all_assessments(self):
        self.assertEqual(
            len(self.r.get_all()), len(self.r.assessments)
        )


# ─── Agent Behavior Probes ────────────────────────────────────────────────────

class TestAgentBehaviors(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from catalog.retriever import CatalogRetriever
        cls.r = CatalogRetriever(CATALOG_PATH)
        cls.catalog_urls = {a["url"] for a in cls.r.assessments}

    # ── probe 1: no recs on first vague turn ───────────────────────────────

    def test_no_recommendations_on_bare_vague_query(self):
        """'I need an assessment' is vague — agent must ask, not recommend."""
        msgs = [{"role": "user", "content": "I need an assessment"}]
        result = _run_agent(msgs, self.r, reply="What role are you hiring for?")
        self.assertEqual(result["recommendations"], [],
                         "Vague query must not produce recommendations")

    def test_no_recommendations_on_single_word_query(self):
        msgs = [{"role": "user", "content": "hello"}]
        result = _run_agent(msgs, self.r, reply="How can I help you today?")
        self.assertEqual(result["recommendations"], [])

    # ── probe 2: recommends with sufficient context ────────────────────────

    def test_recommends_after_role_and_level_provided(self):
        recs = [
            {"name": "Java 8 (New)",
             "url": "https://www.shl.com/products/product-catalog/view/java-8-new/",
             "test_type": "K"},
            {"name": "OPQ32r",
             "url": "https://www.shl.com/products/product-catalog/view/opq32r/",
             "test_type": "P"},
        ]
        msgs = [
            {"role": "user", "content": "Hiring a mid-level Java developer who works with stakeholders"},
            {"role": "assistant", "content": "What seniority level?"},
            {"role": "user", "content": "Mid-level, around 4 years experience"},
        ]
        result = _run_agent(msgs, self.r, reply="Here are 2 assessments.", recs=recs)
        self.assertGreater(len(result["recommendations"]), 0,
                           "Should recommend after role + level provided")

    # ── probe 3: all URLs from catalog only ───────────────────────────────

    def test_all_returned_urls_in_catalog(self):
        recs = [
            {"name": "Java 8 (New)",
             "url": "https://www.shl.com/products/product-catalog/view/java-8-new/",
             "test_type": "K"},
        ]
        msgs = [{"role": "user", "content": "Hiring a senior Java engineer"}]
        result = _run_agent(msgs, self.r, recs=recs)
        for rec in result["recommendations"]:
            self.assertIn(rec["url"], self.catalog_urls,
                          f"URL not in catalog: {rec['url']}")

    # ── probe 4: max 10 recommendations ──────────────────────────────────

    def test_maximum_10_recommendations_enforced(self):
        recs = [
            {"name": "Java 8 (New)",
             "url": "https://www.shl.com/products/product-catalog/view/java-8-new/",
             "test_type": "K"}
        ] * 15
        msgs = [{"role": "user", "content": "Give me every Java assessment available"}]
        result = _run_agent(msgs, self.r, recs=recs)
        self.assertLessEqual(len(result["recommendations"]), 10)

    # ── probe 5: hallucinated URL filtered ───────────────────────────────

    def test_hallucinated_url_is_filtered_out(self):
        recs = [
            {"name": "Fake Hallucinated Tool",
             "url": "https://www.shl.com/not-a-real-assessment/",
             "test_type": "K"},
            {"name": "Java 8 (New)",
             "url": "https://www.shl.com/products/product-catalog/view/java-8-new/",
             "test_type": "K"},
        ]
        msgs = [{"role": "user", "content": "Java developer assessment"}]
        result = _run_agent(msgs, self.r, recs=recs)
        urls = [r["url"] for r in result["recommendations"]]
        self.assertNotIn("https://www.shl.com/not-a-real-assessment/", urls,
                         "Hallucinated URL must be filtered")
        self.assertIn(
            "https://www.shl.com/products/product-catalog/view/java-8-new/",
            urls,
            "Valid URL should survive filtering",
        )

    # ── probe 6: schema keys always present ──────────────────────────────

    def test_schema_keys_always_present(self):
        for query in [
            "What is the weather today?",
            "Tell me a joke",
            "I need an assessment",
        ]:
            msgs = [{"role": "user", "content": query}]
            result = _run_agent(msgs, self.r,
                                reply="I can only help with SHL assessments.")
            self.assertIn("reply", result)
            self.assertIn("recommendations", result)
            self.assertIn("end_of_conversation", result)
            self.assertIsInstance(result["reply"], str)
            self.assertIsInstance(result["recommendations"], list)
            self.assertIsInstance(result["end_of_conversation"], bool)

    # ── probe 7: graceful empty input ────────────────────────────────────

    def test_empty_messages_returns_greeting(self):
        from agent.agent import run_agent
        result = run_agent([], self.r)
        self.assertIn("reply", result)
        self.assertGreater(len(result["reply"]), 0)
        self.assertEqual(result["recommendations"], [])

    # ── probe 8: recommendations deduplicated ─────────────────────────────

    def test_duplicate_urls_deduplicated(self):
        recs = [
            {"name": "Java 8 (New)",
             "url": "https://www.shl.com/products/product-catalog/view/java-8-new/",
             "test_type": "K"},
            {"name": "Java 8 (New)",
             "url": "https://www.shl.com/products/product-catalog/view/java-8-new/",
             "test_type": "K"},
        ]
        msgs = [{"role": "user", "content": "Java developer mid-level"}]
        result = _run_agent(msgs, self.r, recs=recs)
        urls = [r["url"] for r in result["recommendations"]]
        self.assertEqual(len(urls), len(set(urls)), "Duplicate URLs must be removed")

    # ── probe 9: refine mid-conversation keeps context ────────────────────

    def test_refine_updates_shortlist_not_restart(self):
        """Agent with 3-turn history should still recommend (not re-clarify)."""
        recs = [
            {"name": "OPQ32r",
             "url": "https://www.shl.com/products/product-catalog/view/opq32r/",
             "test_type": "P"},
        ]
        msgs = [
            {"role": "user", "content": "Hiring a sales manager"},
            {"role": "assistant", "content": "Here are some assessments…"},
            {"role": "user", "content": "Actually, also add personality tests"},
        ]
        result = _run_agent(msgs, self.r, reply="Updated list with personality.", recs=recs)
        self.assertGreater(len(result["recommendations"]), 0,
                           "Refine turn should still produce recommendations")


# ─── FastAPI Schema Tests ──────────────────────────────────────────────────────

class TestAPISchema(unittest.TestCase):

    def test_valid_single_user_message(self):
        from api.main import ChatRequest
        req = ChatRequest(messages=[{"role": "user", "content": "Hiring a developer"}])
        self.assertEqual(len(req.messages), 1)

    def test_empty_messages_rejected(self):
        from api.main import ChatRequest
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            ChatRequest(messages=[])

    def test_messages_exceeding_limit_rejected(self):
        from api.main import ChatRequest
        from pydantic import ValidationError
        msgs = [{"role": "user", "content": f"msg {i}"} for i in range(21)]
        with self.assertRaises(ValidationError):
            ChatRequest(messages=msgs)

    def test_last_message_must_be_user(self):
        from api.main import ChatRequest
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            ChatRequest(messages=[
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
            ])

    def test_empty_content_rejected(self):
        from api.main import ChatRequest
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            ChatRequest(messages=[{"role": "user", "content": "   "}])

    def test_whitespace_content_stripped(self):
        from api.main import ChatRequest
        req = ChatRequest(messages=[{"role": "user", "content": "  hello  "}])
        self.assertEqual(req.messages[0].content, "hello")

    def test_invalid_role_rejected(self):
        from api.main import ChatRequest
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            ChatRequest(messages=[{"role": "system", "content": "Hi"}])


# ─── Recall Evaluation ────────────────────────────────────────────────────────

class TestRecallEvaluation(unittest.TestCase):
    """
    Evaluation on known query-answer pairs.
    Mean Recall@10 must exceed 0.5 (we target >0.8).
    """

    KNOWN_PAIRS = [
        (
            "numerical reasoning test for finance analyst",
            ["Numerical Reasoning",
             "Verify - Numerical Reasoning (Interactive)",
             "Verify G+ (Interactive)"],
        ),
        (
            "personality and behaviour assessment for all seniority levels",
            ["Occupational Personality Questionnaire (OPQ32)", "OPQ32r"],
        ),
        (
            "Python developer coding skills technical test",
            ["Python (New)", "Coding Pro - Advanced Python"],
        ),
        (
            "entry level customer service representative",
            ["Customer Service - Short Form", "Entry Level - Short Form"],
        ),
        (
            "leadership development 360 degree feedback",
            ["360 Feedback", "Workplace Effectiveness Survey (WES)"],
        ),
        (
            "verbal reasoning critical thinking graduate scheme",
            ["Verbal Reasoning", "Verify - Verbal Ability (Interactive)"],
        ),
        (
            "sales executive personality motivation",
            ["OPQ32r", "Motivational Questionnaire (MQ)"],
        ),
    ]

    @classmethod
    def setUpClass(cls):
        from catalog.retriever import CatalogRetriever
        cls.r = CatalogRetriever(CATALOG_PATH)

    def test_recall_on_known_pairs(self):
        total_recall = 0.0
        n = len(self.KNOWN_PAIRS)
        print(f"\n{'Query':<50} {'Recall':>6}  Hits")
        print("─" * 70)
        for query, expected in self.KNOWN_PAIRS:
            results = self.r.search(query, k=10)
            result_names = {r["name"] for r in results}
            hits = sum(1 for e in expected if e in result_names)
            recall = hits / len(expected)
            total_recall += recall
            print(f"{query[:49]:<50} {recall:>6.2f}  {hits}/{len(expected)}")

        mean_recall = total_recall / n
        print(f"\nMean Recall@10: {mean_recall:.3f}")
        self.assertGreater(
            mean_recall, 0.5,
            f"Mean Recall@10 too low: {mean_recall:.3f} (target > 0.5)",
        )


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("SHL Assessment Recommender — Test Suite")
    print("=" * 70)
    unittest.main(verbosity=2)
