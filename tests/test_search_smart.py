import unittest

from search_intent_lexicon import (
    INTENT_NEGATIVE,
    strip_intent_phrases,
)
from search_smart import (
    expand_search_query,
    fuzzy_correct_entities,
    parse_human_price_filters,
    parse_query_intent,
    prepare_search_query,
    suggest_completions,
)


def _search_index_available() -> bool:
    try:
        import search_index  # noqa: F401
    except ImportError:
        return False
    return True


class TestIntentPhraseStrip(unittest.TestCase):
    def test_negative_phrase_removed(self):
        s = strip_intent_phrases("игровой ноутбук без лагов на работу")
        self.assertNotIn("лаг", s.casefold())
        self.assertIn("ноутбук", s.casefold())

    def test_ssd_phrase_removed(self):
        s = strip_intent_phrases("ssd твёрдотельный накопитель 1tb")
        self.assertNotIn("твёрдотельн", s.casefold())
        self.assertIn("накопит", s.casefold())

    def test_intent_negative_alias(self):
        self.assertIn("не тормозит", INTENT_NEGATIVE)


class TestParseHumanPrice(unittest.TestCase):
    def test_do_50k(self):
        mn, mx = parse_human_price_filters("красный телефон до 50к")
        self.assertIsNone(mn)
        self.assertEqual(mx, 50000.0)

    def test_ot_20k(self):
        mn, mx = parse_human_price_filters("от 20к ноутбук")
        self.assertEqual(mn, 20000.0)
        self.assertIsNone(mx)


class TestParseQueryIntent(unittest.TestCase):
    def test_ram_and_price(self):
        p = parse_query_intent("ноутбук 16 ГБ RAM до 80000")
        self.assertEqual(p.ram_gb, 16)
        self.assertEqual(p.price_max, 80000.0)
        self.assertIn("ноутбук", p.clean_q.casefold())

    def test_gb_variants(self):
        p = parse_query_intent("8gb ram смартфон")
        self.assertEqual(p.ram_gb, 8)

    def test_diagonal_parse(self):
        p = parse_query_intent('ноутбук 15.6" для дома')
        self.assertEqual(p.diagonal_inch, 15.6)

    def test_ssd_parse(self):
        p = parse_query_intent("ноутбук 512 гб ssd")
        self.assertEqual(p.ssd_gb, 512)
        self.assertNotIn("512", p.clean_q)


class TestFuzzyCorrect(unittest.TestCase):
    def test_typo_close_to_brand(self):
        brands = ["Samsung", "Sony", "Xiaomi"]
        cats = ["Смартфоны", "Ноутбуки"]
        q = "Samsng Galaxy"
        fixed, changed = fuzzy_correct_entities(q, brands, cats, score_cutoff=88)
        self.assertTrue(changed)
        self.assertIn("Samsung", fixed)


class TestQueryExpansion(unittest.TestCase):
    def test_translit_two_pass(self):
        from search_smart import _apply_word_aliases

        s = _apply_word_aliases("noytbuk для работы")
        self.assertIn("ноутбук", s.casefold())

    def test_phone_synonyms(self):
        e = expand_search_query("купить телефон")
        self.assertIn("смартфон", e.casefold())
        self.assertIn("телефон", e.casefold())

    def test_quiet_synonyms(self):
        e = expand_search_query("тихий пылесос")
        self.assertIn("бесшумн", e.casefold())


class TestSuggestCompletions(unittest.TestCase):
    def test_prefix(self):
        choices = ["Samsung Galaxy S21", "Sony Xperia", "Samsung A52"]
        out = suggest_completions("sam", choices, limit=5)
        self.assertTrue(any("Samsung" in x for x in out))

    def test_short_prefix_no_noise(self):
        choices = ["Samsung Galaxy S21", "Sony Xperia"]
        self.assertEqual(suggest_completions("", choices, limit=5), [])
        self.assertEqual(suggest_completions("s", choices, limit=5), [])


class TestPlanSmartSearchScenarios(unittest.TestCase):
    def test_samsung_galaxy_aliases(self):
        from search_smart import _apply_word_aliases

        aliased = _apply_word_aliases("самсунг галакси")
        low = aliased.casefold()
        self.assertIn("samsung", low)
        self.assertIn("galaxy", low)

    def test_notebook_ram_price_parse(self):
        p = parse_query_intent("ноутбук 16 ГБ RAM до 80000")
        self.assertEqual(p.ram_gb, 16)
        self.assertEqual(p.price_max, 80000.0)


@unittest.skipUnless(_search_index_available(), "search_index dependencies missing")
class TestSearchIndexParseReexport(unittest.TestCase):
    def test_parse_query_intent_delegates(self):
        from search_index import parse_query_intent as pqi_si
        from search_smart import parse_query_intent as pqi_ss

        a = pqi_si("до 30к телефон")
        b = pqi_ss("до 30к телефон")
        self.assertEqual(a.price_max, b.price_max)
        self.assertEqual(a.clean_q, b.clean_q)


@unittest.skipUnless(_search_index_available(), "search_index dependencies missing")
class TestRetrievalFluffAndCamera(unittest.TestCase):
    def test_compact_drops_subjective_good_camera(self):
        from search_index import compact_query_for_retrieval

        c = compact_query_for_retrieval("Смартфоны хорошей камерой")
        low = c.casefold()
        self.assertNotIn("хорош", low)
        self.assertIn("смартфон", low)
        self.assertIn("камер", low)

    def test_tokenize_aligns_with_compact_fluff(self):
        from search_index import tokenize_search_query

        t = tokenize_search_query("смартфоны хорошей камерой")
        self.assertNotIn("хорошей", t)
        self.assertIn("смартфоны", t)
        self.assertIn("камерой", t)

    def test_camera_inflection_matches_card_text(self):
        from search_index import token_matches_in_offer_text

        hay = "xiaomi | смартфоны | основная камера: 48 мп".lower()
        self.assertTrue(token_matches_in_offer_text("камерой", hay))

    def test_prepare_phone_good_camera(self):
        sq, _parsed, _ref, _rq, _fz, _fd = prepare_search_query(
            "телефон с хорошей камерой",
            company_names=["Samsung"],
            category_names=["Смартфоны", "Ноутбуки"],
        )
        low = sq.casefold()
        self.assertNotIn("хорош", low)
        self.assertIn("смартфон", low)
        self.assertIn("камер", low)

    def test_compact_drops_conversational_words_for_gpu_query(self):
        from search_index import compact_query_for_retrieval

        c = compact_query_for_retrieval(
            "короче хочу найти для друга игровой компьютер с видеокартой 4070"
        )
        low = c.casefold()
        self.assertNotIn("короче", low)
        self.assertNotIn("друга", low)
        self.assertIn("видеокарт", low)
        self.assertIn("4070", low)

    def test_tokenize_drops_conversational_words_for_gpu_query(self):
        from search_index import tokenize_search_query

        t = tokenize_search_query(
            "короче хочу найти для друга игровой компьютер с видеокартой 4070"
        )
        self.assertNotIn("короче", t)
        self.assertNotIn("друга", t)
        self.assertIn("видеокартой", t)
        self.assertIn("4070", t)


if __name__ == "__main__":
    unittest.main()
