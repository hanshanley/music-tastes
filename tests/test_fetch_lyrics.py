"""Tests for Genius matching, including the API-free slug fallback.

The slug route guesses a URL from artist and title. That is only safe because the
guess is verified against the page's own <title> before being accepted, so these
tests concentrate on the accept/reject boundary. A false accept silently attaches
the wrong lyrics to a song, which would corrupt every downstream measure without
raising an error anywhere.
"""

from __future__ import annotations

import pytest

from music_tastes.fetch_lyrics import (
    TIER_A,
    _slugify,
    looks_instrumental,
    is_probably_english,
    pick_match,
    slug_candidates,
    verify_page,
)


def _page(title: str) -> str:
    return f"<html><head><title>{title}</title></head><body></body></html>"


class TestSlugConstruction:
    def test_basic_slug(self):
        urls = slug_candidates("Hello", "Adele")
        assert "https://genius.com/Adele-hello-lyrics" in urls

    def test_leading_the_capitalised(self):
        urls = slug_candidates("Blinding Lights", "The Weeknd")
        assert any("The-weeknd-blinding-lights-lyrics" in u for u in urls)

    def test_ampersand_becomes_and(self):
        assert _slugify("Sam & Dave", True) == "Sam-and-dave"

    def test_apostrophes_dropped(self):
        assert _slugify("Don't Stop", False) == "dont-stop"

    def test_soundtrack_tag_produces_stripped_variant(self):
        urls = slug_candidates("Sunflower (Spider-Man: Into The Spider-Verse)", "Post Malone")
        assert any(u.endswith("-sunflower-lyrics") for u in urls)

    def test_leading_article_alternative_keeps_original_first(self):
        urls = slug_candidates("Example", "The Example Band")
        assert urls[0] == "https://genius.com/The-example-band-example-lyrics"
        assert "https://genius.com/Example-band-example-lyrics" in urls

    def test_dotted_initials_alternative(self):
        urls = slug_candidates("Example", "P.M. Example")
        assert "https://genius.com/Pm-example-example-lyrics" in urls
        assert len(urls) == len(set(urls))

    def test_missing_article_alternative_keeps_original_first(self):
        urls = slug_candidates("Example", "Example Band")
        assert urls[0] == "https://genius.com/Example-band-example-lyrics"
        assert "https://genius.com/The-example-band-example-lyrics" in urls

    def test_optional_quoted_nickname_alternative(self):
        urls = slug_candidates("Example", 'First "Nickname" Last')
        assert "https://genius.com/First-nickname-last-example-lyrics" in urls
        assert "https://genius.com/First-last-example-lyrics" in urls

    def test_missing_artist_does_not_invent_an_article_only_artist(self):
        assert slug_candidates("Example", " ") == []

    def test_alternate_slug_still_requires_matching_page_identity(self, monkeypatch):
        from music_tastes import fetch_lyrics
        from music_tastes.http import Response

        monkeypatch.setattr(fetch_lyrics, "get", lambda *a, **k: Response(
            "", 200, _page("Different Act – Example Lyrics | Genius Lyrics"), "", False
        ))
        assert fetch_lyrics.match_via_slug("Example", "The Example Band") is None


class TestVerifyPage:
    def test_accepts_exact_match(self):
        got = verify_page(_page("Adele – Hello Lyrics | Genius Lyrics"), "Hello", "Adele")
        assert got is not None
        ts, as_, _, _ = got
        assert ts >= TIER_A["title"] and as_ >= TIER_A["artist"]

    def test_rejects_wrong_song(self):
        got = verify_page(
            _page("Queen – Bohemian Rhapsody Lyrics | Genius Lyrics"), "Hello", "Adele"
        )
        assert got is not None
        ts, as_, _, _ = got
        # Must fall well below the acceptance bar.
        assert ts < TIER_A["title"] or as_ < TIER_A["artist"]

    def test_handles_artist_rename(self):
        # Genius shows the current name; the chart preserves the historical billing.
        got = verify_page(
            _page("Lady A – Need You Now Lyrics | Genius Lyrics"),
            "Need You Now",
            "Lady Antebellum",
        )
        assert got is not None
        ts, as_, _, _ = got
        assert ts == pytest.approx(1.0)
        assert as_ > 0.5  # clears tier B, not tier A

    def test_returns_none_without_title_tag(self):
        assert verify_page("<html><body>no title</body></html>", "Hello", "Adele") is None

    def test_returns_none_on_unsplittable_title(self):
        assert verify_page(_page("JustSomeText"), "Hello", "Adele") is None


class TestPickMatch:
    def _hit(self, title, artist, rank=0, state="complete"):
        return {
            "title": title,
            "artist_names": artist,
            "primary_artist": {"name": artist},
            "lyrics_state": state,
            "_rank": rank,
            "id": abs(hash(title + artist)) % 10**6,
        }

    def test_tier_a_exact(self):
        hits = [self._hit("Respect", "Aretha Franklin")]
        match, ts, as_, tier = pick_match(hits, "Respect", "Aretha Franklin")
        assert tier == "A" and match is not None

    def test_rejects_translation_pages(self):
        hits = [self._hit("Adele - Hello (Traducción al Español)", "Genius Traducciones")]
        match, _, _, tier = pick_match(hits, "Hello", "Adele")
        assert match is None and tier is None

    def test_rejects_editorial_aggregator(self):
        hits = [self._hit("Hits of the 2000s [TOP 64]", "Pop Genius")]
        match, _, _, tier = pick_match(hits, "I Gotta Feeling", "The Black Eyed Peas")
        assert match is None

    def test_real_artist_containing_genius_not_rejected(self):
        # "GZA/Genius" is a real act; the junk filter is anchored to avoid it.
        hits = [self._hit("Liquid Swords", "GZA/Genius")]
        match, _, _, _ = pick_match(hits, "Liquid Swords", "GZA/Genius")
        assert match is not None

    def test_incomplete_lyrics_skipped(self):
        hits = [self._hit("Respect", "Aretha Franklin", state="unreleased")]
        match, _, _, _ = pick_match(hits, "Respect", "Aretha Franklin")
        assert match is None

    def test_no_match_reports_best_observed_similarity(self):
        # Diagnosability: a failure should not report a flat zero.
        hits = [self._hit("Totally Different Song", "Someone Else")]
        match, ts, as_, tier = pick_match(hits, "Respect", "Aretha Franklin")
        assert match is None
        assert ts >= 0.0 and as_ >= 0.0


class TestLyricHeuristics:
    def test_instrumental_detected(self):
        assert looks_instrumental("[Instrumental]")

    def test_instrumental_section_does_not_exclude_a_vocal_song(self):
        assert not looks_instrumental("[Verse]\nYou and me together\n[Instrumental]\n[Outro]")
        assert looks_instrumental("[Intro]\n[Instrumental]\n...")

    def test_short_text_is_not_evidence_of_an_instrumental(self):
        assert not looks_instrumental("[Intro]\nla la")

    def test_normal_lyric_not_instrumental(self):
        text = "I heard there was a secret chord that David played and it pleased the Lord " * 3
        assert not looks_instrumental(text)

    def test_english_detection(self):
        # The heuristic deliberately requires at least 20 words: language cannot be
        # judged from a fragment, so short texts are never called English.
        text = (
            "I don't know why you say goodbye and I say hello to you my friend "
            "in the world today because we all live together in a yellow submarine"
        )
        assert is_probably_english(text)

    def test_below_word_minimum_is_not_english(self):
        assert not is_probably_english("I don't know why you say goodbye")

    def test_short_text_not_confidently_english(self):
        assert not is_probably_english("hola")
