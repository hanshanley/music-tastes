"""Tests for song identity resolution.

These cover the normalization decisions that determine what counts as "the same
song", which is the foundation every later stage sits on. Getting these wrong
silently changes the denominator of every share reported in the findings.
"""

from __future__ import annotations

import pytest

from music_tastes.resolve_songs import (
    normalize_artist,
    normalize_title,
    split_artist,
    strip_accents,
    title_variants,
)


class TestNormalizeTitle:
    def test_case_and_whitespace(self):
        assert normalize_title("  Hey   JUDE ") == "hey jude"

    def test_diacritics_folded(self):
        # The two chart archives disagree on this exact title, which is what
        # motivated accent folding in the first place.
        assert normalize_title("Nel Blu Dipinto Di Blu (Volaré)") == normalize_title(
            "Nel Blu Dipinto Di Blu (Volare)"
        )

    @pytest.mark.parametrize(
        "raw",
        [
            "Respect (Remastered)",
            "Respect (2015 Remaster)",
            "Respect (Single Version)",
            "Respect [Album Version]",
            "Respect (Radio Edit)",
            "Respect (Mono)",
        ],
    )
    def test_edition_suffixes_stripped(self, raw):
        assert normalize_title(raw) == "respect"

    def test_stacked_suffixes_stripped(self):
        assert normalize_title("Respect (Album Version) (Remastered)") == "respect"

    def test_meaningful_parenthetical_kept(self):
        # Not an edition marker, so it is part of the title and must survive.
        got = normalize_title("Sunflower (Spider-Man: Into The Spider-Verse)")
        assert "spider" in got

    def test_rerecording_not_collapsed(self):
        # Re-recordings chart as separate releases and must stay distinct.
        assert normalize_title("All Too Well (Taylor's Version)") != normalize_title(
            "All Too Well"
        )


class TestNormalizeArtist:
    def test_leading_the_removed(self):
        assert normalize_artist("The Beatles") == normalize_artist("Beatles")

    def test_featured_artist_ignored(self):
        # Featured credits vary between weeks and archives, so they are excluded
        # from the identity key.
        assert normalize_artist("Mark Ronson Featuring Bruno Mars") == "mark ronson"

    @pytest.mark.parametrize(
        "billing",
        ["Drake Feat. Rihanna", "Drake ft. Rihanna", "Drake Featuring Rihanna"],
    )
    def test_feature_separators(self, billing):
        assert normalize_artist(billing) == "drake"

    def test_punctuation_stripped(self):
        assert normalize_artist("Bobby \"Boris\" Pickett") == "bobby boris pickett"


class TestSplitArtist:
    def test_primary_and_featured(self):
        primary, featured = split_artist("Drake Featuring Kanye West")
        assert primary == "Drake"
        assert featured == ["Kanye West"]

    def test_no_feature(self):
        primary, featured = split_artist("Queen")
        assert primary == "Queen"
        assert featured == []


class TestTitleVariants:
    def test_double_a_side_split(self):
        got = title_variants("Foolish Games/You Were Meant For Me")
        assert "foolish games" in got
        assert "you were meant for me" in got

    def test_parenthetical_stripped_variant_present(self):
        got = title_variants("Sunflower (Spider-Man: Into The Spider-Verse)")
        assert "sunflower" in got

    def test_full_form_first(self):
        got = title_variants("Hello")
        assert got[0] == "hello"

    def test_no_empty_variants(self):
        assert all(v for v in title_variants("A (Remaster)"))


def test_strip_accents():
    assert strip_accents("Beyoncé") == "Beyonce"
    assert strip_accents("volaré") == "volare"
