"""Theme taxonomy and anchor phrases for Method A.

The taxonomy exists to answer one question precisely: of the songs that are about
romantic relationships, which ones argue that the narrator does *not* need the
relationship? That is a claim about the narrator's stance, not about the mood of the
song, and the two come apart constantly. "I Will Survive" is a breakup song with a
self-sufficiency stance; "Nothing Compares 2 U" is a breakup song with a longing
stance. The anchors below are written to separate stance from mood.

These same definitions are the codebook shown to the LLM in Method B and to the human
labellers building the gold set, so all three are answering an identical question.
"""

from __future__ import annotations

# Is the song about a romantic/sexual relationship at all?
RELATIONSHIP_ANCHORS = [
    "this song is about being in love with someone",
    "this song is about a romantic relationship between two people",
    "this song is about wanting to be with someone romantically",
    "this song is about breaking up with a partner",
    "this song is about desiring someone sexually",
    "a song addressed to a lover or a partner",
]

# Explicit non-relationship topics, so the relationship score is a contrast rather
# than an absolute threshold. Hit songs are overwhelmingly about relationships, so a
# bare similarity cutoff would classify nearly everything as one.
NON_RELATIONSHIP_ANCHORS = [
    "this song is about dancing and having a party",
    "this song is about money, success and fame",
    "this song is about a place, a city or a hometown",
    "this song is about God, faith and religion",
    "this song is about war, politics and social protest",
    "this song is about friendship between friends",
    "this song is about childhood memories and growing up",
    "this song is about cars, driving and the road",
    "this song is about drugs and getting high",
    "this song is about grief for someone who died",
    "this song is a novelty song about a fictional character or a dance craze",
    "this song is about working a hard job for low pay",
]

# Stance taken toward the relationship, conditional on it being a relationship song.
STANCE_ANCHORS: dict[str, list[str]] = {
    "devotion_commitment": [
        "I will love you forever and never leave you",
        "you are my everything and I want to marry you",
        "we belong together and I am committed to you for life",
        "I am so happy to be with you, you make my life complete",
    ],
    "longing_pursuit": [
        "I wish you were mine, I cannot stop thinking about you",
        "please notice me, give me a chance to be with you",
        "I want you so badly but you do not feel the same way",
        "I am waiting for you to come back to me",
    ],
    "heartbreak_loss": [
        "you left me and my heart is broken, I cannot stop crying",
        "I am lost and miserable without you since we broke up",
        "I still love you even though you hurt me and left",
        "I regret losing you and I want you back",
    ],
    "casual_physical": [
        "let us go home together tonight, no strings attached",
        "this is purely physical, I just want your body",
        "we do not need to put a label on what this is",
        "one night together and then we go our separate ways",
    ],
    "independence_self_sufficiency": [
        "I do not need you, I am better off on my own",
        "I am fine by myself and I do not need a partner to be happy",
        "I pay my own bills and I do not need anyone to take care of me",
        "I am stronger since you left and I have moved on without you",
        "I do not want a relationship, I am focused on myself",
        "you are replaceable, I will be perfectly happy without you",
    ],
    "conflict_resentment": [
        "you cheated on me and I am furious with you",
        "I am telling you to get out of my life for good",
        "we fight all the time and you treat me badly",
        "I am calling you out for lying to me",
    ],
}

# Discriminative axis for the headline question.
#
# Validation against songs with known stances showed that argmax over STANCE_ANCHORS
# reliably identifies devotion, longing and heartbreak, but collapses independence
# into heartbreak: "I Will Survive", "Since U Been Gone", "thank u, next", "Truth
# Hurts" and "Stronger (What Doesn't Kill You)" were all labelled heartbreak_loss.
# The cause is that embedding similarity tracks topic (this is a breakup song) rather
# than stance (and the narrator is fine about it), and both stances share almost all
# of their surface vocabulary.
#
# These two groups are therefore scored as an explicit contrast: a song high on
# RESOLVED and low on YEARNING is a "I don't need you" song, whatever its overall
# mood. The difference between them is a continuous feature, not a hard label, so the
# analysis can report a distribution instead of forcing a threshold.
POST_BREAKUP_RESOLVED = [
    "I am over you and I feel free now that you are gone",
    "losing you was the best thing that ever happened to me",
    "I do not need you, I am better off on my own",
    "I have moved on and I am stronger without you",
    "do not call me, I am not taking you back",
    "I am happier alone than I ever was with you",
]

POST_BREAKUP_YEARNING = [
    "please come back to me, I want you back",
    "I still cry over you every single night",
    "I cannot move on because I am still in love with you",
    "nothing feels right without you here beside me",
    "I would do anything to have you back again",
    "I am begging you not to leave me",
]

# Transparent keyword route, run alongside the embeddings so a reader can audit why a
# song scored the way it did. Phrases are matched on normalized lowercase text.
RELATIONSHIP_KEYWORDS = [
    "love", "baby", "heart", "kiss", "kissing", "lover", "romance", "darling",
    "sweetheart", "girlfriend", "boyfriend", "honey", "in love", "my girl", "my man",
    "hold me", "touch me", "your arms", "fall in love", "break up", "broke up",
    "together", "forever", "cheat", "cheating", "marry", "wedding",
]

INDEPENDENCE_PHRASES = [
    "don't need you", "dont need you", "do not need you", "don't need a man",
    "don't need no man", "don't need nobody", "don't need anybody",
    "better off alone", "better off without", "better without you",
    "on my own", "by myself", "myself", "independent", "independent woman",
    "i don't need", "i dont need", "without you i", "moved on", "moving on",
    "over you", "i'm fine", "im fine", "survive", "stronger", "walk away",
    "leave you", "i'm done", "im done", "not coming back", "goodbye",
    "single", "free", "my own money", "my own",
]

# Stance is scored against these; keep in sync with STANCE_ANCHORS keys.
STANCE_LABELS = list(STANCE_ANCHORS.keys())

# Which stances count as "does not need the relationship" when reporting the
# headline share. conflict_resentment is deliberately excluded: anger at a partner is
# still engagement with the relationship, not a claim of not needing it.
INDEPENDENT_STANCES = {"independence_self_sufficiency"}
