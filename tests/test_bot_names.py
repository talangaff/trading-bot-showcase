"""Auto-generated bot names: a big word pool, and never a repeat."""

from webapp import bots, genome


def test_word_pool_is_large():
    assert len(bots._ADJ) * len(bots._NOUN) >= 4000
    assert len(set(bots._ADJ)) == len(bots._ADJ) and len(set(bots._NOUN)) == len(bots._NOUN)


def test_generated_names_never_repeat(monkeypatch):
    # shrink the pool to 4 combinations so collisions are certain
    monkeypatch.setattr(bots, "_ADJ", ("a", "b"))
    monkeypatch.setattr(bots, "_NOUN", ("x", "y"))
    names = []
    for _ in range(30):
        n = bots._fleet_name(1)
        bots.create_bot(n, genome=genome.random_genome(), user_id=1)
        names.append(n)
    assert len(set(names)) == 30


def test_dedupe_renames_generated_duplicates_but_not_custom_names():
    g = genome.random_genome()
    first = bots.create_bot("swift-otter g1", genome=g, user_id=1)
    dup = bots.create_bot("swift-otter g1", genome=g, user_id=1, origin="mutation")
    mine = bots.create_bot("My bot", genome=g, pool="custom", user_id=1, origin="manual")
    mine2 = bots.create_bot("My bot", genome=g, pool="custom", user_id=1, origin="manual")
    assert bots.dedupe_generated_names() == 1
    assert bots.get_bot(first)["name"] == "swift-otter g1"
    assert bots.get_bot(dup)["name"] != "swift-otter g1"
    assert bots.get_bot(mine)["name"] == bots.get_bot(mine2)["name"] == "My bot"
