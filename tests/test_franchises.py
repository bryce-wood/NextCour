from nextcour.ingest.franchises import typed_groups, untyped_groups


def as_sets(groups):
    return sorted(sorted(members) for members in groups.values())


def test_untyped_links_chain_everything_together():
    related = {1: [2], 2: [1, 3], 3: [2], 4: []}
    assert as_sets(untyped_groups(related)) == [[1, 2, 3], [4]]


# Catalog shows 1-3 are franchise A, 4-5 franchise B, 6 a crossover of both, 7 a side story of A,
# 8 a show without an AniList ID. AniList IDs are the catalog IDs times 100.
MEMBERS = [1, 2, 3, 4, 5, 6, 7, 8]
ANILIST = {show: [show * 100] for show in range(1, 8)}
SEQUELS = [(100, 200, "SEQUEL"), (200, 100, "PREQUEL"), (200, 300, "SEQUEL"), (400, 500, "SEQUEL")]
CROSSOVER = [(600, 100, "PARENT"), (600, 400, "PARENT"), (100, 600, "SIDE_STORY"), (400, 600, "SIDE_STORY")]
SIDE_STORY = [(700, 300, "PARENT"), (300, 700, "SIDE_STORY")]


def test_a_crossover_with_two_parent_franchises_stands_alone():
    groups = typed_groups(MEMBERS, ANILIST, SEQUELS + CROSSOVER + SIDE_STORY, related={})
    assert as_sets(groups) == [[1, 2, 3, 7], [4, 5], [6], [8]]


def test_character_and_other_relations_never_join_franchises():
    edges = [*SEQUELS, (100, 400, "CHARACTER"), (200, 500, "OTHER")]
    assert as_sets(typed_groups([1, 2, 3, 4, 5], ANILIST, edges, related={})) == [[1, 2, 3], [4, 5]]


def test_a_crossover_joins_once_its_parents_turn_out_to_be_one_franchise():
    # 6 is a side story of both 1 and 7; 7 is itself a side story of 1.
    edges = [(600, 100, "PARENT"), (600, 700, "PARENT"), (700, 100, "PARENT")]
    assert as_sets(typed_groups([1, 6, 7], ANILIST, edges, related={})) == [[1, 6, 7]]


def test_shows_without_anilist_ids_join_only_one_franchise():
    related = {8: [1, 2], 9: [3, 4]}
    groups = typed_groups([1, 2, 3, 4, 5, 8, 9], ANILIST, SEQUELS, related)
    assert as_sets(groups) == [[1, 2, 3, 8], [4, 5], [9]]
