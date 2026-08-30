"""Graph explorer snapshot: public-API projection, no incidence edges."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from icelake.models.admin import GuildStats, MemoryExport
from icelake.models.facts import FactRecord
from icelake.models.graph import EntityKind, EntityRecord, NodeType, Polarity, RelationEdge
from icelake.models.operations import ProposedEntity, ProposedRelation
from icelake.visualizer.__main__ import parse_args, run
from icelake.visualizer.html import render_html, write_html
from icelake.visualizer.models import VizAlias
from icelake.visualizer.snapshot import (
    SERVER_NODE_ID,
    CenterAmbiguousError,
    CenterError,
    build_snapshot,
    collect_user_ids,
    compact_fact,
    ego_keep,
    entity_node_id,
    match_entity,
    snapshot_from_export,
    user_node_id,
)

GUILD = "500000000000000001"
ALICE = "100000000000000001"
BOB = "200000000000000002"


def _fact(*, fid: str, subject: str | None, text: str) -> FactRecord:
    now = datetime(2026, 8, 29, tzinfo=UTC)
    return FactRecord(
        id=fid,
        guild_id=GUILD,
        subject_id=subject,
        text=text,
        created_at=now,
        observed_at=now,
    )


def _export(
    *,
    facts: tuple[FactRecord, ...] = (),
    entities: tuple[EntityRecord, ...] = (),
    relations: tuple[RelationEdge, ...] = (),
) -> MemoryExport:
    return MemoryExport(guild_id=GUILD, facts=facts, entities=entities, relations=relations)


def _stats(**counts: int) -> GuildStats:
    return GuildStats(guild_id=GUILD, **counts)


class TestPureSnapshot:
    def test_users_server_entities_and_identity_edge(self) -> None:
        klim = EntityRecord(
            guild_id=GUILD,
            slug="klim",
            name="klim",
            kind=EntityKind.PERSON,
            linked_user_id=ALICE,
        )
        orphan = EntityRecord(guild_id=GUILD, slug="linux", name="linux", kind=EntityKind.CONCEPT)
        rel = RelationEdge(
            guild_id=GUILD,
            src_type=NodeType.USER,
            src_id=ALICE,
            dst_type=NodeType.ENTITY,
            dst_id="linux",
            verb="likes",
            polarity=Polarity.POSITIVE,
            weight=1.2,
            evidence_fact_ids=("fct1",),
        )
        export = _export(
            facts=(
                _fact(fid="fct1", subject=ALICE, text="alice likes linux"),
                _fact(fid="fct2", subject=None, text="the server loves movie night"),
            ),
            entities=(klim, orphan),
            relations=(rel,),
        )
        snap = snapshot_from_export(
            export,
            _stats(total_facts=2, active_facts=2, user_count=1, entity_count=2, relation_count=1),
            {ALICE: ("alice", (VizAlias(alias="alice", source="display_name", weight=0.7),))},
        )
        ids = {n.id: n for n in snap.nodes}
        assert SERVER_NODE_ID in ids
        assert ids[user_node_id(ALICE)].label == "alice"
        assert ids[entity_node_id("klim")].linked_user_id == ALICE
        kinds = {e.kind.value for e in snap.edges}
        verbs = {e.verb for e in snap.edges}
        assert kinds == {"relation", "identity"}
        assert "is" in verbs and "likes" in verbs
        assert any(f.subject_id is None for f in snap.facts)
        assert compact_fact(export.facts[0]).id == "fct1"

    def test_collects_related_and_linked_users(self) -> None:
        export = _export(
            facts=(
                _fact(fid="f", subject=ALICE, text="mentions bob").model_copy(
                    update={"related_user_ids": (BOB,)}
                ),
            ),
            entities=(
                EntityRecord(
                    guild_id=GUILD,
                    slug="klim",
                    name="klim",
                    kind=EntityKind.PERSON,
                    linked_user_id=BOB,
                ),
            ),
            relations=(
                RelationEdge(
                    guild_id=GUILD,
                    src_type=NodeType.ENTITY,
                    src_id="klim",
                    dst_type=NodeType.USER,
                    dst_id=ALICE,
                    verb="is",
                    polarity=Polarity.NEUTRAL,
                ),
            ),
        )
        assert collect_user_ids(export) == {ALICE, BOB}

    def test_ego_filters_to_neighborhood(self) -> None:
        alice = user_node_id(ALICE)
        bob = user_node_id(BOB)
        linux = entity_node_id("linux")
        rel_alice = RelationEdge(
            guild_id=GUILD,
            src_type=NodeType.USER,
            src_id=ALICE,
            dst_type=NodeType.ENTITY,
            dst_id="linux",
            verb="likes",
            polarity=Polarity.POSITIVE,
        )
        rel_bob = RelationEdge(
            guild_id=GUILD,
            src_type=NodeType.USER,
            src_id=BOB,
            dst_type=NodeType.ENTITY,
            dst_id="linux",
            verb="hates",
            polarity=Polarity.NEGATIVE,
        )
        export = _export(
            facts=(
                _fact(fid="a", subject=ALICE, text="alice likes linux"),
                _fact(fid="b", subject=BOB, text="bob hates linux"),
            ),
            entities=(
                EntityRecord(guild_id=GUILD, slug="linux", name="linux", kind=EntityKind.CONCEPT),
            ),
            relations=(rel_alice, rel_bob),
        )
        snap = snapshot_from_export(
            export,
            _stats(),
            {ALICE: ("alice", ()), BOB: ("bob", ())},
            center_id=alice,
            depth=1,
        )
        ids = {n.id for n in snap.nodes}
        assert ids == {alice, linux}
        assert bob not in ids

    def test_ego_missing_center_raises(self) -> None:
        with pytest.raises(CenterError):
            snapshot_from_export(_export(), _stats(), {}, center_id="user:missing")

    def test_match_entity_by_alias(self) -> None:
        entities = (
            EntityRecord(
                guild_id=GUILD,
                slug="koji-sushi",
                name="Koji Sushi",
                kind=EntityKind.PLACE,
                aliases=("koji",),
            ),
        )
        assert match_entity(entities, "Koji") == entity_node_id("koji-sushi")
        assert match_entity(entities, "nope") is None

    def test_ego_keep_depth_zero_is_center_only(self) -> None:
        from icelake.visualizer.models import VizEdge, VizEdgeKind

        edges = (
            VizEdge(
                id="1",
                source="a",
                target="b",
                verb="likes",
                polarity="positive",
                kind=VizEdgeKind.RELATION,
            ),
        )
        assert ego_keep(edges, "a", 0) == {"a"}
        assert ego_keep(edges, "a", 1) == {"a", "b"}
        assert ego_keep((), "solo", 3) == {"solo"}


class TestLiveSnapshot:
    async def test_build_snapshot_from_remember(self, make_client) -> None:
        client, _ = make_client(llm=False)
        await client.start()
        await client.facts.remember(
            guild_id=GUILD,
            subject_id=ALICE,
            text="alice likes cycling on weekends around the lake",
            subject_username="alice",
            entities=(ProposedEntity(name="cycling", kind=EntityKind.CONCEPT),),
            relations=(ProposedRelation(verb="likes", from_token=ALICE, to_entity="cycling"),),
        )
        await client.facts.remember(
            guild_id=GUILD,
            subject_id=None,
            text="the community bonds over late night movie nights together",
        )
        snap = await build_snapshot(client, GUILD)
        assert any(n.type.value == "user" and n.label == "alice" for n in snap.nodes)
        assert any(n.type.value == "server" for n in snap.nodes)
        assert any(e.verb == "likes" for e in snap.edges)
        assert any(f.subject_id is None for f in snap.facts)
        html = render_html(snap)
        assert "alice" in html and "cycling" in html and GUILD in html
        await client.close()

    async def test_center_user_and_entity(self, make_client) -> None:
        client, _ = make_client(llm=False)
        await client.start()
        await client.facts.remember(
            guild_id=GUILD,
            subject_id=ALICE,
            text="alice likes cycling on weekends around the lake",
            subject_username="alice",
            entities=(ProposedEntity(name="cycling", kind=EntityKind.CONCEPT),),
            relations=(ProposedRelation(verb="likes", from_token=ALICE, to_entity="cycling"),),
        )
        as_user = await build_snapshot(client, GUILD, center="alice", depth=1)
        assert as_user.center == user_node_id(ALICE)
        as_ent = await build_snapshot(client, GUILD, center="cycling", depth=1)
        assert as_ent.center == entity_node_id("cycling")
        await client.close()

    async def test_center_server_keeps_server_node(self, make_client) -> None:
        client, _ = make_client(llm=False)
        await client.start()
        await client.facts.remember(
            guild_id=GUILD, subject_id=None, text="the server hosts a weekly game night"
        )
        snap = await build_snapshot(client, GUILD, center="the server", depth=0)
        assert snap.center == SERVER_NODE_ID
        assert {n.id for n in snap.nodes} == {SERVER_NODE_ID}
        await client.close()

    async def test_unresolved_center(self, make_client) -> None:
        client, _ = make_client(llm=False)
        await client.start()
        with pytest.raises(CenterError, match="could not resolve"):
            await build_snapshot(client, GUILD, center="nobody-here")
        await client.close()

    async def test_ambiguous_center(self, make_client) -> None:
        from icelake.models.identity import AliasSource

        client, _ = make_client(llm=False)
        await client.start()
        await client.identity.register_alias(GUILD, ALICE, "sam", AliasSource.DISPLAY_NAME)
        await client.identity.register_alias(GUILD, BOB, "sam", AliasSource.DISPLAY_NAME)
        with pytest.raises(CenterAmbiguousError, match="ambiguous"):
            await build_snapshot(client, GUILD, center="sam")
        await client.close()


class TestHtmlAndCli:
    def test_write_html(self, tmp_path: Path) -> None:
        snap = snapshot_from_export(_export(), _stats(), {})
        path = tmp_path / "graph.html"
        write_html(path, snap)
        text = path.read_text(encoding="utf-8")
        assert "icelake" in text and GUILD in text

    def test_parse_requires_storage(self) -> None:
        args = parse_args(["--storage", "sqlite://:memory:", "--guild", GUILD, "--out", "x.html"])
        assert args.guild == GUILD
        assert args.depth == 2

    async def test_run_list_guilds(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        db = tmp_path / "mem.db"
        url = f"sqlite:///{db}"
        from icelake.api.client import DiscordMemory
        from tests.conftest import make_config

        client = DiscordMemory(make_config(storage=url), llm=None)
        await client.start()
        await client.facts.remember(
            guild_id=GUILD, subject_id=ALICE, text="alice keeps a sourdough starter going"
        )
        await client.close()

        args = parse_args(["--storage", url, "--list-guilds"])
        code = await run(args)
        assert code == 0
        assert GUILD in capsys.readouterr().out

    async def test_run_writes_file(self, tmp_path: Path) -> None:
        db = tmp_path / "mem.db"
        url = f"sqlite:///{db}"
        from icelake.api.client import DiscordMemory
        from tests.conftest import make_config

        client = DiscordMemory(make_config(storage=url), llm=None)
        await client.start()
        await client.facts.remember(
            guild_id=GUILD, subject_id=ALICE, text="alice likes cycling around town often"
        )
        await client.close()

        out = tmp_path / "out.html"
        args = parse_args(["--storage", url, "--guild", GUILD, "--out", str(out)])
        assert await run(args) == 0
        assert out.is_file() and "alice" in out.read_text(encoding="utf-8")

    async def test_run_missing_guild(self, tmp_path: Path) -> None:
        args = parse_args(["--storage", f"sqlite:///{tmp_path / 'x.db'}"])
        assert await run(args) == 1

    async def test_run_unresolved_center(self, tmp_path: Path) -> None:
        db = tmp_path / "mem.db"
        url = f"sqlite:///{db}"
        from icelake.api.client import DiscordMemory
        from tests.conftest import make_config

        client = DiscordMemory(make_config(storage=url), llm=None)
        await client.start()
        await client.facts.remember(
            guild_id=GUILD, subject_id=ALICE, text="alice likes cycling around town often"
        )
        await client.close()
        args = parse_args(
            [
                "--storage",
                url,
                "--guild",
                GUILD,
                "--center",
                "zzz-nobody",
                "--out",
                str(tmp_path / "o.html"),
            ]
        )
        assert await run(args) == 1
