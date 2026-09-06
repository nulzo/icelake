Relation endpoints named after a guild member now collapse to that member's user
node at write time (no more person-entity twins), and every person-facing graph
query (`relations_of`, `between`, `shared`, `similar_users`, `entity_stances`,
`neighbors`) follows `entity.linked_user_id` so pre-existing twins still resolve
to the member. Reads go through a single seam (`GraphApi._incident`) that unions
the member's user node with every entity twin and collapses the result, so edges
stored against a twin (a member mentioned by name before they spoke) surface on
that member's star. Added `graph.shared(a, b)`: the identity-collapsed,
weight-ranked intersection of two members' entity edges, with polarity
preserved. Bot IDs are no longer written to `FactRecord.related_user_ids`.
