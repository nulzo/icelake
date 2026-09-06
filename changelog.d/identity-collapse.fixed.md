Relation endpoints named after a guild member now collapse to that member's user
node at write time (no more person-entity twins), and every person-facing graph
query (`relations_of`, `between`, `shared`, `similar_users`, `entity_stances`,
`neighbors`) follows `entity.linked_user_id` so pre-existing twins still resolve
to the member. Added `graph.shared(a, b)`: the identity-collapsed, weight-ranked
intersection of two members' entity edges, with polarity preserved. Bot IDs are
no longer written to `FactRecord.related_user_ids`.
