# One catalogued engineering item, as shown in the part picker.
string eng_item_id        # 3DX object id (hex)
string title
string part_number
string revision
string maturity           # IN_WORK, RELEASED, OBSOLETE, ...
string collab_space
string sync_status        # new | synced | modified | archived
bool step_cached          # STEP file is present locally
bool step_available       # 3DX reports a STEP derived output
string step_path          # local path, empty when not cached
string thumbnail_path     # local path, empty when not cached
string plan_status        # CURRENT | DOWNLOADED | UPDATED_FROM_REMOTE | STALE | NONE
string last_synced        # ISO-8601 UTC
