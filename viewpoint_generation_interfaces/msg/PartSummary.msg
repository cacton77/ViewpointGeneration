# One catalogued part, as shown in the part picker.
string backend
string part_id
string title
string part_number
string revision
string maturity           # IN_WORK, RELEASED, ... (empty when the backend has none)
string container          # collaborative space, folder, ...
string sync_status        # new | synced | modified | archived
bool geometry_cached      # STEP is present locally
bool geometry_available   # the backend can produce STEP
string geometry_path      # local path, empty when not cached
string thumbnail_path     # local path, empty when not cached
string plan_status        # CURRENT | DOWNLOADED | UPDATED_FROM_REMOTE | SELECTED | STALE | NONE
string last_synced        # ISO-8601 UTC
