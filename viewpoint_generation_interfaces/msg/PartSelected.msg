# Published (latched) when an operator selects a part for inspection.
# The plan producer subscribes, loads the model, and plans against it.
string backend            # registry name of the part's PLM backend, e.g. dx3
string part_id            # backend-native id; opaque
string title
string part_number
string revision
string change_token       # the part's change token at selection time
string geometry_path      # local STEP (or mesh) path, as seen inside the container
string mesh_units         # only meaningful for non-STEP files ('m', 'mm', ...)
string plan_status        # CURRENT | DOWNLOADED | UPDATED_FROM_REMOTE | SELECTED | STALE | NONE
                          # SELECTED: an operator picked this exact plan; load it, with a
                          # warning, even when it targets an earlier change token.
string plan_file_path     # local plan file, empty when there is no plan
string plan_dir           # where the producer writes new plans for this part
string plm_context_json   # PLM context (JSON object) the producer embeds as each plan's
                          # plm_context, adding only stage/generated_at/software_version
