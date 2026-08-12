# Published when an operator selects a part for inspection.
# The viewpoint generation node subscribes and loads the model.
string eng_item_id
string title
string part_number
string revision
string cestamp            # CAD revision stamp the selection was made against
string step_file_path     # absolute path inside the container
string mesh_units         # units to load the model in ('m', 'mm', ...)
string plan_status        # CURRENT | DOWNLOADED | UPDATED_FROM_REMOTE | STALE | NONE
string plan_file_path     # local plan JSON, empty when there is no plan
