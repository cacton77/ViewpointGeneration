# plm_bridge extraction: status

Status file for the work that moves the PLM catalog out of ViewpointGeneration
into its own repository. Updated on every push.

## Where everything is

| What | Where |
|---|---|
| New repository | <https://github.com/cacton77/plm_bridge> (private), branch `main` (Option A: created with `gh`) |
| ViewpointGeneration | branch `plm-bridge` (from `feature-3dx-integration` @ `53c0ba1`) |
| inspection-docker | branch `plm-bridge` (from `feature-3dx-integration` @ `68bfdbd`) |
| Workspace path used for the new repo on the lab machine | `inspection-docker/ros2_ws/src/plm_bridge` (`ros2_ws/src` exists) |
| ViewpointGeneration worktree | `/home/macs/plm_bridge_work/ws/src/ViewpointGeneration` |
| inspection-docker worktree | `/home/macs/plm_bridge_work/inspection-docker` |
| Test workspace (colcon) | `/home/macs/plm_bridge_work/ws` (the ViewpointGeneration worktree plus the new repo, mounted into a throwaway `inspection-docker:jazzy` container) |

The worktrees sit outside `ros2_ws/src` so the lab workspace never sees two
copies of `viewpoint_generation`. Colin's checkouts (ViewpointGeneration on
`main`, inspection-docker on `feature-3dx-integration`) had uncommitted work
and were not touched.

## Latest pushed commits

| Repository | Branch | Commit |
|---|---|---|
| plm_bridge | `main` | `fbdd07b` and later (see `git log -1` there; REPORT.md, QUESTIONS.md) |
| ViewpointGeneration | `plm-bridge` | this commit (see `git log -1`); Phase 1 = `88445ad`, Phase 2 = `52ae8b6` |
| inspection-docker | `plm-bridge` | `68bfdbd` (no changes yet) |

## Current phase

**Phase 4 (extract to the plm_bridge repository): working.** The code has
moved: this branch no longer contains the catalog, the picker, or the STEP
loader, and depends on `plm_interfaces` and `cad_loader` from plm_bridge.
Phases 0, 1 and 2 are done, and the live 3DX smoke checklist passes on the
extracted packages. Phase 3 (Onshape/local backends) is deferred at Colin's
request. Results are in `REPORT.md` in the
new repository; open questions are in `QUESTIONS.md` there.

Scope change from Colin (2026-10-30): 3DX only. No Onshape backend and no
Onshape references in the new package for now; Phase 3 is deferred.

## Clone on a fresh machine

```bash
git clone https://github.com/cacton77/inspection-docker.git -b plm-bridge
mkdir -p inspection-docker/ros2_ws/src && cd inspection-docker/ros2_ws/src
git clone https://github.com/cacton77/ViewpointGeneration.git -b plm-bridge
git clone https://github.com/cacton77/plm_bridge.git        # private: needs access
cd .. && colcon build      # inside the inspection-docker container
```

ViewpointGeneration on this branch needs plm_bridge in the same workspace (it
depends on `plm_interfaces` and `cad_loader`).
