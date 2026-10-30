# plm_bridge extraction: status

**Finished** (2026-10-30), including the follow-ups from Colin's answers. The
PLM catalog moved out of this package into the plm_bridge repository. Read
`REPORT.md` there for results, and `QUESTIONS.md` there for his answers and
the two questions still open. Phase 3 (Onshape and local backends) is
deferred at his request.

## Where everything is

| What | Where |
|---|---|
| New repository | <https://github.com/cacton77/plm_bridge> (public since 2026-10-30, BSD-3-Clause), branch `main` |
| ViewpointGeneration | this branch, `plm-bridge` (from `feature-3dx-integration` @ `53c0ba1`) |
| inspection-docker | branch `plm-bridge` (from `feature-3dx-integration` @ `68bfdbd`) |
| New repository on the lab machine | `inspection-docker/ros2_ws/src/plm_bridge` (`ros2_ws/src` exists) |
| ViewpointGeneration worktree | `/home/macs/plm_bridge_work/ws/src/ViewpointGeneration` |
| inspection-docker worktree | `/home/macs/plm_bridge_work/inspection-docker` |
| Test workspace | `/home/macs/plm_bridge_work/ws` (built in throwaway `inspection-docker:jazzy` containers; helper scripts in `/home/macs/plm_bridge_work/tools`) |

The worktrees are outside `ros2_ws/src`, so the lab workspace never sees two
copies of `viewpoint_generation`. Colin's checkouts were not touched:
ViewpointGeneration on `main` and inspection-docker on
`feature-3dx-integration`, both with uncommitted work. The cell's real
`catalog/` was not touched either; tests used copies of it.

## Latest pushed commits

| Repository | Branch | Commit |
|---|---|---|
| plm_bridge | `main` | `20b8e8d` (license, link tests, final report) |
| ViewpointGeneration | `plm-bridge` | this commit; code at `afc94f9` (re-selection fix) |
| inspection-docker | `plm-bridge` | `b46ae16` (Colin's reorganisation), `d59ba93` (plm_bridge on the new layout), then a docs commit |

Phase commits on this branch: Phase 0 `b4a950d`, Phase 1 `88445ad`, Phase 2
`52ae8b6`, Phase 4 `b23e871`.

## Clone on a fresh machine

```bash
git clone https://github.com/cacton77/inspection-docker.git -b plm-bridge
cd inspection-docker/ros2_ws/src
git clone https://github.com/cacton77/ViewpointGeneration.git -b plm-bridge
git clone https://github.com/cacton77/plm_bridge.git
cd .. && colcon build      # inside the inspection-docker container
```

ViewpointGeneration on this branch needs plm_bridge in the same workspace,
because it depends on `plm_interfaces` and `cad_loader`. inspection-docker's
`ros2_ws/src/shared.repos` on its `plm-bridge` branch lists both.
