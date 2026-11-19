# ADR-003：小组 GitHub 协作与双远端同步工作流

- **状态**：已采纳
- **背景**：本项目为小组多人协作开发，同时课程评分明确将「课程结束前的持续 Commit / PR 更新次数」计入中间能力提升分。

## 仓库拓扑
- `origin`（小组 Fork 仓库）：`https://github.com/zhang040705/LLM-Course.git`
- `upstream`（老师课程官方仓库）：`https://github.com/CAI-testbooks/LLM-Course.git`

## 决策与工作流
1. **小组日常分支开发**：每位成员基于 `origin/main` 拉独立分支开发，自测通过后及时 Push 到 `origin` 并通过小组内 PR 合并至 `main`，防止长期离线积累导致大冲突。
2. **定期同步老师仓库 PR**：按里程碑（V0–V4）或每周定期同步 `upstream/main` 后，由人工确认将小组稳定 `main` 更新至老师仓库的作业 PR，完整留存各阶段迭代记录与指标提升轨迹。
3. **Agent 权限边界**：AI Agent 可协助本地开发、提交及经确认后推送至个人功能分支，但禁止越权直接推送或修改 `upstream` 仓库。
