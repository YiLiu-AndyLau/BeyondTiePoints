# BeyondTiePoints 代码审查报告（2026-03-18）

> 审查范围：仓库内全部 Python 源码（`main.py`、`adjustment_core/*`、`model/*`、`pretrain/*`、`rpc.py`、`rs_image.py`、`utils.py`）。
> 审查方法：静态阅读 + 基础语法检查（`compileall` / `ast.parse`）。

## 一、总体结论

当前代码整体结构清晰、模块划分合理，核心流程（数据加载 → 格网构建 → 特征提取 → DDP 优化 → RPC 回写）可读性较强。

但在**可公开发布**层面，仍存在若干风险点：

- 有会直接导致运行失败的“配置/环境耦合”问题；
- 有会引发数值不稳定（NaN）的实现细节；
- 有会掩盖真实错误、增加排障成本的异常处理方式；
- 有部分参数语义与默认值不一致，会导致用户开箱即失败。

## 二、问题清单（按优先级）

### P0-1：`--encoder_path` 默认值和使用方式冲突，默认配置会直接失败

- 现象：
  - 参数 `--encoder_path` 默认值是一个文件路径（`.../backbone.pth`）；
  - 代码实际使用时又拼接 `adapter.pth`，把它当目录。
- 风险：
  - 使用默认参数时，`os.path.join(args.encoder_path, 'adapter.pth')` 变为 `.../backbone.pth/adapter.pth`，几乎必定 `FileNotFoundError`。
- 建议：
  - 将参数改名为 `--encoder_dir`；或
  - 将默认值改为目录；并在启动时做路径存在性检查，给出清晰报错。

### P0-2：DDP 初始化强依赖 `LOCAL_RANK` + `nccl`，对单卡/CPU/非 torchrun 场景不兼容

- 现象：
  - `setup_ddp()` 直接 `dist.init_process_group(backend='nccl')`，并读取 `os.environ['LOCAL_RANK']`。
- 风险：
  - 非分布式启动（如本地单卡调试、CI 语法检查）会因缺少 `LOCAL_RANK` 直接崩溃；
  - 无 GPU 或 NCCL 不可用场景也无法运行。
- 建议：
  - 支持 `WORLD_SIZE==1` 的降级路径；
  - `LOCAL_RANK` 改为 `os.getenv(..., 0)` 并根据 `torch.cuda.is_available()` 切换 backend（如 gloo）；
  - 在 README 中明确启动方式（`torchrun --nproc_per_node=...`）。

### P1-1：影像读取缺少空值校验，坏数据会在构造阶段触发隐式错误

- 现象：
  - `cv2.imread(..., IMREAD_GRAYSCALE)` 结果未判空，直接 `np.stack([self.image]*3, axis=-1)`。
- 风险：
  - 缺图、路径错误、损坏文件时，错误栈会偏离真实原因，影响排障。
- 建议：
  - 读取后立即检查 `if self.image is None: raise FileNotFoundError(...)`；
  - 报错中带上影像路径和样本 ID。

### P1-2：`feature_sampling` 在零距离邻域下存在 NaN 风险

- 现象：
  - 先 `dists / sum(dists)`，若 query 与近邻重合导致 `sum(dists)==0`，会先产生 NaN；后续再 `+1e-6` 已无法完全补救。
- 风险：
  - 传播到 loss 后触发不稳定、训练中断或“隐式退化”。
- 建议：
  - 对分母加 epsilon：`sum_d = torch.sum(dists, dim=1, keepdim=True).clamp_min(1e-6)`；
  - 或改为基于 `softmax(-dists)` 的稳定权重。

### P1-3：`stop_criterion='error'` 分支对 `<=0/NaN` 指标处理不完整，可能导致“永不更新 best model”

- 现象：
  - 仅在 `current_metric_val > 0` 时才更新 best；
  - 当误差统计无效（NaN）或特殊情况下为 0 时，best 状态不会更新。
- 风险：
  - 训练结束后 `best_model_state` 可能为空，回退逻辑不一定符合预期。
- 建议：
  - 显式处理 `NaN/Inf`，并定义 `0` 的行为；
  - 首轮可强制保存一次基线权重，避免空 best。

### P2-1：`load_imgs_bundle` 将每张图的 `tie_points_height` 覆盖为第 1 张图的数据

- 现象：
  - `images[-1].tie_points_height = images[0].tie_points_height`。
- 风险：
  - 语义上明显不合理，后续若使用该字段会埋雷。
- 建议：
  - 删除该覆盖；每张图独立维护其 tie-point height。

### P2-2：`create_checkerboard` 静默吞掉所有异常，调试成本高

- 现象：
  - `except Exception: pass`。
- 风险：
  - 失败时没有任何日志，无法定位 I/O、格式、路径问题。
- 建议：
  - 至少在非 auto 模式打印 warning（附路径和异常信息）。

### P2-3：候选图像索引 `select_imgs` 缺少边界检查

- 现象：
  - 直接 `img_folders = [img_folders[i] for i in select_img_idxs]`。
- 风险：
  - 非法索引会抛 `IndexError`，但缺少友好提示。
- 建议：
  - 在解析后验证索引范围并报可读错误。

## 三、已执行检查

1. 语法编译检查：`python -m compileall -q .`（通过）
2. AST 解析检查：遍历全部 `*.py` 使用 `ast.parse`（通过）

## 四、发布前最小修复建议（建议优先完成）

1. 修复 `--encoder_path` 默认值/语义冲突（P0）。
2. 为 DDP 增加单进程降级路径（P0）。
3. 补齐影像读入判空（P1）。
4. 修复 `feature_sampling` 的零分母问题（P1）。
5. 清理静默异常和不合理字段覆盖（P2）。

---

如需，我下一步可以直接按上述优先级提交一组“可公开发布前”的最小修复 PR（包含参数重命名兼容、运行前自检、数值稳定性补丁、日志改进）。
