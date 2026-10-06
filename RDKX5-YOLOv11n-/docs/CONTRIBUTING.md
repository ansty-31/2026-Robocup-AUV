# 贡献指南

首先，感谢你考虑为RDKX5-YOLOv11n-项目做出贡献！

本文档提供了参与项目开发的指导方针。

## 📋 目录

- [行为准则](#行为准则)
- [如何贡献](#如何贡献)
- [报告Bug](#报告bug)
- [提出新功能](#提出新功能)
- [提交代码](#提交代码)
- [代码规范](#代码规范)
- [提交信息规范](#提交信息规范)
- [代码审查流程](#代码审查流程)
- [项目结构与环境](#项目结构与环境)
- [开发环境设置](#开发环境设置)
- [测试](#测试)
- [文档](#文档)
- [问题？](#问题)
- [感谢](#感谢)

> 索引补齐于 2026-09-26：原目录只列到「提交信息规范」，后面几节（代码审查流程 / 开发环境设置 /
> 测试 / 文档 / 问题 / 感谢）与新增的「项目结构与环境」都没进目录，已一并补上。

## 行为准则

参与本项目的所有人都应遵守以下准则：

- 使用友好和包容的语言
- 尊重不同的观点和经验
- 优雅地接受建设性批评
- 关注对社区最有利的事情
- 对其他社区成员表示同情

## 如何贡献

### 报告Bug

如果你发现了bug，请创建一个issue并提供以下信息：

**Bug报告模板：**

```markdown
## Bug描述
简要描述问题

## 复现步骤
1. 执行命令 '...'
2. 看到错误 '...'

## 预期行为
应该发生什么

## 实际行为
实际发生了什么

## 环境信息
- OS: [e.g. Ubuntu 22.04]
- Python版本: [e.g. 3.10]
- RDK X5固件版本: [e.g. 2.0]
- 工具链版本: [e.g. 1.2.8]

## 日志
粘贴相关日志

## 截图
如果适用，添加截图
```

> ⚠️ 已过时（2026-09-26）：上面模板的「环境信息」请按本仓库实际值填，别照抄示例 ——
> Python 是 conda `yolov8` 的 **3.9.25**（示例写的 3.10 只是占位），工具链仍是 OpenExplorer **v1.2.8**
> （镜像 `openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8-py310`，见 `scripts/3_export/quantize.sh`），
> 量化环境与 PC 训练环境是**两套**，报告 bug 时都要写清。

### 提出新功能

如果你有新功能建议，请创建一个issue：

**功能建议模板：**

```markdown
## 功能描述
描述你想要的功能

## 动机
为什么需要这个功能？解决什么问题？

## 替代方案
你考虑过哪些替代方案？

## 额外信息
其他相关信息
```

### 提交代码

1. **Fork仓库**

   点击页面右上角的"Fork"按钮

2. **克隆你的Fork**

   ```bash
   git clone git@github.com:your-username/RDKX5-YOLOv11n-.git
   cd RDKX5-YOLOv11n-
   ```

   > ⚠️ 已过时（2026-09-26）：上面的克隆地址是模板占位符，本仓库不是这个远端。实际情况：
   > git 仓库根是 `/home/ansty/RDKX5/auv_vision`，远端 `https://github.com/ansty-31/2026-Robocup-AUV.git`
   > （`git -C /home/ansty/RDKX5/auv_vision remote -v` 可核）；本项目目录 `RDKX5-YOLOv11n-/` 是该仓库的
   > **子目录**。所以在项目里跑 `git status` / `git log` 作用于父仓库、显示路径相对当前目录 —— 提交时注意
   > 别把父仓库里其它目录（`AUV` / `pc` …）的改动一起带进来。

3. **创建分支**

   ```bash
   git checkout -b feature/your-feature-name
   ```

   分支命名规范：
   - `feature/xxx` - 新功能
   - `fix/xxx` - Bug修复
   - `docs/xxx` - 文档更新
   - `refactor/xxx` - 代码重构
   - `test/xxx` - 测试相关

4. **进行修改**

   - 遵循[代码规范](#代码规范)
   - 添加必要的测试
   - 更新相关文档

5. **提交更改**

   ```bash
   git add .
   git commit -m "feat: add new feature"
   ```

   遵循[提交信息规范](#提交信息规范)

6. **推送到GitHub**

   ```bash
   git push origin feature/your-feature-name
   ```

7. **创建Pull Request**

   - 访问你的GitHub仓库页面
   - 点击"Pull Request"按钮
   - 填写PR描述，说明你的更改

## 代码规范

### Python代码

遵循PEP 8规范：

```python
# 好的示例
def process_image(img, target_size=640):
    """
    处理图像
    
    Args:
        img: 输入图像
        target_size: 目标尺寸
    
    Returns:
        处理后的图像
    """
    # 实现代码
    pass


# 类命名使用大驼峰
class YOLODetector:
    def __init__(self, model_path):
        self.model_path = model_path
    
    def detect(self, image):
        """执行检测"""
        pass


# 常量使用全大写
MAX_DETECTIONS = 100
DEFAULT_CONF_THRESH = 0.25
```

### 文档注释

所有公共函数和类都应该有文档字符串：

```python
def bgr_to_nv12(img, target_size=640):
    """
    将BGR图像转换为NV12格式
    
    Args:
        img (numpy.ndarray): BGR格式的输入图像，shape=(H, W, 3)
        target_size (int): 目标尺寸，默认640
    
    Returns:
        numpy.ndarray: NV12格式的图像数据，shape=(H*1.5, W)
    
    Example:
        >>> img = cv2.imread('test.jpg')
        >>> nv12 = bgr_to_nv12(img, 640)
    """
    # 实现代码
    pass
```

### 命名规范

- 变量名：小写下划线 `my_variable`
- 函数名：小写下划线 `my_function()`
- 类名：大驼峰 `MyClass`
- 常量：全大写 `MY_CONSTANT`
- 私有成员：前缀下划线 `_private_method()`

## 提交信息规范

使用约定式提交（Conventional Commits）：

```
<类型>(<范围>): <简短描述>

<详细描述>

<尾部>
```

### 类型

- `feat`: 新功能
- `fix`: Bug修复
- `docs`: 文档更新
- `style`: 代码格式（不影响功能）
- `refactor`: 重构
- `perf`: 性能优化
- `test`: 测试相关
- `chore`: 构建过程或辅助工具

### 示例

```bash
# 新功能
git commit -m "feat(detector): add confidence threshold parameter"

# Bug修复
git commit -m "fix(nms): correct IOU calculation error"

# 文档更新
git commit -m "docs(readme): update installation instructions"

# 性能优化
git commit -m "perf(postprocess): optimize DFL decoding speed"
```

## 代码审查流程

提交PR后：

1. **自动检查** - GitHub Actions会自动运行测试
2. **代码审查** - 维护者会审查你的代码
3. **讨论修改** - 根据反馈进行必要的修改
4. **合并** - 审查通过后，PR会被合并到主分支

> ⚠️ 已过时（2026-09-26）：第 1 条不成立 —— 仓库里**没有 `.github/` 目录**（也没有任何 CI 配置，
> `ls -a` 与 `git ls-files` 均无），PR 不会自动跑任何检查；第 2 条的「测试」也不存在（见下文「测试」节）。
> 当前的「审查」实际是人工核对产物：ONNX 张量契约、`output/*.bin` + `*_quant_info.json`、
> `runs/auv5/` 下的报表，以及线 C 的人眼判定日志（`output/preview/auv5_pose/judgment/`）。

## 项目结构与环境

> 本节为 2026-09-26 新增：**路径全部 `ls` 逐条核过**（验证命令见本节末），表内路径相对**仓库根**
> （`RDKX5-YOLOv11n-/`）书写。它同时补上了下面「开发环境设置」/「测试」两节里那套 venv + pytest
> 流程的实际情况 —— 那两节今天照抄跑不通，原因见各自下方的 ⚠️ 标注。

### 目录

| 路径 | 内容 | 权威索引（改完要同步） |
|---|---|---|
| `scripts/1_prepare/` | 切帧 / 相机标定 / 预处理 640×640 / 筛图 / 数据集重划分；`pose/` 是门 4 角点专用（`prepare_pose_dataset.py`、`map_pose_dataset.py`），`provenance/` 做素材溯源 | `scripts/README.md` |
| `scripts/2_train/` | `train_yolo11n.py`（detect / pose 共用）、`build_stage_dataset.py`、`eval_benchmark.py` | `scripts/README.md` |
| `scripts/3_export/` | `modify_ultralytics.py`（head 补丁切换）、`export_onnx.py`、`prepare_calibration.py`、`quantize.sh`（起 OE docker） | `scripts/README.md` |
| `configs/` | 量化配置（`yolo11n_config.yaml` / `gate_kpt_config.yaml`）+ 板端**只读镜像**（`vision.yaml` / `front_camera.yaml` / `front_camera_air.yaml`）+ `backup/` | `configs/README.md`（**板端为准**：先改板端、再从板子同步回来） |
| `data/` | `raw/` 原始素材 · `frames/` 切帧 · `datasets/` 训练集 · `derived/` 筛图与送标包 · `calib/` 标定 | `data/README.md` |
| `weights/` | `*.pt` / `*.onnx`：命名与代次规则、每份权重的来源与 md5 | `weights/README.md` |
| `output/` | 交付物 `.bin` + `*_quant_info.json`；`output/preview/auv5_pose/` 是线 C（人眼判定 + 分期重训）工作区 | `output/README.md`、`output/preview/auv5_pose/INDEX.md` |
| `runs/` | 训练 / 评测 / 量化记录（`runs/auv5/`、`runs/prov/`） | `runs/auv5/REPORT_auv5_pose.md` |
| `experiment/` | 域 / 增强消融实验（`experiment/scripts/`、`experiment/runs/`、`experiment/data/`） | `experiment/README.md` |
| `docs/` | 本指南、中英 README、门 4 角点打标规范 `labeling_gate_pose.md`、`tutorial_zh.md` | — |
| `cleanup_record/` | 每次整理/清理的**记录 + MOVES 表**（改路径必须在这里留痕） | `cleanup_record/reports/RESTRUCTURE_2026-09-25.md` |
| `_archive/` | 退役但**不删**的产物（旧 `runs` / 旧 `output` / 空壳目录 / 中间产物） | `_archive/README.md` |
| `raw-data/` | 素材抢救区（坏卡裸流），**按天聚类**：`20260926/`…`20261005/` ＋ `_dups/` | `raw-data/README.md` |

### 约定（改代码或改路径前必读）

- **两套环境，别混**：PC 侧所有 Python 脚本（`1_prepare` / `2_train` / `3_export`）跑在 conda `yolov8`
  （`/home/ansty/anaconda3/envs/yolov8/bin/python`，ultralytics 8.3.0 + torch 2.3.1+cu118）；
  **量化不在 conda 里**：`scripts/3_export/quantize.sh` 自己 `docker run`
  `openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8-py310` 执行 `hb_mapper makertbin`，
  容器内 `/data` = 仓库根，配置里的 `onnx_model` / `cal_data_dir` / `output_model_file_prefix`
  都写**容器内路径**，不要手写 docker 命令、也不要改成宿主机路径。
- **`head.py` 补丁生命周期**：训练 / `predict` / `val` / 筛图要**原版**，导出 ONNX 要**补丁版**
  （detect 6 输出 / pose 9 输出）。`train_yolo11n.py` 自动「restore → 训练 → 重打补丁」，
  所以训练后直接 `predict` 会因输出头分裂而报错 —— 先
  `python scripts/3_export/modify_ultralytics.py --restore`。
- **输出契约不可破**：detect **6** 张量 / pose **9** 张量；类别顺序 `[blue_ball, gate, red_ball]`
  （由数据集 `names` 顺序决定）；pose 关键点 `TL,TR,BR,BL`、通道 `[x,y,v]×4`（=12）；
  训练/导出/量化统一 `imgsz=640`，板端输入 nv12。
- **校准集按任务分区**：detect 用 `calibration_data_detect/`，pose 用 `calibration_data/`，
  混用会静默掉点（校准分布不对 → 量化阈值偏 → int8 精度掉）。
- **只归整、不删除**：路径变更写进 `cleanup_record/`（MOVES 表 + 记录 md），退役产物进 `_archive/`，
  旧权重按 `.<代次>` 后缀保留（规则见 `weights/README.md`）。
- **索引跟着产物走**：动过目录或产物，就同步更新上表的权威索引文件，别让文档路径与实际漂移。

<details>
<summary>本节路径的验证命令（2026-09-26 实跑）</summary>

```bash
cd /home/ansty/RDKX5/auv_vision/RDKX5-YOLOv11n-
ls -d scripts/{1_prepare,2_train,3_export} configs data/{raw,frames,datasets,derived,calib} \
      weights output runs experiment docs cleanup_record _archive raw-data
ls output/preview/auv5_pose/ output/preview/auv5_pose/{docs,tools,judgment,logs,tables,filelists,renders}
ls scripts/README.md data/README.md configs/README.md output/README.md \
   experiment/README.md _archive/README.md cleanup_record/reports/RESTRUCTURE_2026-09-25.md \
   docs/labeling_gate_pose.md docs/tutorial_zh.md runs/auv5/REPORT_auv5_pose.md
```

</details>

## 开发环境设置

```bash
# 1. 克隆仓库
git clone git@github.com:your-username/RDKX5-YOLOv11n-.git
cd RDKX5-YOLOv11n-

# 2. 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 3. 安装开发依赖
pip install -r requirements.txt
pip install -r requirements-dev.txt  # 包含测试、格式化工具

# 4. 安装pre-commit hooks（可选）
pre-commit install
```

> ⚠️ 已过时（2026-09-26）：上面这套流程在本仓库**照抄跑不通**，会装出一个跑不了脚本的解释器：
>
> - 实际用的是 conda `yolov8`：`/home/ansty/anaconda3/envs/yolov8/bin/python`
>   （ultralytics 8.3.0 + torch 2.3.1+cu118）。系统 `python3` 没装 torch/ultralytics，
>   直接跑仓库脚本会 `ModuleNotFoundError`。交互式先 `conda activate yolov8`，
>   非交互式就用上面那个绝对路径解释器（等价）。
> - `requirements-dev.txt` 与 `.pre-commit-config.yaml` **都不存在**（`ls` 无此文件），所以
>   `pip install -r requirements-dev.txt` 与 `pre-commit install` 都会失败；
>   `requirements.txt` 存在（ultralytics / opencv / Pillow / numpy / onnx / onnxruntime / tqdm / PyYAML）。
> - **量化不在这个环境里跑**：`scripts/3_export/quantize.sh` 自己 `docker run`
>   `openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8-py310` 执行 `hb_mapper makertbin`，
>   容器内 `/data` 挂的就是仓库根。本机 GPU 是 RTX 4060 Laptop 8GB，训练用 `--device 0`
>   （`--batch 4 --workers 2`，worker 多了会与 CUDA fork 死锁）。
>
> 环境细节见 `scripts/README.md` 顶部的「运行环境」说明。

## 测试

添加新功能时，请添加相应的测试：

```bash
# 运行所有测试
python -m pytest tests/

# 运行特定测试
python -m pytest tests/test_inference.py

# 查看覆盖率
python -m pytest --cov=rdk_deployment tests/
```

> ⚠️ 已过时（2026-09-26）：本仓库**没有 `tests/` 目录**，也没有 `rdk_deployment` 这个包，
> 上面三条命令今天都会以「file or directory not found / no tests ran」结束。当前的质量保证方式是：
>
> 1. 脚本自身的 `--help` 与 dry-run（例：`python scripts/1_prepare/resplit_dataset.py --dry-run`）；
> 2. **产物核对**：导出后核对 ONNX 张量个数与形状（detect 6 / pose 9），量化后核对
>    `output/*.bin` 与 `*_quant_info.json`；报表在 `runs/auv5/`；
> 3. **板端实测**：精度与帧率只有在 RDK X5 上跑过才算数，没实测就写「未验证」。
>
> 若将来引入自动化测试，请同时补 `tests/` 与 CI 配置，并回来把这节改回真实状态。

## 文档

更新或添加功能时，请更新相应文档：

- `README.md` - 项目主页
- `docs/` - 详细文档
- 代码注释 - 函数和类的文档字符串

> ⚠️ 已过时（2026-09-26）：除上面三条，本仓库还有一批**分区索引 / 记录文件**，动了对应目录就必须一起更新
> （否则文档里写的路径会与实际漂移）：
>
> - `scripts/README.md`（脚本索引，含路径书写约定）、`configs/README.md`（板端镜像同步流程）、
>   `data/README.md`（数据分区）、`weights/README.md`（权重清单：来源 / md5 / 使用者）、
>   `output/README.md`（交付物与量化记录）、`experiment/README.md`
> - `output/preview/auv5_pose/INDEX.md`（线 C 工作区索引；其 `docs/` 下有 RUNBOOK / CRITERIA / DECISIONS）
> - `cleanup_record/`（整理记录 + MOVES 表）、`_archive/README.md`（归档内容与恢复方法）
> - `docs/labeling_gate_pose.md`（门 4 角点打标规范；标注口径一变，这里必须同步）

## 问题？

如果你有任何问题：

- 查看[根目录 README](../README.md)（目录分区 / 全流程命令 / 注意事项）与[部署教程](tutorial_zh.md)
- 在[Discussions](https://github.com/your-username/RDKX5-YOLOv11n-/discussions)提问
- 发送邮件到 your-email@example.com

> ⚠️ 已过时（2026-09-26）：上面两条联系方式都是模板占位符（`your-username` / `your-email@example.com`），
> 指向的仓库与邮箱并不存在；真实远端是 `https://github.com/ansty-31/2026-Robocup-AUV.git`。
> 组内协作以仓库内的分区索引文件为准（见上一节），别用这两个占位符。

## 感谢

再次感谢你的贡献！每一个PR都让这个项目变得更好 ❤️
