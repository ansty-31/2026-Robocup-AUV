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

## 文档

更新或添加功能时，请更新相应文档：

- `README.md` - 项目主页
- `docs/` - 详细文档
- 代码注释 - 函数和类的文档字符串

## 问题？

如果你有任何问题：

- 查看[根目录 README](../README.md)（目录分区 / 全流程命令 / 注意事项）与[部署教程](tutorial_zh.md)
- 在[Discussions](https://github.com/your-username/RDKX5-YOLOv11n-/discussions)提问
- 发送邮件到 your-email@example.com

## 感谢

再次感谢你的贡献！每一个PR都让这个项目变得更好 ❤️
