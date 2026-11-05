# 使用说明

面向不熟悉命令行的用户。

## 首次安装

1. 安装 Python 3.11 或更高版本
2. 在项目目录打开命令行，执行：

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -e .
copy .env.example .env
```

3. 编辑 `.env`，填入 `DEEPSEEK_API_KEY` 或 `MIMO_API_KEY`
4. 将文档放入 `data` 文件夹
5. 执行 `docrag sync`

## 日常使用

双击 `start.bat`，在 `问题 >` 提示符后输入问题并回车。

| 命令 | 作用 |
|---|---|
| 问题 | 提问 |
| `/update` | 重新同步文档 |
| `/status` | 查看索引状态 |
| `/quit` | 退出 |

## 文档更新

将新文档放入 `data`，在窗口中输入 `/update`。程序只处理改动过的文件，删除的文件会自动清理。

## 常见问题

**中文乱码**：使用 Windows Terminal 打开，不要使用 PowerShell ISE。

**重建索引**：删除 `.docrag` 目录，重新执行 `/update`。

**更换电脑**：拷贝项目目录（`.venv`、`models` 可省略），重新执行安装步骤。
