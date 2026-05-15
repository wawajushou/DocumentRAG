"""
config.py

配置中心 - 使用 Pydantic Settings 进行配置管理

优势：
1. 启动时自动校验所有配置项，拦截错误配置
2. 类型安全，IDE 自动补全
3. 支持环境变量自动加载
4. 支持配置项范围校验（如 temperature 在 0-2 之间）
"""

from typing import Optional, List
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    应用配置类

    所有配置项都从环境变量或 .env 文件加载，
    并在实例化时进行严格的类型和范围校验。

    使用方法：
        settings = Settings()  # 自动从 .env 加载
        print(settings.chunk_size)  # 800
        print(settings.api_provider)  # "deepseek"
    """

    # ========== API 提供商配置 ==========
    api_provider: str = Field(
        default="deepseek",
        description="LLM 提供商名称（deepseek 或 mimo）"
    )

    # DeepSeek 配置
    deepseek_api_key: Optional[str] = Field(default=None, description="DeepSeek API 密钥")
    deepseek_api_base_url: str = Field(default="https://api.deepseek.com", description="DeepSeek API 地址")
    deepseek_model_name: str = Field(default="deepseek-chat", description="DeepSeek 模型名称")

    # MiMo 配置
    mimo_api_key: Optional[str] = Field(default=None, description="MiMo API 密钥")
    mimo_api_base_url: str = Field(default="https://api.xiaomimimo.com/v1", description="MiMo API 地址")
    mimo_model_name: str = Field(default="mimo-v2-pro", description="MiMo 模型名称")

    # ========== 模型配置 ==========
    embedding_model_name: str = Field(
        default="BAAI/bge-small-zh-v1.5",
        description="向量嵌入模型名称"
    )

    # ========== 文档处理配置 ==========
    chunk_size: int = Field(
        default=800,
        ge=100,
        le=2000,
        description="文本分块大小（字符数）"
    )
    chunk_overlap: int = Field(
        default=200,
        ge=0,
        description="文本分块重叠大小（字符数）"
    )

    # ========== 检索配置 ==========
    retrieve_top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="检索返回的文档数量"
    )
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="LLM 温度参数（0-2，越低越确定）"
    )

    # ========== 路径配置 ==========
    vector_db_path: str = Field(default="./chroma_db", description="向量数据库路径")
    data_dir: str = Field(default="./data", description="文档数据目录")
    model_scope_cache_dir: str = Field(default="./models", description="模型缓存目录")

    # ========== 功能开关 ==========
    use_model_scope: bool = Field(default=True, description="是否使用 ModelScope 下载模型")
    use_unstructured: bool = Field(default=False, description="是否使用 Unstructured 加载器")

    # ========== 支持的文件格式 ==========
    supported_extensions: List[str] = Field(
        default=[".docx", ".pdf", ".txt"],
        description="支持的文档格式"
    )

    # ========== 配置校验 ==========
    @field_validator("api_provider")
    @classmethod
    def validate_api_provider(cls, v: str) -> str:
        """校验 API 提供商名称"""
        allowed = ["deepseek", "mimo"]
        if v.lower() not in allowed:
            raise ValueError(f"api_provider 必须是 {allowed} 之一，当前值: {v}")
        return v.lower()

    @field_validator("chunk_overlap")
    @classmethod
    def validate_chunk_overlap(cls, v: int, info) -> int:
        """校验 chunk_overlap 必须小于 chunk_size"""
        if "chunk_size" in info.data and v >= info.data["chunk_size"]:
            raise ValueError(f"chunk_overlap ({v}) 必须小于 chunk_size ({info.data['chunk_size']})")
        return v

    def get_active_api_key(self) -> str:
        """
        获取当前激活提供商的 API Key

        Returns:
            str: API 密钥

        Raises:
            ValueError: 未配置对应的 API Key
        """
        if self.api_provider == "deepseek":
            if not self.deepseek_api_key:
                raise ValueError("未配置 DEEPSEEK_API_KEY，请在 .env 文件中设置")
            return self.deepseek_api_key
        elif self.api_provider == "mimo":
            if not self.mimo_api_key:
                raise ValueError("未配置 MIMO_API_KEY，请在 .env 文件中设置")
            return self.mimo_api_key
        else:
            raise ValueError(f"不支持的提供商: {self.api_provider}")

    def get_active_base_url(self) -> str:
        """
        获取当前激活提供商的 API 地址

        Returns:
            str: API 基础 URL
        """
        if self.api_provider == "deepseek":
            return self.deepseek_api_base_url
        elif self.api_provider == "mimo":
            return self.mimo_api_base_url
        raise ValueError(f"不支持的提供商: {self.api_provider}")

    def get_active_model_name(self) -> str:
        """
        获取当前激活提供商的模型名称

        Returns:
            str: 模型名称
        """
        if self.api_provider == "deepseek":
            return self.deepseek_model_name
        elif self.api_provider == "mimo":
            return self.mimo_model_name
        raise ValueError(f"不支持的提供商: {self.api_provider}")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False  # 环境变量不区分大小写


# 全局配置单例（延迟初始化）
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """
    获取全局配置单例

    使用延迟初始化，确保配置在首次访问时才加载，
    便于单元测试时替换配置。

    Returns:
        Settings: 配置对象
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """
    重置全局配置（仅用于测试）

    在单元测试中，可以调用此函数清除缓存的配置，
    然后重新加载新的配置。
    """
    global _settings
    _settings = None
