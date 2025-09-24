#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
import sys
import time
import warnings
from typing import Optional, List, Dict, Annotated
import uuid
import hashlib
import json
from dataclasses import dataclass

import numpy as np
from fastapi import (
    FastAPI,
    HTTPException,
    UploadFile,
    File,
    Form,
    BackgroundTasks,
    Depends,
)
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import argparse

# 禁用警告
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "indextts"))

from indextts.infer_v2 import IndexTTS2


# 配置日志记录
def setup_logging(log_level: str = "INFO"):
    """配置日志记录"""
    log_format = (
        "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
    )

    # 配置根日志器
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),  # 控制台输出
            logging.FileHandler("logs/api.log", encoding="utf-8"),  # 文件输出
        ],
    )

    # 创建专用的日志器
    logger = logging.getLogger("IndexTTS-API")
    return logger


# 初始化日志器
logger = setup_logging()


class TTSRequest(BaseModel):
    """TTS请求模型"""

    text: str = Field(..., description="要合成的文本")
    prompt_audio_id: Optional[str] = Field(
        None, description="音色参考音频文件ID（通过上传接口获得）"
    )
    emo_control_method: int = Field(
        0,
        description="情感控制方式: 0=与音色参考音频相同, 1=使用情感参考音频, 2=使用情感向量控制, 3=使用情感描述文本控制",
    )
    emo_audio_id: Optional[str] = Field(
        None, description="情感参考音频文件ID（通过上传接口获得）"
    )
    emo_weight: float = Field(0.8, description="情感权重", ge=0.0, le=1.0)
    emo_text: Optional[str] = Field(None, description="情感描述文本")
    emo_random: bool = Field(False, description="情感随机采样")
    emo_vector: Optional[List[float]] = Field(
        None, description="情感向量[喜,怒,哀,惧,厌恶,低落,惊喜,平静]"
    )
    max_text_tokens_per_segment: int = Field(
        120, description="分句最大Token数", ge=20, le=500
    )

    # 高级生成参数
    do_sample: bool = Field(True, description="是否进行采样")
    top_p: float = Field(0.8, description="Top-p采样", ge=0.0, le=1.0)
    top_k: Optional[int] = Field(30, description="Top-k采样", ge=0, le=100)
    temperature: float = Field(0.8, description="采样温度", ge=0.1, le=2.0)
    length_penalty: float = Field(0.0, description="长度惩罚", ge=-2.0, le=2.0)
    num_beams: int = Field(3, description="束搜索数量", ge=1, le=10)
    repetition_penalty: float = Field(10.0, description="重复惩罚", ge=0.1, le=20.0)
    max_mel_tokens: int = Field(1500, description="最大mel token数", ge=50, le=3000)


class TTSResponse(BaseModel):
    """TTS响应模型"""

    success: bool
    message: str
    audio_url: Optional[str] = None
    task_id: Optional[str] = None
    duration: Optional[float] = None


class TaskStatus(BaseModel):
    """任务状态模型"""

    task_id: str
    status: str  # pending, processing, completed, failed
    progress: float = 0.0
    message: str = ""
    audio_url: Optional[str] = None
    duration: Optional[float] = None  # 音频时长（毫秒）
    created_at: float
    completed_at: Optional[float] = None


@dataclass
class ServerConfig:
    """服务器配置"""

    host: str = "0.0.0.0"
    port: int = 30000
    model_dir: str = "./checkpoints"
    fp16: bool = False
    deepspeed: bool = False
    cuda_kernel: bool = False
    reload: bool = False


class TTSModelService:
    """TTS模型服务类，负责模型的加载和推理"""

    def __init__(self, config: ServerConfig):
        """初始化TTS模型服务"""
        self.config = config
        self.tts = None
        self.model_version = None
        self._load_model()

    def _load_model(self):
        """加载TTS模型"""
        logger.info("开始加载TTS模型")
        logger.debug(
            f"模型配置: model_dir={self.config.model_dir}, fp16={self.config.fp16}, deepspeed={self.config.deepspeed}, cuda_kernel={self.config.cuda_kernel}"
        )

        # 检查模型目录
        if not os.path.exists(self.config.model_dir):
            logger.error(f"模型目录不存在: {self.config.model_dir}")
            raise RuntimeError(f"模型目录 {self.config.model_dir} 不存在")

        logger.info(f"模型目录验证通过: {self.config.model_dir}")

        # 检查必需的文件
        required_files = [
            "bpe.model",
            "gpt.pth",
            "config.yaml",
            "s2mel.pth",
            "wav2vec2bert_stats.pt",
        ]

        logger.info(f"检查必需文件: {required_files}")
        for file in required_files:
            file_path = os.path.join(self.config.model_dir, file)
            if not os.path.exists(file_path):
                logger.error(f"必需文件不存在: {file_path}")
                raise RuntimeError(f"必需文件 {file_path} 不存在")
            logger.debug(f"文件存在: {file_path}")

        logger.info("所有必需文件验证通过")

        # 初始化TTS模型
        config_path = os.path.join(self.config.model_dir, "config.yaml")
        logger.info(f"初始化IndexTTS2模型，配置文件: {config_path}")

        try:
            self.tts = IndexTTS2(
                model_dir=self.config.model_dir,
                cfg_path=config_path,
                use_fp16=self.config.fp16,
                use_deepspeed=self.config.deepspeed,
                use_cuda_kernel=self.config.cuda_kernel,
            )

            self.model_version = getattr(self.tts, "model_version", "1.0")
            logger.info(f"TTS模型加载成功，版本: {self.model_version}")

        except Exception as e:
            logger.error(f"TTS模型加载失败: {e}")
            raise

    def is_ready(self) -> bool:
        """检查模型是否已准备好"""
        return self.tts is not None

    def infer(self, **kwargs):
        """执行TTS推理"""
        if not self.is_ready():
            logger.error("TTS模型未加载，无法执行推理")
            raise RuntimeError("TTS模型未加载")

        logger.debug(f"开始执行TTS推理，参数数量: {len(kwargs)}")

        try:
            result = self.tts.infer(**kwargs)
            logger.debug("TTS推理成功完成")
            return result
        except Exception as e:
            logger.error(f"TTS推理失败: {e}")
            raise


class TTSTaskManager:
    """TTS任务管理器"""

    def __init__(self):
        self.tasks: Dict[str, TaskStatus] = {}
        # 创建输出目录
        os.makedirs("outputs/api", exist_ok=True)
        os.makedirs("uploads", exist_ok=True)
        logger.info("任务管理器初始化完成，输出目录已创建")

    def get_task(self, task_id: str) -> Optional[TaskStatus]:
        """获取任务状态"""
        return self.tasks.get(task_id)

    def create_task(self) -> str:
        """创建新任务"""
        task_id = str(uuid.uuid4()).replace("-", "")
        task_status = TaskStatus(
            task_id=task_id,
            status="pending",
            message="任务已创建",
            created_at=time.time(),
        )
        self.tasks[task_id] = task_status
        logger.info(f"创建新任务: {task_id}")
        return task_id

    def update_task(self, task_id: str, **kwargs):
        """更新任务状态"""
        if task_id in self.tasks:
            old_status = self.tasks[task_id].status
            for key, value in kwargs.items():
                setattr(self.tasks[task_id], key, value)

            # 日志记录状态变更
            if "status" in kwargs and kwargs["status"] != old_status:
                logger.info(
                    f"任务 {task_id} 状态更新: {old_status} -> {kwargs['status']}"
                )
            if "progress" in kwargs:
                logger.debug(f"任务 {task_id} 进度更新: {kwargs['progress']*100:.1f}%")
        else:
            logger.warning(f"尝试更新不存在的任务: {task_id}")

    def delete_task(self, task_id: str) -> bool:
        """删除任务"""
        if task_id in self.tasks:
            # 删除音频文件
            audio_path = os.path.join("outputs/api", f"tts_{task_id}.wav")
            if os.path.exists(audio_path):
                os.remove(audio_path)
                logger.debug(f"删除音频文件: {audio_path}")

            # 删除任务记录
            del self.tasks[task_id]
            logger.info(f"任务已删除: {task_id}")
            return True

        logger.warning(f"尝试删除不存在的任务: {task_id}")
        return False

    def list_tasks(
        self, limit: int = 50, status: Optional[str] = None
    ) -> List[TaskStatus]:
        """列出任务"""
        tasks = list(self.tasks.values())

        # 按状态过滤
        if status:
            tasks = [t for t in tasks if t.status == status]

        # 按创建时间排序并限制数量
        tasks.sort(key=lambda x: x.created_at, reverse=True)
        return tasks[:limit]


class TTSFileManager:
    """TTS文件管理器"""

    MAPPING_FILE = os.path.join("uploads", "config.json")

    @staticmethod
    def load_file_mapping() -> Dict[str, str]:
        """加载文件ID到文件名的映射"""
        if not os.path.exists(TTSFileManager.MAPPING_FILE):
            return {}
        
        try:
            with open(TTSFileManager.MAPPING_FILE, "r", encoding="utf-8") as f:
                return json.loads(f.read())
        except (json.JSONDecodeError, FileNotFoundError):
            logger.warning("文件映射文件损坏或不存在，创建新的映射")
            return {}

    @staticmethod
    def save_file_mapping(mapping: Dict[str, str]):
        """保存文件ID到文件名的映射"""
        os.makedirs(os.path.dirname(TTSFileManager.MAPPING_FILE), exist_ok=True)
        try:
            with open(TTSFileManager.MAPPING_FILE, "w", encoding="utf-8") as f:
                f.write(json.dumps(mapping, ensure_ascii=False, indent=2))
        except Exception as e:
            logger.error(f"保存文件映射失败: {e}")
            raise

    @staticmethod
    def add_file_mapping(file_id: str, filename: str):
        """添加文件ID到文件名的映射"""
        mapping = TTSFileManager.load_file_mapping()
        mapping[file_id] = filename
        TTSFileManager.save_file_mapping(mapping)
        logger.debug(f"添加文件映射: {file_id} -> {filename}")

    @staticmethod
    def normalize_emo_vec(emo_vec: List[float]) -> List[float]:
        """标准化情感向量"""
        k_vec = [0.75, 0.70, 0.80, 0.80, 0.75, 0.75, 0.55, 0.45]
        tmp = np.array(k_vec) * np.array(emo_vec)
        if np.sum(tmp) > 0.8:
            tmp = tmp * 0.8 / np.sum(tmp)
        return tmp.tolist()

    @staticmethod
    def validate_and_get_file_path(file_id: Optional[str]) -> Optional[str]:
        """验证文件ID并返回安全的文件路径"""
        if not file_id:
            return None

        # 基本格式验证：只允许字母数字、下划线和连字符，防止路径遍历攻击
        if not isinstance(file_id, str):
            raise ValueError(f"无效的文件ID类型: {type(file_id)}")

        # 验证文件ID只包含安全字符
        if not file_id.replace("_", "").replace("-", "").isalnum():
            raise ValueError(
                f"无效的文件ID格式，只能包含字母、数字、下划线和连字符: {file_id}"
            )

        # 验证长度（防止过长的文件名）
        if len(file_id) == 0 or len(file_id) > 100:
            raise ValueError(f"文件ID长度必须在1-100个字符之间: {file_id}")

        # 从映射文件中查找对应的文件名
        mapping = TTSFileManager.load_file_mapping()
        if file_id not in mapping:
            raise ValueError(f"文件ID {file_id} 不存在")

        filename = mapping[file_id]
        file_path = os.path.join("uploads", filename)

        # 验证文件确实存在且可读
        if not os.path.isfile(file_path):
            raise ValueError(f"文件 {file_path} 不存在或不是文件")

        return file_path

    @staticmethod
    def calculate_file_hash(file_content: bytes) -> str:
        """计算文件内容的SHA256哈希值"""
        return hashlib.sha256(file_content).hexdigest()

    @staticmethod
    def validate_audio_file(file_path: str) -> tuple[bool, str]:
        """验证音频文件是否可以正确读取

        Returns:
            tuple[bool, str]: (是否有效, 错误信息)
        """
        logger.debug(f"开始验证音频文件: {file_path}")

        try:
            import torchaudio

            # 尝试使用torchaudio读取（项目主要使用的库）
            audio, sr = torchaudio.load(file_path)

            # 基本验证：确保有音频数据
            if audio.size(0) == 0 or audio.size(1) == 0:
                logger.warning(f"音频文件为空: {file_path}")
                return False, "音频文件为空"

            # 验证采样率合理性
            if sr < 8000 or sr > 192000:
                logger.warning(f"音频采样率异常: {sr}Hz")
                return False, f"音频采样率异常: {sr}Hz"

            duration = audio.size(1) / sr
            logger.debug(f"音频文件验证成功: 采样率{sr}Hz, 时长{duration:.2f}秒")
            return True, f"验证成功，采样率: {sr}Hz，时长: {duration:.2f}秒"

        except Exception as e:
            logger.error(f"音频文件验证失败: {file_path}, 错误: {e}")
            return False, f"音频格式不支持: {str(e)}"

    @staticmethod
    def get_audio_duration_ms(file_path: str) -> Optional[float]:
        """获取音频文件时长（毫秒）
        
        Returns:
            Optional[float]: 音频时长（毫秒），如果获取失败返回None
        """
        try:
            import torchaudio
            
            # 使用torchaudio获取音频信息
            audio_info = torchaudio.info(file_path)
            duration_seconds = audio_info.num_frames / audio_info.sample_rate
            duration_ms = duration_seconds * 1000
            
            logger.debug(f"音频时长: {file_path} -> {duration_ms:.1f}ms")
            return duration_ms
            
        except Exception as e:
            logger.error(f"获取音频时长失败: {file_path}, 错误: {e}")
            return None


class TTSService:
    """TTS业务服务类，整合模型服务和任务管理"""

    def __init__(self, model_service: TTSModelService, task_manager: TTSTaskManager):
        self.model_service = model_service
        self.task_manager = task_manager

    def generate_tts(self, request: TTSRequest, task_id: str) -> str:
        """生成TTS音频"""
        logger.info(f"开始生成TTS音频 - 任务ID: {task_id}")
        logger.debug(
            f"TTS请求参数: text_length={len(request.text)}, emo_method={request.emo_control_method}"
        )

        try:
            # 更新任务状态
            self.task_manager.update_task(task_id, status="processing", progress=0.1)

            # 准备输出路径
            output_path = os.path.join("outputs/api", f"tts_{task_id}.wav")
            logger.debug(f"输出文件路径: {output_path}")

            # 验证并获取音频文件路径
            prompt_audio_path = TTSFileManager.validate_and_get_file_path(
                request.prompt_audio_id
            )
            emo_ref_path = TTSFileManager.validate_and_get_file_path(
                request.emo_audio_id
            )

            logger.debug(
                f"音频文件路径 - 音色: {prompt_audio_path}, 情感: {emo_ref_path}"
            )

            # 处理情感控制参数
            emo_weight = request.emo_weight
            vec = None

            if request.emo_control_method == 0:  # emotion from speaker
                emo_ref_path = None
            elif request.emo_control_method == 1:  # emotion from reference audio
                emo_weight = emo_weight * 0.8  # normalize for better experience
                if not emo_ref_path:
                    logger.error("情感控制方式为1但未提供情感参考音频")
                    raise ValueError("使用情感参考音频时必须提供emo_audio_id")
            elif request.emo_control_method == 2:  # emotion from custom vectors
                if request.emo_vector and len(request.emo_vector) == 8:
                    vec = TTSFileManager.normalize_emo_vec(request.emo_vector)
                else:
                    logger.error(
                        f"情感向量格式错误，期望8个值，实际: {len(request.emo_vector) if request.emo_vector else 'None'}"
                    )
                    raise ValueError("情感向量必须包含8个值")

            # 处理情感描述文本
            emo_text = request.emo_text if request.emo_text else None

            # 准备生成参数
            kwargs = {
                "do_sample": request.do_sample,
                "top_p": request.top_p,
                "top_k": request.top_k if request.top_k and request.top_k > 0 else None,
                "temperature": request.temperature,
                "length_penalty": request.length_penalty,
                "num_beams": request.num_beams,
                "repetition_penalty": request.repetition_penalty,
                "max_mel_tokens": request.max_mel_tokens,
            }

            # 更新进度
            self.task_manager.update_task(task_id, progress=0.3)

            logger.info(f"开始执行TTS推理 - 任务ID: {task_id}")

            # 执行TTS生成
            output = self.model_service.infer(
                spk_audio_prompt=prompt_audio_path,
                text=request.text,
                output_path=output_path,
                emo_audio_prompt=emo_ref_path,
                emo_alpha=emo_weight,
                emo_vector=vec,
                use_emo_text=(request.emo_control_method == 3),
                emo_text=emo_text,
                use_random=request.emo_random,
                max_text_tokens_per_segment=request.max_text_tokens_per_segment,
                **kwargs,
            )

            logger.info(f"TTS生成成功 - 任务ID: {task_id}, 输出: {output}")

            # 计算音频时长
            audio_duration_ms = TTSFileManager.get_audio_duration_ms(output_path)
            if audio_duration_ms is not None:
                logger.debug(f"音频时长: {audio_duration_ms:.1f}ms")
            
            # 更新任务状态
            self.task_manager.update_task(
                task_id,
                status="completed",
                progress=1.0,
                audio_url=f"/api/v1/audio/{task_id}.wav",
                duration=audio_duration_ms,
                completed_at=time.time(),
                message="生成完成",
            )

            return output

        except Exception as e:
            logger.error(f"TTS生成失败 - 任务ID: {task_id}, 错误: {e}")

            # 更新任务状态为失败
            self.task_manager.update_task(
                task_id, status="failed", message=str(e), completed_at=time.time()
            )
            raise e


def cleanup_old_files():
    """清理旧文件"""
    logger.info("开始清理旧文件")
    current_time = time.time()
    cleaned_count = 0

    # 清理超过24小时的输出文件
    outputs_dir = "outputs/api"
    if os.path.exists(outputs_dir):
        logger.debug(f"检查输出目录: {outputs_dir}")
        for filename in os.listdir(outputs_dir):
            file_path = os.path.join(outputs_dir, filename)
            if os.path.getctime(file_path) < current_time - 24 * 3600:
                try:
                    os.remove(file_path)
                    cleaned_count += 1
                    logger.debug(f"已删除过期文件: {file_path}")
                except Exception as e:
                    logger.warning(f"删除文件失败: {file_path}, 错误: {e}")

    logger.info(f"文件清理完成，已删除 {cleaned_count} 个文件")

    # 清理超过1小时的上传文件（暂时禁用）
    # uploads_dir = "uploads"
    # if os.path.exists(uploads_dir):
    #     for filename in os.listdir(uploads_dir):
    #         file_path = os.path.join(uploads_dir, filename)
    #         if os.path.getctime(file_path) < current_time - 3600:
    #             os.remove(file_path)


# 全局服务实例
tts_service: Optional[TTSService] = None


def get_tts_service() -> TTSService:
    """获取TTS服务实例（依赖注入）"""
    global tts_service

    if tts_service is None:
        logger.error("TTS服务未初始化，这不应该发生")
        raise RuntimeError("TTS服务未初始化")

    return tts_service


def init_tts_service(config: ServerConfig) -> TTSService:
    """初始化TTS服务"""
    logger.info("IndexTTS API服务器初始化开始")
    logger.info(
        f"配置信息: host={config.host}, port={config.port}, model_dir={config.model_dir}"
    )

    try:
        # 初始化模型服务
        logger.info("初始化模型服务")
        model_service = TTSModelService(config)

        # 初始化任务管理器
        logger.info("初始化任务管理器")
        task_manager = TTSTaskManager()

        # 创建 TTS 服务
        logger.info("创建 TTS 服务实例")
        service = TTSService(model_service, task_manager)

        logger.info(f"TTS模型加载完成，版本: {model_service.model_version}")
        logger.info(f"API服务器启动在 http://{config.host}:{config.port}")
        logger.info(f"API文档地址: http://{config.host}:{config.port}/docs")

        return service

    except Exception as e:
        logger.error(f"TTS服务初始化失败: {e}")
        raise


app = FastAPI(
    title="IndexTTS API Server",
    version="1.0.0",
    description="IndexTTS文本转语音API服务",
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """根路径"""
    return {"message": "IndexTTS API Server", "version": "1.0.0"}


@app.get("/api/v1/health")
async def health_check(service: Annotated[TTSService, Depends(get_tts_service)]):
    """健康检查"""
    logger.debug("执行健康检查")
    is_ready = service.model_service.is_ready()
    result = {"status": "healthy", "model_loaded": is_ready}
    logger.debug(f"健康检查结果: {result}")
    return result


@app.post("/api/v1/tts", response_model=TTSResponse)
async def generate_tts(
    request: TTSRequest,
    background_tasks: BackgroundTasks,
    service: Annotated[TTSService, Depends(get_tts_service)],
):
    """生成TTS音频（异步）"""
    logger.info(f"接收异步TTS请求 - 文本长度: {len(request.text)}")
    logger.debug(f"请求详情: {request.model_dump()}")

    # 创建任务
    task_id = service.task_manager.create_task()

    # 添加后台任务
    background_tasks.add_task(service.generate_tts, request, task_id)
    logger.info(f"异步TTS任务已加入队列 - 任务ID: {task_id}")

    return TTSResponse(
        success=True, message="任务已创建，请使用task_id查询状态", task_id=task_id
    )


@app.post("/api/v1/tts/sync", response_model=TTSResponse)
async def generate_tts_sync(
    request: TTSRequest, service: Annotated[TTSService, Depends(get_tts_service)]
):
    """生成TTS音频（同步）"""
    logger.info(f"接收同步TTS请求 - 文本长度: {len(request.text)}")
    logger.debug(f"同步请求详情: {request.dict()}")

    start_time = time.time()
    task_id = service.task_manager.create_task()

    try:
        # 同步生成
        logger.info(f"开始同步生成TTS - 任务ID: {task_id}")
        output_path = service.generate_tts(request, task_id)
        processing_time = time.time() - start_time

        # 获取生成的音频时长
        task = service.task_manager.get_task(task_id)
        audio_duration_ms = task.duration if task else None

        logger.info(f"同步TTS生成成功 - 任务ID: {task_id}, 处理耗时: {processing_time:.2f}秒, 音频时长: {audio_duration_ms}ms")

        return TTSResponse(
            success=True,
            message="生成完成",
            audio_url=f"/api/v1/audio/{task_id}.wav",
            task_id=task_id,
            duration=audio_duration_ms,
        )

    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(
            f"同步TTS生成失败 - 任务ID: {task_id}, 处理耗时: {processing_time:.2f}秒, 错误: {e}"
        )

        return TTSResponse(success=False, message=f"生成失败{str(e)}", task_id=task_id, duration=None)


@app.get("/api/v1/task/{task_id}", response_model=TaskStatus)
async def get_task_status(
    task_id: str, service: Annotated[TTSService, Depends(get_tts_service)]
):
    """获取任务状态"""
    logger.debug(f"查询任务状态 - 任务ID: {task_id}")

    task = service.task_manager.get_task(task_id)
    if not task:
        logger.warning(f"查询不存在的任务 - 任务ID: {task_id}")
        raise HTTPException(status_code=404, detail="任务不存在")

    logger.debug(f"任务状态: {task.status}, 进度: {task.progress}")
    return task


@app.get("/api/v1/audio/{task_id}.wav")
async def get_audio(
    task_id: str, service: Annotated[TTSService, Depends(get_tts_service)]
):
    """获取生成的音频文件（支持.wav后缀，兼容各种客户端）"""
    logger.info(f"请求下载音频文件 - 任务ID: {task_id}")

    task = service.task_manager.get_task(task_id)
    if not task:
        logger.warning(f"音频下载失败，任务不存在 - 任务ID: {task_id}")
        raise HTTPException(status_code=404, detail="任务不存在")

    if task.status != "completed":
        logger.warning(
            f"音频下载失败，任务未完成 - 任务ID: {task_id}, 状态: {task.status}"
        )
        raise HTTPException(status_code=400, detail="音频尚未生成完成")

    audio_path = os.path.join("outputs/api", f"tts_{task_id}.wav")
    if not os.path.exists(audio_path):
        logger.error(f"音频文件不存在 - 路径: {audio_path}")
        raise HTTPException(status_code=404, detail="音频文件不存在")

    logger.info(f"音频文件下载开始 - 任务ID: {task_id}, 文件: {audio_path}")

    return FileResponse(
        audio_path, media_type="audio/wav", filename=f"tts_{task_id}.wav"
    )


@app.post("/api/v1/upload/audio")
async def upload_audio(
    file: UploadFile = File(...), file_id: str = Form(...)
):
    """上传音频文件"""
    logger.info(f"接收音频文件上传请求 - filename: {file.filename}, file_id: {file_id}")

    if not file.filename.lower().endswith((".wav", ".mp3", ".m4a", ".flac")):
        logger.warning(f"不支持的音频格式 - filename: {file.filename}")
        raise HTTPException(status_code=400, detail="不支持的音频格式")

    logger.debug(f"开始读取文件内容 - filename: {file.filename}")
    # 读取文件内容
    content = await file.read()
    logger.debug(f"文件读取完成 - 文件大小: {len(content)} bytes")

    # 验证用户提供的file_id格式（不允许包含路径分隔符等危险字符）
    if not file_id.replace("_", "").replace("-", "").isalnum():
        logger.warning(f"无效的file_id格式 - file_id: {file_id}")
        raise HTTPException(
            status_code=400, detail="file_id只能包含字母、数字、下划线和连字符"
        )
    if len(file_id) > 100:  # 限制长度避免文件名过长
        logger.warning(f"file_id过长 - file_id: {file_id}")
        raise HTTPException(status_code=400, detail="file_id长度不能超过100个字符")

    # 统一使用SHA256哈希作为实际文件名
    sha256_hash = TTSFileManager.calculate_file_hash(content)
    file_extension = os.path.splitext(file.filename)[1]
    saved_filename = f"{sha256_hash}{file_extension}"
    saved_path = os.path.join("uploads", saved_filename)
    
    logger.info(f"用户file_id: {file_id}, 实际保存文件名: {saved_filename}")
    
    # 保存新文件
    logger.debug(f"保存文件 - 路径: {saved_path}")
    with open(saved_path, "wb") as buffer:
        buffer.write(content)

    # 验证音频文件格式
    logger.debug("开始验证音频文件格式")
    is_valid, message = TTSFileManager.validate_audio_file(saved_path)
    if not is_valid:
        # 验证失败，删除文件并返回错误
        logger.warning(f"音频文件验证失败，删除文件: {saved_path}")
        os.remove(saved_path)
        raise HTTPException(status_code=400, detail=f"音频文件格式错误: {message}")

    # 添加文件ID到文件名的映射关系
    try:
        TTSFileManager.add_file_mapping(file_id, saved_filename)
        logger.info(f"成功添加文件映射: {file_id} -> {saved_filename}")
    except Exception as e:
        # 映射添加失败，删除文件
        logger.error(f"添加文件映射失败，删除文件: {saved_path}")
        os.remove(saved_path)
        raise HTTPException(status_code=500, detail=f"保存文件映射失败: {str(e)}")

    logger.info(
        f"文件上传成功 - 文件ID: {file_id}, 实际文件名: {saved_filename}, 大小: {len(content)} bytes, 验证: {message}"
    )

    return {
        "success": True,
        "message": f"文件上传成功 - {message}",
        "file_id": file_id,
        "original_filename": file.filename,
        "saved_filename": saved_filename,
        "file_size": len(content),
        "upload_time": time.time(),
    }


@app.get("/api/v1/file/{file_id}")
async def get_file_info(file_id: str):
    """获取上传文件信息"""
    logger.debug(f"查询文件信息 - 文件ID: {file_id}")

    try:
        file_path = TTSFileManager.validate_and_get_file_path(file_id)
        if not file_path:
            logger.warning(f"文件不存在 - 文件ID: {file_id}")
            raise HTTPException(status_code=404, detail="文件不存在")

        # 获取文件信息
        file_stat = os.stat(file_path)
        filename = os.path.basename(file_path)

        logger.debug(f"文件信息查询成功 - 文件ID: {file_id}, 文件名: {filename}")

        return {
            "file_id": file_id,
            "filename": filename,
            "file_size": file_stat.st_size,
            "created_time": file_stat.st_ctime,
            "modified_time": file_stat.st_mtime,
        }
    except ValueError as e:
        logger.warning(f"文件信息查询失败 - 文件ID: {file_id}, 错误: {e}")
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/v1/files")
async def get_file_mapping():
    """获取所有文件ID到文件名的映射关系"""
    logger.debug("查询文件映射关系")
    
    try:
        mapping = TTSFileManager.load_file_mapping()
        logger.debug(f"返回映射数量: {len(mapping)}")
        
        return {
            "success": True,
            "files": mapping,
            "count": len(mapping)
        }
    except Exception as e:
        logger.error(f"获取文件映射失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取文件映射失败: {str(e)}")


@app.get("/api/v1/tasks")
async def list_tasks(
    service: Annotated[TTSService, Depends(get_tts_service)],
    limit: int = 50,
    status: Optional[str] = None,
):
    """列出任务"""
    logger.debug(f"查询任务列表 - limit: {limit}, status: {status}")

    tasks = service.task_manager.list_tasks(limit=limit, status=status)
    logger.debug(f"返回任务数量: {len(tasks)}")

    return {"tasks": tasks}


@app.delete("/api/v1/task/{task_id}")
async def delete_task(
    task_id: str, service: Annotated[TTSService, Depends(get_tts_service)]
):
    """删除任务和相关文件"""
    logger.info(f"请求删除任务 - 任务ID: {task_id}")

    if not service.task_manager.delete_task(task_id):
        logger.warning(f"删除任务失败，任务不存在 - 任务ID: {task_id}")
        raise HTTPException(status_code=404, detail="任务不存在")

    logger.info(f"任务删除成功 - 任务ID: {task_id}")
    return {"success": True, "message": "任务已删除"}


def create_server_config_from_env() -> ServerConfig:
    """从环境变量创建服务器配置"""
    return ServerConfig(
        host=os.environ.get("INDEXTTS_HOST", "0.0.0.0"),
        port=int(os.environ.get("INDEXTTS_PORT", "30000")),
        model_dir=os.environ.get("INDEXTTS_MODEL_DIR", "./checkpoints"),
        fp16=os.environ.get("INDEXTTS_FP16", "false").lower() == "true",
        deepspeed=os.environ.get("INDEXTTS_DEEPSPEED", "false").lower() == "true",
        cuda_kernel=os.environ.get("INDEXTTS_CUDA_KERNEL", "false").lower() == "true",
        reload=os.environ.get("INDEXTTS_RELOAD", "false").lower() == "true",
    )


def main():
    """主函数"""
    global tts_service

    # 先设置基本的日志，供启动过程使用
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    main_logger = logging.getLogger("IndexTTS-Main")

    parser = argparse.ArgumentParser(
        description="IndexTTS API Server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务器主机地址")
    parser.add_argument("--port", type=int, default=30000, help="服务器端口")
    parser.add_argument(
        "--model_dir", type=str, default="./checkpoints", help="模型检查点目录"
    )
    parser.add_argument("--fp16", action="store_true", help="使用FP16推理")
    parser.add_argument("--deepspeed", action="store_true", help="使用DeepSpeed加速")
    parser.add_argument("--cuda_kernel", action="store_true", help="使用CUDA内核")
    parser.add_argument("--reload", action="store_true", help="开发模式热重载")

    args = parser.parse_args()

    # 创建服务器配置
    config = ServerConfig(
        host=args.host,
        port=args.port,
        model_dir=args.model_dir,
        fp16=args.fp16,
        deepspeed=args.deepspeed,
        cuda_kernel=args.cuda_kernel,
        reload=args.reload,
    )

    # 将配置写入环境变量，供服务读取
    os.environ["INDEXTTS_MODEL_DIR"] = config.model_dir
    os.environ["INDEXTTS_HOST"] = config.host
    os.environ["INDEXTTS_PORT"] = str(config.port)
    os.environ["INDEXTTS_FP16"] = str(config.fp16).lower()
    os.environ["INDEXTTS_DEEPSPEED"] = str(config.deepspeed).lower()
    os.environ["INDEXTTS_CUDA_KERNEL"] = str(config.cuda_kernel).lower()

    try:
        main_logger.info("启动IndexTTS API服务器")
        main_logger.info(f"服务配置: {config}")

        # 在服务器启动前初始化TTS服务
        main_logger.info("开始初始化TTS服务")
        tts_service = init_tts_service(config)
        main_logger.info("TTS服务初始化完成")

        # 启动服务器（使用单worker避免模型重复加载）
        main_logger.info("启动Uvicorn服务器")
        uvicorn.run(
            app,  # 直接传递app对象，而不是字符串
            host=config.host,
            port=config.port,
            workers=1,  # 固定使用单worker
            reload=config.reload,
            log_config=None,  # 使用我们自己的日志配置
        )

    except KeyboardInterrupt:
        main_logger.info("服务器被用户中断")
    except Exception as e:
        main_logger.error(f"服务器启动失败: {e}")
        sys.exit(1)
    finally:
        # 清理资源
        main_logger.info("正在关闭服务器...")
        cleanup_old_files()
        main_logger.info("服务器已关闭")


if __name__ == "__main__":
    main()
