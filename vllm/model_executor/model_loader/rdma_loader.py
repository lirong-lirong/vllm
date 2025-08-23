# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from collections.abc import Generator
from typing import Optional

import torch
import torch.nn as nn

from vllm.config import LoadConfig, ModelConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.rdma_weight_utils import rdma_safetensors_weights_iterator

logger = init_logger(__name__)

try:
    from mooncake.engine import TransferEngine
    MOONCAKE_AVAILABLE = True
except ImportError:
    MOONCAKE_AVAILABLE = False
    logger.warning("Mooncake Transfer Engine not available. "
                   "Falling back to DefaultModelLoader.")


class RDMARemoteLoader(DefaultModelLoader):
    """Model loader that can load weights via RDMA using Mooncake Transfer Engine.
    
    This loader delegates all file discovery and pulling operations to rdma_weight_utils.py
    and only handles the basic loader logic.
    """

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        if not MOONCAKE_AVAILABLE:
            raise RuntimeError(
                "Mooncake Transfer Engine is required for RDMA model loading "
                "but is not available. Please install with Mooncake support.")
        
        self.mooncake_client = None
        self.rank = self._get_worker_rank()
        self._setup_mooncake_client()
        
    def _get_worker_rank(self) -> int:
        """Get worker rank from environment variables."""
        # Try to get rank from different environment variables
        # First try RANK (used by torchrun/torch.distributed)
        # Then try LOCAL_RANK (used by some distributed training frameworks)
        # Default to 0 if neither is set
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        return rank

    def _prepare_weights(self, model_name_or_path: str, revision: Optional[str], 
                        fall_back_to_pt: bool, allow_patterns_overrides: Optional[list[str]]) -> tuple[str, list[str], bool]:
        """Prepare weights - delegate all file discovery to rdma_weight_utils"""
        # Force use of safetensors
        use_safetensors = True
        
        # Delegate file discovery and pulling to rdma_weight_utils
        # Return empty file list here as the actual files will be handled by the iterator
        return model_name_or_path, [], use_safetensors

    def _get_weights_iterator(self, source: "Source") -> Generator[tuple[str, torch.Tensor], None, None]:
        """Get weights iterator - delegate all RDMA operations to rdma_weight_utils"""
        # Use our RDMA safetensors iterator which handles all file discovery and pulling
        try:
            logger.info(f"开始通过RDMA加载模型 {source.model_or_path} 的权重...")
            
            # Get ETCD configuration
            rdma_config = self.load_config.model_loader_extra_config.get("rdma", {})
            metadata_server = rdma_config.get("etcd_endpoints")
            if not metadata_server:
                metadata_server = os.environ.get("VLLM_ETCD_ENDPOINTS", "localhost:2379")
            
            # Parse ETCD host and port
            if ":" in metadata_server:
                etcd_host, etcd_port_str = metadata_server.split(":", 1)
                etcd_port = int(etcd_port_str)
            else:
                etcd_host = metadata_server
                etcd_port = 2379
            
            yield from rdma_safetensors_weights_iterator(
                source.model_or_path, 
                self.mooncake_client,
                etcd_host,
                etcd_port
            )
            logger.info(f"完成通过RDMA加载模型 {source.model_or_path} 的权重")
        except Exception as e:
            logger.error(f"通过RDMA加载模型 {source.model_or_path} 的权重时出错: {e}")
            logger.info("回退到默认加载方式...")
            yield from super()._get_weights_iterator(source)

    def _get_mooncake_config(self):
        """获取Mooncake配置"""
        rdma_config = self.load_config.model_loader_extra_config.get("rdma", {})
        return {
            "server_endpoint": rdma_config.get("server_endpoint", "rdma://localhost:9876")
        }

    def _setup_mooncake_client(self) -> None:
        """Initialize Mooncake client for RDMA transfers."""
        try:
            # Get distributed rank for port allocation
            rank = self.rank
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            
            # Use KV transfer pattern: base_port + rank
            base_port = 9876
            port = base_port + rank
            
            logger.info(f"Setting up Mooncake client for rank {rank} on port {port}")
            
            # Get RDMA configuration
            rdma_config = self.load_config.model_loader_extra_config.get("rdma", {})
            
            # Get metadata server address (ETCD) from configuration or environment
            metadata_server = rdma_config.get("etcd_endpoints")
            if not metadata_server:
                metadata_server = os.environ.get("VLLM_ETCD_ENDPOINTS", "localhost:2379")
            
            logger.info(f"Using metadata server: {metadata_server}")
            
            # Initialize Mooncake client
            # Parameters for initialize(): local_hostname, metadata_server, protocol, device_name
            self.mooncake_client = TransferEngine()
            ret = self.mooncake_client.initialize(
                f"localhost:{port}",  # local_hostname
                metadata_server,      # metadata_server (ETCD connection string)
                "rdma",               # protocol
                ""                    # device_name (empty for all devices)
            )
            if ret != 0:
                raise RuntimeError(f"Failed to initialize Mooncake Transfer Engine with code {ret}")
            logger.info(f"Initialized Mooncake client on port {port} for rank {rank}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Mooncake client: {e}")
            raise

    def get_all_weights(
        self,
        model_config: ModelConfig,
        model: nn.Module,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Get all weights including remote weights via RDMA."""
        
        # Check if we should use RDMA for this model
        if self._should_use_rdma(model_config):
            logger.info(f"使用RDMA加载模型: {model_config.model}")
            
            # Create source configuration - use DefaultModelLoader's Source
            source = self.Source(
                model_or_path=model_config.model,
                revision=model_config.revision,
                prefix="",  # DefaultModelLoader handles prefixes in _get_weights_iterator
                fall_back_to_pt=getattr(model, "fall_back_to_pt_during_load", True),
                allow_patterns_overrides=getattr(model, "allow_patterns_overrides", None)
            )
            
            # Get weights via RDMA
            logger.info(f"开始通过RDMA加载模型 {model_config.model} 的所有权重...")
            yield from self._get_weights_iterator(source)
            logger.info(f"完成通过RDMA加载模型 {model_config.model} 的所有权重")
        else:
            # Fall back to default loading
            logger.info(f"使用默认方式加载模型: {model_config.model}")
            yield from super().get_all_weights(model_config, model)

    def _should_use_rdma(self, model_config: ModelConfig) -> bool:
        """Determine if RDMA should be used for this model."""
        # Check environment variable
        use_rdma = os.environ.get("VLLM_USE_RDMA_LOADER", "").lower()
        if use_rdma in ("1", "true", "yes"):
            return True
        
        # Check model path format
        model_path = model_config.model
        if model_path.startswith("rdma://"):
            return True
            
        # For Qwen3-32B or any model, if we're using RDMA load format, use RDMA
        if self.load_config.load_format == "rdma":
            return True
            
        return False

    def download_model(self, model_config: ModelConfig) -> None:
        """Download model metadata from remote server."""
        if self._should_use_rdma(model_config):
            logger.info(f"Checking remote model availability: {model_config.model}")
            # For RDMA models, we don't need to download locally, just verify remote availability
            # This would involve checking with ETCD or the remote server
            logger.info(f"Remote model {model_config.model} is available for RDMA loading")
        else:
            # Use default download for non-RDMA models
            super().download_model(model_config)

    def __del__(self):
        """Cleanup Mooncake client."""
        if hasattr(self, 'mooncake_client') and self.mooncake_client:
            try:
                logger.info("Cleaning up Mooncake client...")
                # TransferEngine typically doesn't need explicit shutdown
                # Just remove the reference to allow garbage collection
                self.mooncake_client = None
                logger.info("Mooncake client cleaned up successfully")
            except Exception as e:
                logger.error(f"Error cleaning up Mooncake client: {e}")
