"""TensorRT engines for the networks that are not YOLO models: the disc heat map.

The YOLO models are exported by Ultralytics (tensorrt_engines.py). A network of our own
is exported to ONNX in half precision and built into a strongly typed engine, for one
input shape, stored next to its weights (`best.1x9x544x960.fp16.engine`). Like the
others it is tied to this GPU and driver: rebuild after a driver or TensorRT update
(scripts/export_tensorrt.py --heatmap). Without an engine the network runs in PyTorch.
"""

from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

from ..config.settings import get_setting
from ..utils.logger import get_logger
from . import health

logger = get_logger("NETWORK_ENGINES")

WORKSPACE_BYTES = 2 << 30
_trt_logger = None  # TensorRT wants one logger for the whole process


def _logger() -> Any:
    global _trt_logger
    if _trt_logger is None:
        import tensorrt as trt

        _trt_logger = trt.Logger(trt.Logger.WARNING)
    return _trt_logger


def engine_path(weights: Path, input_shape: Sequence[int]) -> Path:
    """Where the engine of a network's weights for one input shape is kept."""
    shape = "x".join(str(int(size)) for size in input_shape)
    return Path(weights).with_name(f"{Path(weights).stem}.{shape}.fp16.engine")


def export_engine(model: Any, input_shape: Sequence[int], target: Path) -> Path:
    """Build the engine of a network (a torch module) for one input shape.

    The network is exported in half precision on the graphics card, as it runs there,
    and the engine keeps the types of the export.
    """
    import tensorrt as trt
    import torch

    model = model.to("cuda").half().eval()
    example = torch.zeros(tuple(input_shape), dtype=torch.float16, device="cuda")
    onnx_path = target.with_suffix(".onnx")
    with torch.no_grad():
        torch.onnx.export(
            model,
            example,
            str(onnx_path),
            input_names=["input"],
            output_names=["output"],
            opset_version=17,
            dynamo=False,
        )
    trt_logger = _logger()
    builder = trt.Builder(trt_logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, trt_logger)
    if not parser.parse(onnx_path.read_bytes()):
        errors = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"the ONNX export of the network does not parse: {errors}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE_BYTES)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT could not build the engine (see its messages above)")
    target.write_bytes(bytes(serialized))
    onnx_path.unlink(missing_ok=True)
    return target


class NetworkEngine:
    """A built engine, called like the network: a tensor of its input shape on the
    graphics card in, its output out."""

    def __init__(self, path: Path):
        import tensorrt as trt
        import torch

        self._torch = torch
        runtime = trt.Runtime(_logger())
        self._engine = runtime.deserialize_cuda_engine(Path(path).read_bytes())
        if self._engine is None:
            raise RuntimeError(f"{path} does not load (built for another GPU or TensorRT?)")
        self._context = self._engine.create_execution_context()
        names = [self._engine.get_tensor_name(i) for i in range(self._engine.num_io_tensors)]
        inputs = [n for n in names if self._engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        outputs = [n for n in names if n not in inputs]
        if len(inputs) != 1 or len(outputs) != 1:
            raise RuntimeError(f"{path}: one input and one output expected, not {names}")
        self._input, self._output = inputs[0], outputs[0]
        self.input_shape: Tuple[int, ...] = tuple(self._engine.get_tensor_shape(self._input))
        self._output_shape = tuple(self._engine.get_tensor_shape(self._output))
        types = {trt.DataType.HALF: torch.float16, trt.DataType.FLOAT: torch.float32}
        self._input_type = types[self._engine.get_tensor_dtype(self._input)]
        self._output_type = types[self._engine.get_tensor_dtype(self._output)]
        self._stream = None  # Its own CUDA stream: the default one makes TensorRT wait more

    def __call__(self, batch: Any) -> Any:
        torch = self._torch
        if tuple(batch.shape) != self.input_shape:
            raise ValueError(f"the engine takes {self.input_shape}, not {tuple(batch.shape)}")
        batch = batch.to(self._input_type).contiguous()
        out = torch.empty(self._output_shape, dtype=self._output_type, device=batch.device)
        self._context.set_tensor_address(self._input, batch.data_ptr())
        self._context.set_tensor_address(self._output, out.data_ptr())
        if self._stream is None:
            self._stream = torch.cuda.Stream(batch.device)
        caller = torch.cuda.current_stream(batch.device)
        self._stream.wait_stream(caller)  # The input is ready
        if not self._context.execute_async_v3(self._stream.cuda_stream):
            raise RuntimeError("the engine did not run")
        caller.wait_stream(self._stream)  # The output is, before it is read
        return out


def load_engine(weights: Path, input_shape: Sequence[int]) -> Optional[NetworkEngine]:
    """The engine of a network for an input shape, if one was built and engines are on
    (models.inference.tensorrt); None otherwise, and the network runs in PyTorch."""
    if not get_setting("models.inference.tensorrt", True):
        return None
    path = engine_path(weights, input_shape)
    if not path.is_file():
        return None
    try:
        engine = NetworkEngine(path)
    except Exception as error:
        logger.warning(f"{path.name} does not load, the network runs in PyTorch: {error}")
        health.report(
            f"Engine {path.name[:40]}", "does not load, the network runs slower", standing=True
        )
        return None
    logger.info(f"Loaded {path.name}")
    return engine
