import numpy as np
import onnx


def get_tensor_value_info(tensor_info: onnx.TensorProto):
    name = tensor_info.name

    shape = None
    elem_type = None
    if tensor_info.HasField("type"):
        shape = []
        tensor_type = tensor_info.type.tensor_type
        elem_type = tensor_type.elem_type
        for d in tensor_type.shape.dim:
            # the dimension may have a definite (integer) value or a symbolic identifier or neither:
            if d.HasField("dim_value"):  # known dimension
                shape.append(int(d.dim_value))
            elif d.HasField("dim_param"):  # unknown dimension with symbolic name
                shape.append(str(d.dim_param))
            else:  # unknown dimension with no name
                shape.append(-1)
    return {"name": name, "shape": shape, "tensor_type": _elem_type_as_numpy(elem_type)}


def _elem_type_as_numpy(elem_type):
    if elem_type == onnx.TensorProto.FLOAT:
        return np.dtype("float32")
    if elem_type == onnx.TensorProto.INT32:
        return np.dtype("int32")
    if elem_type == onnx.TensorProto.UINT32:
        return np.dtype("uint32")
    if elem_type == onnx.TensorProto.UINT64:
        return np.dtype("uint64")
    if elem_type == onnx.TensorProto.INT16:
        return np.dtype("int16")
    if elem_type == onnx.TensorProto.UINT16:
        return np.dtype("uint16")
    if elem_type == onnx.TensorProto.UINT8:
        return np.dtype("uint8")
    if elem_type == onnx.TensorProto.INT8:
        return np.dtype("int8")
    if elem_type == onnx.TensorProto.FLOAT16:
        return np.dtype("float16")
    raise NotImplementedError(f"Currently doesn't support type: '{elem_type}'.")
