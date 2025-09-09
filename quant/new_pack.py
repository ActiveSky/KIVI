import triton
import triton.language as tl
import random
import numpy as np
import torch

#=================== Quantize ===================
#K cache quantization
def quant_and_pack_kcache(k: torch.FloatTensor, group_size: int, bits: int):
	assert len(k.shape) == 4
	shape = k.shape
	B, nh, T, D = shape
	# ================== Get Scale & Zeros ===============
	assert T % group_size == 0
	num_groups = T // group_size
	new_shape = (B, nh, num_groups, group_size, D)
	# Quantize
	max_int = 2 ** bits - 1
	data = k.view(new_shape)
	mn = torch.min(data, dim=-2, keepdim=True)[0]
	mx = torch.max(data, dim=-2, keepdim=True)[0]
	scale =  (mx - mn) / max_int
	data = data - mn
	data.div_(scale)
	data = data.clamp_(0, max_int).round_().to(torch.int32)
	data = data.view(shape)
	code = pack_tensor(data, bits, pack_dim=2)
	return code, scale, mn

#V cache quantization
def quant_and_pack_vcache(v: torch.FloatTensor, group_size: int, bits: int):
	shape = v.shape
	assert len(shape) == 4
	assert v.shape[-1] % group_size == 0
	num_groups = shape[-1] // group_size
	new_shape = (shape[:-1] + (num_groups, group_size))
	# Quantize
	max_int = 2 ** bits - 1
	data = v.view(new_shape)
	mn = torch.min(data, dim=-1, keepdim=True)[0]
	mx = torch.max(data, dim=-1, keepdim=True)[0]
	scale = (mx - mn) / max_int
	data = data - mn
	data.div_(scale)
	data = data.clamp_(0, max_int).round_().to(torch.int32)
	data = data.view(shape)
	# Pack
	code = pack_tensor(data, bits, pack_dim=3)
	return code, scale, mn

#=================== Dequantize ===================
def unpack_and_dequant_kcache(k_code: torch.FloatTensor, 
							  scale: torch.FloatTensor, 
							  mn: torch.FloatTensor,
							  group_size: int, 
							  bits: int,
							  ):
	pack_dim = 2
	assert bits in [2, 4, 8]
	assert len(k_code.shape) == 4
	data = unpack_tensor(k_code, bits, pack_dim=pack_dim)
	shape = data.shape
	num_groups = shape[pack_dim] // group_size
	data = data.view(shape[:pack_dim] + (num_groups, group_size,) + shape[pack_dim+1:])
	data = data.to(torch.float16)
	data = data * scale + mn 
	return data.view(shape)

	
def unpack_and_dequant_vcache(v_code: torch.FloatTensor, 
							  scale: torch.FloatTensor, 
							  mn: torch.FloatTensor,
							  group_size: int, 
							  bits: int,
							  ):
	assert bits in [2, 4, 8]
	assert len(v_code.shape) == 4
	data = unpack_tensor(v_code, bits, pack_dim=3)
	shape = data.shape
	num_groups = shape[-1] // group_size
	data = data.view(shape[:-1] + (num_groups, group_size,))
	data = data.to(torch.float16)
	data = data * scale + mn 
	return data.view(shape)

#=================== Pack ===================
def pack_tensor(data, bits, pack_dim):
	# Pack
	shape = data.shape
	feat_per_int = 32 // bits
	assert bits in [2,4,8], "Only 2, 4, 8 bits are supported"
	assert shape[pack_dim] % feat_per_int == 0, "Dimension length must be divisible by number of features per int"
	# BS, nh, T, nd // 16 # 16 is for 2bit
	code = torch.zeros(shape[:pack_dim] + (shape[pack_dim] // feat_per_int,)+shape[pack_dim+1:], 
					dtype=torch.int32, 
					device=data.device)
	i = 0
	row = 0
	unpacked_indices = [slice(None)] * len(data.shape)
	packed_indices = [slice(None)] * len(data.shape)
	while row < code.shape[pack_dim]:
		packed_indices[pack_dim] = row
		for j in range(i, i + (32 // bits)):
			unpacked_indices[pack_dim] = j
			code[packed_indices] |= data[unpacked_indices] << (bits * (j - i))
		i += 32 // bits
		row += 1
	return code

#=================== Unpack ===================
def unpack_tensor(v_code: torch.FloatTensor, 
				  bits: int, 
				  pack_dim: int):
	assert bits in [2,4,8]
	shape = v_code.shape
	feat_per_int = 32 // bits
	new_shape = shape[:pack_dim] + (shape[pack_dim] * feat_per_int,) + shape[pack_dim+1:]
	unpacked_v_code = torch.zeros(new_shape, dtype=torch.int8, device=v_code.device)
	i = torch.arange(new_shape[pack_dim], device=v_code.device) // feat_per_int
	j = torch.arange(new_shape[pack_dim], device=v_code.device) % feat_per_int
	num = 0xFF >> (8 - bits)
	packed_indices = [slice(None)] * len(new_shape)
	packed_indices[pack_dim] = i
	if pack_dim == 2:
		unpacked_v_code = ((v_code[packed_indices] >> (j * bits)[None, None, :, None]).to(torch.int16)) & num
	elif pack_dim == 3:
		unpacked_v_code = ((v_code[packed_indices] >> (j * bits)).to(torch.int16)) & num
	else:
		raise NotImplementedError
	return unpacked_v_code


@triton.jit
def _pack_along_last_dim(
	bits: tl.constexpr,
	intensor_ptr,
	code_ptr,
	N,
	num_feats: tl.constexpr,
	feat_per_int: tl.constexpr,
	BLOCK_SIZE_N: tl.constexpr
):
	num_int_per_y_dim = num_feats // feat_per_int
	bid = tl.program_id(axis=0)
	yid = tl.program_id(axis=1)
	offs_N = bid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
	block_start = intensor_ptr + offs_N * num_feats + yid * feat_per_int # offset of the first element at current tile
	packed = tl.zeros((BLOCK_SIZE_N,), dtype=tl.int32)
	for i in range(feat_per_int):
		ptr = block_start + i
		element = tl.load(ptr, mask=offs_N<N, other=0.)
		element = element << (i * bits)
		# Combine the value using bitwise OR
		packed = packed | element
	tl.store(code_ptr + offs_N * num_int_per_y_dim + yid, packed, mask=offs_N < N)



@triton.jit
def _minmax_along_last_dim(
	x_ptr,        # 输入数据x的指针，指向需要计算最小值和最大值的数据
	mn_ptr, mx_ptr,  # 分别指向存储最小值和最大结果的指针
	total_elements: tl.constexpr,   # 输入数据中的总元素数量，编译时常量
	N: tl.constexpr,  # 组数，编译时常量
	num_groups: tl.constexpr,   # 分组数量，编译时常量
	group_size: tl.constexpr,  # 每组的大小，编译时常量
	BLOCK_SIZE_N: tl.constexpr  # 块大小N，编译时常量
):
	bid = tl.program_id(axis=0)  # 获取当前程序在axis=0维度上的ID
	offsets_b = bid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)  # 计算在B维度上的偏移量
	# 计算在数据中的总偏移量，使用广播机制生成二维偏移量数组
	offsets = offsets_b[:, None] * group_size + tl.arange(0, group_size)[None, :]
	# 创建掩码，确保偏移量不超过总元素数量
	mask = offsets < total_elements
	# 根据偏移量和掩码加载数据
	x = tl.load(x_ptr + offsets, mask=mask)
	# 沿着最后一个维度计算每组的最大值
	mx_val = tl.max(x, axis=1)
	# 沿着最后一个维度计算每组的最小值
	mn_val = tl.min(x, axis=1)
	# tl.device_print('shape', mn_val[:, None].shape)
	tl.store(mn_ptr+offsets_b, mn_val, mask=offsets_b<N*num_groups)
	tl.store(mx_ptr+offsets_b, mx_val, mask=offsets_b<N*num_groups)



def triton_quantize_and_pack_along_last_dim(data: torch.Tensor, group_size: int, bit: int):
<<<<<<< HEAD
	"""
	对4D张量沿最后一个维度进行量化和打包操作，使用Triton进行GPU加速优化。
	
	该函数是KIVI量化框架的核心组件，主要用于KV cache的量化和压缩，
	支持2-bit、4-bit、8-bit量化，通过分组量化和位打包技术显著减少显存占用。
	
	Args:
		data: 输入的4D张量，形状为(B, nh, D, T)
		group_size: 分组大小，用于分组量化
		bit: 量化位数，支持2、4、8位
		
	Returns:
		code: 打包后的量化数据，形状为(B, nh, D, T//feat_per_int)
		scale: 每组的缩放因子，形状为(B, nh, D, num_groups)
		mn: 每组的最小值，形状为(B, nh, D, num_groups)
	"""
	# 1. 输入验证和形状处理
	assert len(data.shape) == 4
	shape = data.shape
	B, nh, D, T = shape
	# ================== Get Scale & Zeros ===============
	assert T % group_size == 0
	num_groups = T // group_size
	new_shape = (B * nh * D, num_groups, group_size)
	scale_mn_shape = B, nh, D, num_groups
	
	# 2. 重塑数据为分组量化格式
	# 将数据重塑为(B*nh*D, num_groups, group_size)以便进行分组量化
	data = data.reshape(new_shape)
	
	# 3. 使用Triton内核并行计算每组的最小值和最大值
	# 预分配内存存储最值，避免动态内存分配开销
=======
	assert len(data.shape) == 4 #确保是4维张量
	shape = data.shape
	B, nh, D, T = shape
	# ================== Get Scale & Zeros ===============
	assert T % group_size == 0 #确保要量化的张量长度是group_size的整数倍
	num_groups = T // group_size # 计算分组数量
	new_shape = (B * nh * D, num_groups, group_size) # data要重新调整的新张量形状
	scale_mn_shape = B, nh, D, num_groups # scale和mn张量的形状
	# Quantize
	data = data.reshape(new_shape) # 调整data的形状
>>>>>>> a1570ef (docs: add Chinese comments to quantization functions)
	mx = torch.empty((B * nh * D, num_groups), device=data.device, dtype=data.dtype)
	mn = torch.empty((B * nh * D, num_groups), device=data.device, dtype=data.dtype)
	BLOCK_SIZE_N = 128
	# 配置网格：基于数据总量和块大小
	grid = lambda meta: (triton.cdiv(data.shape[0]*data.shape[1], BLOCK_SIZE_N),)
	with torch.cuda.device(data.device):
		# 调用Triton内核并行计算最值，使用8个warp最大化SM利用率
		_minmax_along_last_dim[grid](data, mn, mx,
							 data.numel(), data.shape[0], num_groups, group_size,
<<<<<<< HEAD
							 BLOCK_SIZE_N=BLOCK_SIZE_N, num_warps=8) 
	
	# 4. 量化过程
	# 计算缩放因子：(max - min) / (2^bit - 1)
	scale = (mx - mn) / (2 ** bit - 1)
	# 减去最小值进行零点偏移
=======
							 BLOCK_SIZE_N=BLOCK_SIZE_N, num_warps=8) #并行计算每组的最小值和最大值
	# mn = torch.min(data, dim=-1, keepdim=True)[0].squeeze(-1)
	# mx = torch.max(data, dim=-1, keepdim=True)[0].squeeze(-1)
	#量化的核心步骤
	scale = (mx - mn) / (2 ** bit - 1) #计算缩放因子
>>>>>>> a1570ef (docs: add Chinese comments to quantization functions)
	data = data - mn.unsqueeze(-1)
	# 应用缩放因子进行归一化
	data.div_(scale.unsqueeze(-1))
	# 限制在[0, 2^bit-1]范围内并取整，转换为int32
	data = data.clamp_(0, 2 ** bit - 1).round_().to(torch.int32)
	
	# 5. 位打包操作
	# 重塑数据为2D张量以便打包操作
	data = data.view(-1, T)
	# 计算每个32位整数能存储的特征数（例如：2-bit时为16个）
	feat_per_int = 32 // bit
	# 创建打包后的张量形状
	packshape = (np.prod(shape[:-1]), shape[-1] // feat_per_int,)
	# 预分配打包后的张量
	code = torch.zeros(*packshape, device=data.device, dtype=torch.int32)
	# 配置打包网格
	grid = lambda meta: (triton.cdiv(data.shape[0], BLOCK_SIZE_N), data.shape[1] // feat_per_int,)
	with torch.cuda.device(data.device):
		# 调用Triton内核进行位打包，将多个低位值打包到单个32位整数中
		_pack_along_last_dim[grid](bit, data, code, data.shape[0], 
								data.shape[1], feat_per_int, 
								BLOCK_SIZE_N=BLOCK_SIZE_N, 
								num_warps=8)
	
	# 6. 返回打包后的数据和量化参数
	# 将打包后的数据重塑为原始4D形状（除了最后一个维度）
	# 返回缩放因子和最小值用于后续反量化
	return code.view(B, nh, D, -1), scale.reshape(scale_mn_shape), mn.reshape(scale_mn_shape)
	
