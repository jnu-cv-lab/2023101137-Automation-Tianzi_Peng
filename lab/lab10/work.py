import torch
import math

def get_sinusoidal_pe(seq_len, d_model):
    """生成传统的 Sinusoidal 位置编码矩阵"""
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    
    pe[:, 0::2] = torch.sin(position * div_term) 
    pe[:, 1::2] = torch.cos(position * div_term) 
    return pe

def rotate_2d(vector, theta):
    """通过旋转矩阵对二维向量进行旋转"""
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    rot_matrix = torch.tensor([
        [cos_theta, -sin_theta],
        [sin_theta,  cos_theta]
    ])
    return torch.matmul(rot_matrix, vector)

def apply_rope(x, pos):
    """
    对高维向量 x 应用 RoPE。
    假设 x 的维度 d_model 是偶数。
    """
    d_model = x.shape[0]
    out = torch.zeros_like(x)
    
    for i in range(d_model // 2):

        theta = pos / (10000.0 ** (2 * i / d_model))
        cos_th = math.cos(theta)
        sin_th = math.sin(theta)
        
        x_1 = x[2 * i]
        x_2 = x[2 * i + 1]
        out[2 * i]     = x_1 * cos_th - x_2 * sin_th
        out[2 * i + 1] = x_1 * sin_th + x_2 * cos_th
        
    return out

def run_numerical_experiment():
    print("--- 实验：RoPE 与 E+pos 的数值验证 ---")
    d_model = 16
    
    torch.manual_seed(42)
    q = torch.randn(d_model)
    k = torch.randn(d_model)
    
    pos_m, pos_n = 2, 5
    
    pe_matrix = get_sinusoidal_pe(10, d_model)
    pe_m = pe_matrix[pos_m]
    pe_n = pe_matrix[pos_n]
    
    q_abs = q + pe_m
    k_abs = k + pe_n

    score_abs_orig = torch.dot(q_abs, k_abs)
    
    q_abs_shifted = q + pe_matrix[pos_m + 3]
    k_abs_shifted = k + pe_matrix[pos_n + 3]
    score_abs_shifted = torch.dot(q_abs_shifted, k_abs_shifted)
    
    print(f"[E+pos]   位置 (2,5) 的点积: {score_abs_orig:.4f}")
    print(f"[E+pos]   位置 (5,8) 的点积: {score_abs_shifted:.4f}")
    print(f"          结论: E+pos 的点积随绝对位置变化而剧烈改变。\n")
    
    q_rope = apply_rope(q, pos_m)
    k_rope = apply_rope(k, pos_n)
    score_rope_orig = torch.dot(q_rope, k_rope)
    
    q_rope_shifted = apply_rope(q, pos_m + 3)
    k_rope_shifted = apply_rope(k, pos_n + 3)
    score_rope_shifted = torch.dot(q_rope_shifted, k_rope_shifted)
    
    print(f"[RoPE]    位置 (2,5) 的点积: {score_rope_orig:.4f}")
    print(f"[RoPE]    位置 (5,8) 的点积: {score_rope_shifted:.4f}")
    print(f"          结论: 只要相对距离 (m-n) 不变，RoPE 的点积完全相等！")

if __name__ == '__main__':
    run_numerical_experiment()