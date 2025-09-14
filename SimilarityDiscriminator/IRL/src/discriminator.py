import torch
import torch.nn as nn

class StyleDiscriminator(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, num_layers: int = 2, dropout: float = 0.3):
        super(StyleDiscriminator, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM层: 负责处理时序信息
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Dropout层，正则化
        self.dropout = nn.Dropout(dropout)

        # 全连接层，最终输出映射到类别得分
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        定义模型的前向传播。

        Args:
            x (torch.Tensor): 输入的特征序列张量，形状为 (batch_size, sequence_length, input_dim)。

        Returns:
            torch.Tensor: 模型输出的原始得分 (logits)，形状为 (batch_size, num_classes)。
                          后续在计算损失时，会被CrossEntropyLoss隐式地进行Softmax。
        """
        # 初始化LSTM的hidden状态和cell状态
        # h0 和 c0 的形状: (num_layers, batch_size, hidden_dim)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # LSTM前向传播
        # lstm_out 形状: (batch_size, sequence_length, hidden_dim)
        # hidden 是一个元组 (h_n, c_n)，其中h_n是最后一个时间步的隐藏状态
        lstm_out, (h_n, c_n) = self.lstm(x, (h0, c0))

        # 我们通常使用最后一个时间步的隐藏状态作为整个序列的表示
        # h_n 的形状是 (num_layers, batch_size, hidden_dim)
        # 我们取最后一层 (h_n[-1]) 的输出
        last_hidden_state = h_n[-1]
        
        # 应用Dropout
        out = self.dropout(last_hidden_state)

        # 通过全连接层得到最终的类别得分
        logits = self.fc(out)

        return logits

# --- 这是一个用于测试和演示的模块 ---
if __name__ == '__main__':
    # --- 模拟参数 ---
    BATCH_SIZE = 16          # 批处理大小
    SEQ_LENGTH = 50          # 轨迹片段的长度 (例如, 50个时间步)
    ENCODER_OUTPUT_DIM = 128 # 模拟你的Encoder输出的特征维度
    NUM_STYLES = 3           # 要分类的风格数量 (例如, 激进, 保守, 战术)
    LSTM_HIDDEN_DIM = 256    # LSTM隐藏层大小
    LSTM_LAYERS = 2          # LSTM层数
    
    # --- 检查设备 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 实例化模型 ---
    print("\nInitializing model...")
    model = StyleDiscriminator(
        input_dim=ENCODER_OUTPUT_DIM,
        hidden_dim=LSTM_HIDDEN_DIM,
        num_classes=NUM_STYLES,
        num_layers=LSTM_LAYERS
    ).to(device)
    
    print(model)
    
    # --- 创建一个模拟的输入张量 ---
    # 模拟一个批次的，经过encoder处理后的轨迹数据
    dummy_input = torch.randn(BATCH_SIZE, SEQ_LENGTH, ENCODER_OUTPUT_DIM).to(device)
    print(f"\nShape of dummy input tensor: {dummy_input.shape}")

    # --- 前向传播测试 ---
    try:
        output = model(dummy_input)
        print(f"Shape of model output tensor (logits): {output.shape}")
        
        # 验证输出形状是否正确
        assert output.shape == (BATCH_SIZE, NUM_STYLES)
        print("\nModel forward pass successful!")
        
        # 打印一个样本的输出
        print(f"Example output for first sample in batch: {output[0]}")
        
    except Exception as e:
        print(f"\nAn error occurred during the forward pass: {e}")
          