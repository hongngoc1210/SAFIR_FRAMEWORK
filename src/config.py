import torch


class Config:

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Task / labels
    # binary : raw <= 0 -> 0 DOWN, raw > 0 -> 1 UP
    # ternary: raw < 0 -> 0 NEGATIVE, raw == 0 -> 1 NEUTRAL, raw > 0 -> 2 POSITIVE
    label_mode = "binary"
    n_classes = 2
    label_names = ("DOWN", "UP")

    # Module I: Stock-aware daily news pooling
    bert_model = "bert-base-chinese"
    bert_dim = 768
    code_emb_dim = 64
    news_factor_dim = 256
    sap_heads = 8  # must divide bert_dim
    max_text_len = 128

    # Daily news collection support. If text_a contains multiple articles, they are
    # split by common separators; otherwise the model falls back to a single article.
    max_news_per_day = 8
    news_pooling = "sap"  # choices: "cap", "sap", "pa_sap"

    # Module II: News-price fusion + market-gated forecasting
    d_model = 256
    n_heads = 8
    d_ff = 1024
    dropout = 0.1
    n_layers = 3
    lookback = 20
    mkt_dim = 2  # mean_ret, std_ret
    price_feat_dim = 2  # open, close by default

    use_bidirectional_fusion = True
    causal_kernel_size = 5
    graph_dropout = 0.1
    egarch_dim = 64  # supervised risk-head hidden dim, not a full statistical EGARCH fit
    # Risk supervision

    risk_loss_weight = 0.05
    vol_loss_weight = 0.50
    var_loss_weight = 0.50
    var_confidence_z = 1.65

    # Imbalance handling
    use_class_weights = True
    class_weight_power = 0.50  # sqrt inverse frequency; safer than full inverse freq

    # Module III: SEP-style report generation
    enable_sefn = False
    llm_model = "gpt2"
    llm_hidden = 768
    max_explain_len = 128
    enable_ppo = False
    ppo_clip = 0.2
    ppo_epochs = 4
    gamma = 0.99

    # Training
    lr = 1e-4
    batch_size = 16
    fp16 = False
    weight_decay = 1e-2
    grad_clip = 1.0
    grad_checkpointing = False
